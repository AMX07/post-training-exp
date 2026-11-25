"""CS336 Spring 2026 on-policy RFT and GRPO, with separate vLLM inference GPU.

Defaults implement the assignment's suggested experiment, not the older nanochat
adaptation. No execution occurs on import. Launch through the Modal wrapper.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid

import torch
import torch.nn.functional as F

MODEL = 'allenai/OLMo-2-0425-1B'
MODEL_REVISION = 'a1847dff35000b4271fa70afc5db10fd29fedbdf'
DATA_REVISION = '740312add88f781978c0658806c59bc2815b9866'
PROMPT_PATH = Path(__file__).parent / 'vendor/r1_zero.prompt'
GRADER_REVISION = 'c2734a26308710949fe13226960a1e8cece94b7e'
SOURCE_FILES = ('assignment_train.py', 'vendor/assignment_vllm_utils.py',
                'vendor/drgrpo_grader.py', 'vendor/r1_zero.prompt', 'requirements-assignment.txt')


def source_fingerprints():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


@dataclass(frozen=True)
class AssignmentConfig:
    mode: str = 'rft'
    seed: int = 0
    n_train_examples: int = 6400
    n_val_examples: int = 1024
    num_rollout_steps: int = 200
    rollout_batch_size: int = 256  # responses, not questions
    group_size: int = 8
    microbatch_size: int = 8  # 32 accumulation microbatches by default
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    advantage_eps: float = 1e-6
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    sampling_max_tokens: int = 512
    normalization_constant: int = 512  # L; entire RFT denominator is B*G*L
    eval_every: int = 10
    eval_group_size: int = 1
    eval_prompt_batch_size: int = 32
    checkpoint_every: int = 10
    max_model_len: int = 4096
    # Python 3.13 environment: SDPA is the documented portability adaptation.
    attention_implementation: str = 'sdpa'
    vllm_gpu_memory_utilization: float = 0.85
    vllm_port: int = 18080

    def validate(self):
        if self.mode not in ('rft', 'grpo'):
            raise ValueError('mode must be rft or grpo')
        for key in ('n_train_examples', 'n_val_examples', 'num_rollout_steps',
                    'rollout_batch_size', 'group_size', 'microbatch_size',
                    'sampling_max_tokens', 'normalization_constant', 'eval_every',
                    'eval_group_size', 'eval_prompt_batch_size', 'checkpoint_every'):
            if getattr(self, key) < 1:
                raise ValueError(f'{key} must be positive')
        if self.rollout_batch_size % self.group_size or self.rollout_batch_size % self.microbatch_size:
            raise ValueError('rollout batch must divide into whole groups and microbatches')
        if self.group_size < 2 and self.mode == 'grpo':
            raise ValueError('GRPO sample standard deviation requires group_size >= 2')
        if self.n_train_examples < self.rollout_batch_size // self.group_size:
            raise ValueError('training subset must contain at least one prompt batch')


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def persist_rollouts(directory: Path, phase: str, step: int, records: list[dict], commit):
    """Fail closed: no optimizer update may follow a failed persistence commit."""
    path = directory / f'{phase}-step-{step:06d}.json'
    if path.exists():
        raise FileExistsError(f'Refusing to replace rollout artifact: {path}')
    atomic_json(path, {'phase': phase, 'step': step, 'records': records})
    commit()
    return path


def compute_advantages(rewards, group_size, mode='rft', eps=1e-6):
    if rewards.ndim != 1 or len(rewards) % group_size:
        raise ValueError('rewards must consist of whole contiguous groups')
    if mode == 'rft':
        return rewards.clone()
    if mode != 'grpo' or group_size < 2:
        raise ValueError('unsupported mode or invalid GRPO group size')
    groups = rewards.reshape(-1, group_size)
    return ((groups - groups.mean(dim=1, keepdim=True)) /
            (groups.std(dim=1, keepdim=True, correction=1) + eps)).flatten()


def pack_token_sequences(prompt_ids, response_ids, pad_id, device='cpu'):
    if not prompt_ids or len(prompt_ids) != len(response_ids):
        raise ValueError('prompt/response batches must be nonempty and aligned')
    if any(not ids for ids in prompt_ids):
        raise ValueError('each prompt must contain at least one token')
    width = max(len(p) + len(r) for p, r in zip(prompt_ids, response_ids))
    width = max(width, 2)
    tokens = torch.full((len(prompt_ids), width), pad_id, dtype=torch.long, device=device)
    targets = torch.full_like(tokens, -100)
    mask = torch.zeros_like(tokens)
    for i, (prompt, response) in enumerate(zip(prompt_ids, response_ids)):
        seq = prompt + response
        tokens[i, :len(seq)] = torch.tensor(seq, device=device)
        mask[i, :len(seq)] = 1
        targets[i, len(prompt):len(seq)] = tokens[i, len(prompt):len(seq)]
    return tokens[:, :-1], targets[:, 1:], mask[:, :-1]


def loss_from_logits(logits, targets, advantages, mode, total_rollouts, normalization_constant=512):
    if total_rollouts < 1 or normalization_constant < 1:
        raise ValueError('normalization denominators must be positive')
    nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                          targets.reshape(-1), ignore_index=-100,
                          reduction='none').reshape_as(targets)
    weighted = nll * advantages[:, None]
    if mode == 'rft':
        return weighted.sum() / (total_rollouts * normalization_constant)
    if mode == 'grpo':
        return (weighted.sum(dim=1) / (targets != -100).sum(dim=1).clamp_min(1)).sum() / total_rollouts
    raise ValueError('unsupported mode')


def response_entropy(logits, targets):
    """Sum and count, permitting correctly weighted aggregation across microbatches."""
    with torch.no_grad():
        values = logits.float()
        logz = torch.logsumexp(values, dim=-1)
        entropy = logz - (values.softmax(dim=-1) * values).sum(dim=-1)
        mask = targets != -100
        return float(entropy[mask].sum()), int(mask.sum())


def train_update(model, optimizer, records, tokenizer, cfg):
    """One update over the original rollout batch; prune zero advantages exactly."""
    if len(records) != cfg.rollout_batch_size:
        raise ValueError('unexpected rollout batch size')
    device = next(model.parameters()).device
    rewards = torch.tensor([r['reward'] for r in records], dtype=torch.float32, device=device)
    advantages = compute_advantages(rewards, cfg.group_size, cfg.mode, cfg.advantage_eps)
    optimizer.zero_grad(set_to_none=True)
    selected = [i for i, advantage in enumerate(advantages.tolist()) if advantage != 0.0]
    loss_total, entropy_sum, entropy_tokens = 0., 0., 0
    model.train()
    # Dropout would make the re-evaluated on-policy probabilities inconsistent.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    for offset in range(0, len(selected), cfg.microbatch_size):
        indices = selected[offset:offset + cfg.microbatch_size]
        subset = [records[i] for i in indices]
        inputs, targets, attention = pack_token_sequences(
            [r['prompt_token_ids'] for r in subset],
            [r['completion_token_ids'] for r in subset], tokenizer.pad_token_id, device)
        logits = model(inputs, attention_mask=attention, use_cache=False).logits
        loss = loss_from_logits(logits, targets, advantages[indices], cfg.mode,
                                len(records), cfg.normalization_constant)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        entropy, tokens = response_entropy(logits, targets)
        entropy_sum += entropy
        entropy_tokens += tokens
        loss.backward()
        loss_total += float(loss.detach())
        del logits, loss
    grad_norm = 0.
    if selected:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm,
                                                        error_if_nonfinite=True))
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(loss=loss_total, grad_norm=grad_norm,
                token_entropy=entropy_sum / entropy_tokens if entropy_tokens else None,
                entropy_scope='nonzero_advantage_responses',
                entropy_token_count=entropy_tokens, trained_responses=len(selected),
                optimizer_update_applied=bool(selected),
                zero_advantage_fraction=float((advantages == 0).float().mean()))


def rollout_metrics(records, group_size):
    if not records or len(records) % group_size:
        raise ValueError('metrics require complete groups')
    result = {key: sum(r[key] for r in records) / len(records)
              for key in ('reward', 'format_reward', 'answer_reward')}
    result['response_length'] = sum(len(r['completion_token_ids']) for r in records) / len(records)
    result['length_stop_fraction'] = sum(r['finish_reason'] == 'length' for r in records) / len(records)
    groups = [records[i:i + group_size] for i in range(0, len(records), group_size)]
    for k in range(1, group_size + 1):
        result[f'pass@{k}'] = sum(any(r['reward'] > 0 for r in group[:k]) for group in groups) / len(groups)
    result['zero_reward_group_fraction'] = sum(not any(r['reward'] for r in group) for group in groups) / len(groups)
    return result


def generate_records(server, tokenizer, rows, cfg, group_size, seed, step, phase):
    from vendor.drgrpo_grader import r1_zero_reward_fn
    template = PROMPT_PATH.read_text()
    prompts = [template.format(question=row['question']) for row in rows]
    prompt_tokens = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    if any(len(p) + cfg.sampling_max_tokens > cfg.max_model_len for p in prompt_tokens):
        raise ValueError('Prompt plus generation exceeds context; refusing silent truncation')
    sampling = dict(temperature=cfg.sampling_temperature, top_p=cfg.sampling_top_p,
                    top_k=-1, max_tokens=cfg.sampling_max_tokens, n=group_size,
                    seed=seed, stop=['</answer>'], include_stop_str_in_output=True)
    completions = server.generate_completions(prompts, sampling)
    if len(completions) != len(rows) * group_size:
        raise RuntimeError('vLLM returned an incomplete rollout batch')
    records = []
    for i, completion in enumerate(completions):
        group, sample = divmod(i, group_size)
        row = rows[group]
        if '####' not in row['answer']:
            raise ValueError('GSM8K answer lacks expected final-answer delimiter')
        answer = row['answer'].split('####')[-1].strip()
        text = completion.text
        scores = r1_zero_reward_fn(text, answer)
        # Assignment §4.2.1: tokenize prompt and completion TEXT separately;
        # retain original sampled IDs too, rather than silently losing them.
        response_ids = tokenizer.encode(text, add_special_tokens=False)
        records.append(dict(phase=phase, step=step, dataset_index=row['dataset_index'],
                            sample_index=sample, seed=seed, question=row['question'],
                            ground_truth=row['answer'], expected_answer=answer,
                            prompt=prompts[group], text=text,
                            prompt_token_ids=prompt_tokens[group],
                            completion_token_ids=response_ids,
                            sampled_token_ids=list(completion.token_ids),
                            finish_reason=completion.finish_reason,
                            sampling_parameters=sampling, **scores))
    if phase == 'train':
        advantages = compute_advantages(torch.tensor([r['reward'] for r in records]),
                                        group_size, cfg.mode, cfg.advantage_eps).tolist()
        for record, advantage in zip(records, advantages):
            record['advantage'] = advantage
            record['used_for_update'] = advantage != 0.0
    return records


def checkpoint(root, model, tokenizer, optimizer, cfg, next_step, run_id, commit):
    destination = root / 'checkpoints' / f'step-{next_step:06d}'
    if destination.exists():
        raise FileExistsError(f'Checkpoint already exists: {destination}')
    temporary = destination.with_name(destination.name + '-' + uuid.uuid4().hex + '.tmp')
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    torch.save(dict(config=asdict(cfg), next_step=next_step, run_id=run_id,
                    optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), python_rng=random.getstate(),
                    model_revision=MODEL_REVISION, data_revision=DATA_REVISION,
                    source_fingerprints=source_fingerprints()), temporary / 'trainer.pt')
    atomic_json(temporary / 'COMPLETE', {'next_step': next_step})
    temporary.rename(destination)
    atomic_json(root / 'latest.json', {'checkpoint': str(destination.relative_to(root))})
    commit()
    return str(destination)


def run(run_name, output_dir, cfg=None, *, resume=False, commit=lambda: None):
    run_started = time.monotonic()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vendor.assignment_vllm_utils import VLLMServer
    import wandb
    cfg = cfg or AssignmentConfig()
    cfg.validate()
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two visible GPUs required: policy cuda:0, vLLM cuda:1')
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Attach wandb_sec containing WANDB_API_KEY')
    root = Path(output_dir)
    state, source, source_kwargs, start = None, MODEL, {'revision': MODEL_REVISION}, 0
    if resume:
        source = root / json.loads((root / 'latest.json').read_text())['checkpoint']
        if not (source / 'COMPLETE').exists():
            raise RuntimeError('Checkpoint is incomplete')
        state = torch.load(source / 'trainer.pt', map_location='cpu', weights_only=False)
        if (state['config'] != asdict(cfg) or state['model_revision'] != MODEL_REVISION
                or state['data_revision'] != DATA_REVISION
                or state.get('source_fingerprints') != source_fingerprints()):
            raise ValueError('Resume configuration or source revision mismatch')
        start, source_kwargs = state['next_step'], {}
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new run name or resume; refusing to overwrite existing artifacts')
    root.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(source), torch_dtype=torch.bfloat16,
                                                attn_implementation=cfg.attention_implementation,
                                                **source_kwargs).to('cuda:0')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                 betas=(0.9, 0.95), weight_decay=0.)
    if state:
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        del state['optimizer']
    dataset = load_dataset('openai/gsm8k', 'main', revision=DATA_REVISION)
    train = dataset['train'].add_column('dataset_index', list(range(len(dataset['train'])))).shuffle(seed=cfg.seed).select(range(cfg.n_train_examples))
    val = dataset['test'].add_column('dataset_index', list(range(len(dataset['test'])))).shuffle(seed=cfg.seed).select(range(cfg.n_val_examples))
    run_id = state['run_id'] if state else wandb.util.generate_id()
    packages = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'vllm', 'datasets', 'wandb')}
    provenance = dict(config=asdict(cfg), model=MODEL, model_revision=MODEL_REVISION,
                      data_revision=DATA_REVISION, grader_revision=GRADER_REVISION,
                      prompt=PROMPT_PATH.read_text(), packages=packages,
                      source_fingerprints=source_fingerprints(),
                      gpu_names=[torch.cuda.get_device_name(i) for i in range(2)])
    atomic_json(root / 'config.json', provenance)
    attempt = root / 'attempts' / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    for filename in SOURCE_FILES:
        target = attempt / 'source' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parent / filename, target)
    atomic_json(attempt / 'provenance.json', provenance)
    commit()
    wb = wandb.init(project=os.environ.get('WANDB_PROJECT', 'olmo-gsm8k-assignment'),
                    entity=os.environ.get('WANDB_ENTITY') or None, name=run_name,
                    id=run_id, resume='must' if state else 'never', config=provenance,
                    dir=str(root))
    server = VLLMServer(model_id=str(source), revision=source_kwargs.get('revision'),
                        tokenizer_revision=MODEL_REVISION if source_kwargs else None,
                        gpu=1, seed=cfg.seed, port=cfg.vllm_port,
                        max_model_len=cfg.max_model_len,
                        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                        server_log_path=str(attempt / 'vllm-server.log'),
                        launch_server=True)
    try:
        vllm_started = time.monotonic()
        server.start()
        server.init_weight_sync('cuda:0')
        startup_metrics = {'timing/vllm_startup_seconds': time.monotonic() - vllm_started,
                           'timing/startup_seconds': time.monotonic() - run_started}
        atomic_json(attempt / 'metrics' / 'startup.json', startup_metrics)
        commit()
        wb.summary.update(startup_metrics)
        for step in range(start, cfg.num_rollout_steps):
            began = time.monotonic()
            # Synchronize current weights and invalidate old inference caches before
            # every on-policy generation phase, including validation.
            server.sync_policy_weights(model)
            metrics = {'step': step, 'learning_rate': cfg.learning_rate,
                       'timing/weight_sync_seconds': time.monotonic() - began,
                       'timing/evaluation_seconds': 0., 'timing/checkpoint_seconds': 0.}
            if step % cfg.eval_every == 0:
                evaluation_started = time.monotonic()
                validation = []
                for offset in range(0, len(val), cfg.eval_prompt_batch_size):
                    rows = [val[i] for i in range(offset, min(offset + cfg.eval_prompt_batch_size, len(val)))]
                    records = generate_records(server, tokenizer, rows, cfg, cfg.eval_group_size,
                                               cfg.seed + 1000000 + step * len(val) + offset, step, 'eval')
                    # Chunk evaluation for durable visibility before all 1024 finish.
                    persist_rollouts(attempt / 'rollouts', f'eval-{offset:06d}', step, records, commit)
                    validation.extend(records)
                metrics.update({f'val/{k}': v for k, v in rollout_metrics(validation, cfg.eval_group_size).items()})
                metrics['timing/evaluation_seconds'] = time.monotonic() - evaluation_started
            prompts_per_batch = cfg.rollout_batch_size // cfg.group_size
            indices = [(step * prompts_per_batch + i) % len(train) for i in range(prompts_per_batch)]
            generation_started = time.monotonic()
            records = generate_records(server, tokenizer, [train[i] for i in indices], cfg,
                                       cfg.group_size, cfg.seed + step, step, 'train')
            metrics['timing/generation_and_grading_seconds'] = time.monotonic() - generation_started
            persistence_started = time.monotonic()
            persist_rollouts(attempt / 'rollouts', 'train', step, records, commit)
            metrics['timing/rollout_persistence_seconds'] = time.monotonic() - persistence_started
            torch.cuda.synchronize(0)
            update_started = time.monotonic()
            metrics.update({f'train/{k}': v for k, v in rollout_metrics(records, cfg.group_size).items()})
            metrics.update({f'train/{k}': v for k, v in train_update(model, optimizer, records, tokenizer, cfg).items()})
            torch.cuda.synchronize(0)
            metrics['timing/update_seconds'] = time.monotonic() - update_started
            if (step + 1) % cfg.checkpoint_every == 0 or step + 1 == cfg.num_rollout_steps:
                checkpoint_started = time.monotonic()
                wb.summary['latest_checkpoint'] = checkpoint(root, model, tokenizer, optimizer,
                                                             cfg, step + 1, run_id, commit)
                metrics['timing/checkpoint_seconds'] = time.monotonic() - checkpoint_started
            metrics['timing/step_seconds'] = time.monotonic() - began
            metrics['train/peak_gpu_gb'] = torch.cuda.max_memory_allocated(0) / 1e9
            atomic_json(attempt / 'metrics' / f'step-{step:06d}.json', metrics)
            commit()
            wb.log(metrics, step=step)
            print(json.dumps(metrics), flush=True)
        # Evaluate the FINAL updated policy, not just the policy before its last update.
        final_evaluation_started = time.monotonic()
        server.sync_policy_weights(model)
        validation = []
        for offset in range(0, len(val), cfg.eval_prompt_batch_size):
            rows = [val[i] for i in range(offset, min(offset + cfg.eval_prompt_batch_size, len(val)))]
            records = generate_records(server, tokenizer, rows, cfg, cfg.eval_group_size,
                                       cfg.seed + 1000000 + cfg.num_rollout_steps * len(val) + offset,
                                       cfg.num_rollout_steps, 'eval')
            persist_rollouts(attempt / 'rollouts', f'eval-final-{offset:06d}',
                              cfg.num_rollout_steps, records, commit)
            validation.extend(records)
        final_metrics = {f'val/{k}': v for k, v in rollout_metrics(validation, cfg.eval_group_size).items()}
        final_metrics['step'] = cfg.num_rollout_steps
        final_metrics['timing/final_evaluation_seconds'] = time.monotonic() - final_evaluation_started
        final_metrics['timing/total_workload_seconds'] = time.monotonic() - run_started
        atomic_json(attempt / 'metrics' / 'final.json', final_metrics)
        commit()
        wb.log(final_metrics, step=cfg.num_rollout_steps)
        wb.summary['completed'] = True
    finally:
        try:
            server.stop()
        finally:
            try:
                wb.finish()
            finally:
                commit()
    return dict(output_dir=str(root), run_id=run_id, completed_steps=cfg.num_rollout_steps,
                total_workload_seconds=time.monotonic() - run_started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=['rft', 'grpo'], default='rft')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.run_name, args.output_dir, AssignmentConfig(mode=args.mode, seed=args.seed), resume=args.resume)
