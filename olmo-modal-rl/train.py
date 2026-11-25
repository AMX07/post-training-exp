"""OLMo adaptation of nanochat chat_rl.py; see PARITY.md for exact differences."""
import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
import uuid

import torch
import torch.nn.functional as F
from vendor.nanochat_optim import MuonAdamW

UPSTREAM = '92d63d4e8bb4df75c3b71618f31ddde2378b2bcd'
MODEL = 'allenai/OLMo-2-0425-1B'
MODEL_REVISION = 'a1847dff35000b4271fa70afc5db10fd29fedbdf'
CS336_COMMIT = 'c2734a26308710949fe13226960a1e8cece94b7e'
PROMPT_TEMPLATE = (Path(__file__).parent / 'vendor/r1_zero.prompt').read_text()
DATA_REVISION = '740312add88f781978c0658806c59bc2815b9866'

@dataclass
class Config:
    num_epochs: int = 1
    device_batch_size: int = 8
    examples_per_step: int = 16
    num_samples: int = 16
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.02
    weight_decay: float = 0.0
    init_lr_frac: float = 0.05
    eval_every: int = 60
    eval_examples: int = 400
    save_every: int = 60
    # OLMo-only normalization parameters: explicit adaptation, not an upstream default.
    norm_lr: float = 0.005

def grade_completion(answer, completion):
    from vendor.drgrpo_grader import r1_zero_reward_fn
    if '####' not in answer:
        raise ValueError('GSM8K ground truth has no #### answer')
    expected = answer.split('####')[-1].strip()
    return r1_zero_reward_fn(completion, expected)

def reward(answer, completion):
    return grade_completion(answer, completion)['reward']

def prompt(question):
    return PROMPT_TEMPLATE.format(question=question)

def optimizer_groups(model, cfg):
    scale = (model.config.hidden_size / 768) ** -0.5
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    if emb is head:
        raise ValueError('This mapping requires untied embeddings, as in OLMo-2-0425-1B')
    matrices, norms = [], []
    for p in model.parameters():
        if p is emb or p is head:
            continue
        (matrices if p.ndim == 2 else norms).append(p)
    groups = [
        dict(name='lm_head', kind='adamw', params=[head], lr=cfg.unembedding_lr * scale,
             betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        dict(name='embedding', kind='adamw', params=[emb], lr=cfg.embedding_lr * scale,
             betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
    ]
    if norms:
        groups.append(dict(name='olmo_norms', kind='adamw', params=norms, lr=cfg.norm_lr,
                           betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
    for shape in sorted({p.shape for p in matrices}):
        groups.append(dict(name='matrix_' + 'x'.join(map(str, shape)), kind='muon',
                           params=[p for p in matrices if p.shape == shape], lr=cfg.matrix_lr,
                           momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=cfg.weight_decay))
    grouped = [id(p) for g in groups for p in g['params']]
    assert len(grouped) == len(set(grouped)) == len(list(model.parameters()))
    for group in groups:
        group['lr'] *= cfg.init_lr_frac
        group['initial_lr'] = group['lr']
    return groups

def pack_sequences(sequences, prefix_length, pad_id, device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        mask[i, prefix_length:len(seq)] = True
    targets = ids[:, 1:].clone()
    targets[~mask[:, 1:]] = -1
    return ids[:, :-1], targets

def policy_loss(logits, targets, advantages, num_passes, examples_per_step):
    # Deliberately match upstream PER-MICROBATCH valid-token normalization.
    nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                          ignore_index=-1, reduction='none').view_as(targets)
    return (nll * advantages[:, None]).sum() / (
        (targets >= 0).sum().clamp(min=1) * num_passes * examples_per_step)

@torch.no_grad()
def sample(model, tokenizer, question, cfg, count, seed=42):
    from transformers import GenerationConfig
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt(question), add_special_tokens=False)
    if len(tokens) + cfg.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError('Prompt plus completion exceeds model context; refusing silent truncation')
    input_ids = torch.tensor([tokens] * count, device=device)
    generation = GenerationConfig(
        do_sample=True, temperature=cfg.temperature, top_k=cfg.top_k, top_p=1.0,
        max_new_tokens=cfg.max_new_tokens, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
        repetition_penalty=1.0, use_cache=True, stop_strings=["</answer>"],
    )
    # Upstream's Engine uses an independent seeded generator per sampling call.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(seed)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            result = model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                                    generation_config=generation, tokenizer=tokenizer)
    sequences = []
    for row in result.tolist():
        tail = row[len(tokens):]
        if tokenizer.eos_token_id in tail:
            tail = tail[:tail.index(tokenizer.eos_token_id)]
        sequences.append(tokens + tail)
    return sequences, len(tokens)

def preserve_rollouts(directory, phase, step, index, row, sequences, prefix, texts, grades, cfg, commit):
    """Persist every verifier input before training, then commit the mounted Volume.

    Tokens are the post-EOS/padding-trimmed sequences consumed by the trainer.
    A unique directory per invocation keeps resumed attempts from overwriting data.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{phase}-step-{step:06d}-example-{index:06d}.json'
    payload = dict(phase=phase, step=step, dataset_index=index,
                   question=row['question'], ground_truth=row['answer'],
                   prompt=prompt(row['question']), prefix_length=prefix,
                   config=asdict(cfg), model_revision=MODEL_REVISION,
                   dataset_revision=DATA_REVISION, grader_revision=CS336_COMMIT,
                   completions=[dict(sample_index=i, text=text, token_ids=seq,
                                     completion_token_ids=seq[prefix:], grade=grade)
                                for i, (seq, text, grade) in enumerate(zip(sequences, texts, grades, strict=True))])
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    commit()


def evaluate(model, tokenizer, dataset, cfg, *, rollout_dir=None, step=0, commit=lambda: None):
    hits = [0] * cfg.device_batch_size
    count = min(cfg.eval_examples, len(dataset))
    for index in range(count):
        row = dataset[index]
        sequences, prefix = sample(model, tokenizer, row['question'], cfg, cfg.device_batch_size)
        texts = [tokenizer.decode(s[prefix:]) for s in sequences]
        grades = [grade_completion(row['answer'], text) for text in texts]
        if rollout_dir is not None:
            preserve_rollouts(rollout_dir, 'eval', step, index, row, sequences,
                              prefix, texts, grades, cfg, commit)
        outcomes = [grade['reward'] for grade in grades]
        for k in range(len(hits)):
            hits[k] += any(outcomes[:k+1])
        if (index + 1) % 25 == 0:
            print(f'Evaluated {index+1}/{count}', flush=True)
    return {f'pass@{k+1}': h / count for k, h in enumerate(hits)}

def save_checkpoint(model, tokenizer, optimizer, root, next_step, cfg, run_id, commit):
    directory = root / f'step-{next_step:06d}'
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    torch.save(dict(optimizer=optimizer.state_dict(), next_step=next_step,
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    config=asdict(cfg), run_id=run_id, upstream=UPSTREAM,
                    model_revision=MODEL_REVISION, data_revision=DATA_REVISION,
                    cs336_commit=CS336_COMMIT), directory / 'trainer.pt')
    (directory / 'COMPLETE').write_text('ok\n')
    (root / 'latest.tmp').write_text(directory.name)
    (root / 'latest.tmp').replace(root / 'latest')
    commit()
    return str(directory)

def run(run_name, output_dir, *, smoke=False, resume=False, commit=lambda: None):
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import wandb
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required. Launch through modal_app.py')
    if not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Missing WANDB_API_KEY; attach Modal secret wandb-secret')
    cfg = Config()
    if smoke:
        # Explicit separate diagnostic run; never represented as a parity training run.
        cfg = Config(device_batch_size=2, examples_per_step=1, num_samples=2,
                     max_new_tokens=16, eval_examples=2)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    state = None
    if resume:
        directory = root / (root / 'latest').read_text().strip()
        if not (directory / 'COMPLETE').exists():
            raise RuntimeError('Incomplete checkpoint')
        state = torch.load(directory / 'trainer.pt', map_location='cpu', weights_only=False)
        if (state['config'] != asdict(cfg) or state['upstream'] != UPSTREAM
                or state['model_revision'] != MODEL_REVISION
                or state['data_revision'] != DATA_REVISION
                or state['cs336_commit'] != CS336_COMMIT):
            raise ValueError('Checkpoint configuration mismatch')
        model_source = str(directory)
        model_kwargs = {}
    else:
        if (root / 'latest').exists():
            raise ValueError('Run already has checkpoints; use resume or a new run name')
        model_source = MODEL
        model_kwargs = dict(revision=MODEL_REVISION)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    # FP32 master weights and optimizer states; bf16 forward compute via autocast.
    model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=torch.float32,
                                                attn_implementation='sdpa', **model_kwargs).to('cuda')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = MuonAdamW(optimizer_groups(model, cfg))
    start = 0
    if state:
        optimizer.load_state_dict(state['optimizer'])
        start = state['next_step']
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state['optimizer']
    dataset = load_dataset('openai/gsm8k', 'main', revision=DATA_REVISION)
    train = dataset['train'].shuffle(seed=42)
    val = dataset['test'].shuffle(seed=42)
    num_steps = 1 if smoke else len(train) // cfg.examples_per_step * cfg.num_epochs
    run_id = state['run_id'] if state else wandb.util.generate_id()
    wb = wandb.init(project=os.environ.get('WANDB_PROJECT', 'olmo-gsm8k-nanochat-rl'),
                    entity=os.environ.get('WANDB_ENTITY') or None, name=run_name,
                    id=run_id, resume='must' if state else 'never', dir=str(root),
                    config={**asdict(cfg), 'model': MODEL, 'model_revision': MODEL_REVISION,
                            'dataset_revision': DATA_REVISION, 'nanochat_commit': UPSTREAM,
                            'smoke': smoke, 'num_steps': num_steps,
                            'prompt_template': PROMPT_TEMPLATE, 'cs336_commit': CS336_COMMIT,
                            'optimizer_groups': [{k:v for k,v in g.items() if k != 'params'}
                                                 for g in optimizer.param_groups]})
    wb.define_metric('step')
    wb.define_metric('*', step_metric='step')
    print(f'W&B: {wb.url}; {num_steps} steps, starting at {start}', flush=True)
    (root / 'config.json').write_text(json.dumps(dict(wb.config), indent=2))
    rollout_dir = root / 'rollouts' / uuid.uuid4().hex
    try:
        for step in range(start, num_steps):
            t0 = time.monotonic()
            metrics = {'step': step}
            if step % cfg.eval_every == 0:
                metrics.update(evaluate(model, tokenizer, val, cfg,
                                        rollout_dir=rollout_dir, step=step, commit=commit))
            rewards_log, lengths, loss_log, formatted = [], [], 0.0, 0
            for example_step in range(cfg.examples_per_step):
                index = (step * cfg.examples_per_step + example_step) % len(train)
                row = train[index]
                sequences = []
                passes = cfg.num_samples // cfg.device_batch_size
                for sampling_step in range(passes):
                    batch, prefix = sample(model, tokenizer, row['question'], cfg,
                                           cfg.device_batch_size,
                                           hash((step, index, sampling_step)) & 0x7FFFFFFF)
                    sequences.extend(batch)
                texts = [tokenizer.decode(s[prefix:]) for s in sequences]
                grades = [grade_completion(row['answer'], text) for text in texts]
                preserve_rollouts(rollout_dir, 'train', step, index, row, sequences,
                                  prefix, texts, grades, cfg, commit)
                rewards = torch.tensor([g['reward'] for g in grades], device='cuda')
                advantages = rewards - rewards.mean()
                inputs, targets = pack_sequences(sequences, prefix, tokenizer.eos_token_id, 'cuda')
                model.train()
                for p in range(passes):
                    b0, b1 = p * cfg.device_batch_size, (p + 1) * cfg.device_batch_size
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = model(inputs[b0:b1], use_cache=False).logits
                        loss = policy_loss(logits, targets[b0:b1], advantages[b0:b1],
                                           passes, cfg.examples_per_step)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Non-finite loss at step {step}')
                    loss.backward()
                    loss_log += loss.item()
                    del logits, loss
                rewards_log.append(rewards.mean().item())
                lengths.extend(map(len, sequences))
                formatted += sum(g['format_reward'] for g in grades)
            lrm = 1.0 - step / num_steps
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * lrm
                metrics['lr/' + group['name']] = group['lr']
            optimizer.step()
            model.zero_grad(set_to_none=True)
            metrics.update(reward=sum(rewards_log)/len(rewards_log),
                           sequence_length=sum(lengths)/len(lengths), loss=loss_log, lrm=lrm,
                           answer_format_rate=formatted/len(lengths),
                           seconds=time.monotonic()-t0,
                           peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            wb.log(metrics)
            print(json.dumps(metrics), flush=True)
            if (step > 0 and step % cfg.save_every == 0) or step == num_steps-1:
                path = save_checkpoint(model, tokenizer, optimizer, root, step+1, cfg, run_id, commit)
                wb.summary['latest_checkpoint'] = path
        wb.summary['completed'] = True
    finally:
        try:
            wb.finish()
        finally:
            commit()
    return dict(output_dir=str(root), wandb_run_id=run_id, steps=num_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(run(args.run_name, args.output_dir, smoke=args.smoke, resume=args.resume))
