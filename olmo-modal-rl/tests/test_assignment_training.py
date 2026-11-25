"""Numerical and durability contracts for the assignment RFT/GRPO loop.

These tests require no model downloads, vLLM server, or GPU allocation.
"""
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from assignment_train import (
    AssignmentConfig,
    compute_advantages,
    loss_from_logits,
    pack_token_sequences,
    persist_rollouts,
)


def test_rft_retains_raw_rewards_without_group_centering():
    rewards = torch.tensor([0., .5, 1., 0., 0., 0., 0., 0.])
    actual = compute_advantages(rewards, group_size=4, mode='rft')
    torch.testing.assert_close(actual, rewards)
    assert torch.count_nonzero(compute_advantages(torch.zeros(8), 4, 'rft')) == 0


def test_grpo_uses_each_questions_sample_std_and_handles_uniform_groups():
    rewards = torch.tensor([0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1.])
    eps = 1e-6
    actual = compute_advantages(rewards, group_size=4, mode='grpo', eps=eps)
    expected = torch.tensor([-1.5, .5, .5, .5]) * (.5 / (.5 + eps))
    torch.testing.assert_close(actual[:4], expected)
    assert torch.count_nonzero(actual[4:]) == 0
    assert torch.isfinite(actual).all()


def test_response_mask_is_shifted_and_preserves_eos_when_also_padding():
    # Token 2 serves as both an actual sampled EOS and the padding token.
    inputs, targets, attention = pack_token_sequences(
        [[8, 9], [8]], [[3, 2], [4, 5, 2]], pad_id=2,
    )
    assert inputs.tolist() == [[8, 9, 3], [8, 4, 5]]
    assert targets.tolist() == [[-100, 3, 2], [4, 5, 2]]
    assert attention.tolist() == [[1, 1, 1], [1, 1, 1]]
    inputs, targets, attention = pack_token_sequences(
        [[8, 9], [8]], [[3, 2], [2]], pad_id=2,
    )
    assert targets.tolist() == [[-100, 3, 2], [2, -100, -100]]
    assert attention.tolist() == [[1, 1, 1], [1, 1, 0]]


def _loss_inputs():
    torch.manual_seed(7)
    logits = torch.randn(4, 5, 11, requires_grad=True)
    targets = torch.tensor([
        [-100, 3, 4, 2, -100],
        [5, 6, 7, 8, 2],
        [-100, -100, 3, 2, -100],
        [-100, 7, 2, -100, -100],
    ])
    return logits, targets


def test_rft_matches_supervised_logloss_with_original_batch_denominator():
    logits, targets = _loss_inputs()
    rewards = torch.tensor([0., 1., 1., 0.])
    loss = loss_from_logits(logits, targets, rewards, 'rft', total_rollouts=4,
                            normalization_constant=512)
    token_nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                               ignore_index=-100, reduction='none').view_as(targets)
    expected = (token_nll * rewards[:, None]).sum() / (4 * 512)
    torch.testing.assert_close(loss, expected)
    expected_grad = torch.autograd.grad(expected, logits, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.count_nonzero(actual_grad[[0, 3]]) == 0
    assert torch.count_nonzero(actual_grad[targets == -100]) == 0
    assert actual_grad[1, 0, 5] < 0  # Correct-token log probability is increased.


def test_pruning_and_uneven_microbatches_preserve_full_rft_gradient():
    logits, targets = _loss_inputs()
    rewards = torch.tensor([0., 1., 1., 1.])
    full_loss = loss_from_logits(logits, targets, rewards, 'rft', 4, 512)
    full_gradient = torch.autograd.grad(full_loss, logits, retain_graph=True)[0]
    # Prune one rejected response and use microbatches of sizes two and one.
    # Every call must retain the original batch denominator of four.
    accumulated = sum(loss_from_logits(logits[ids], targets[ids], rewards[ids],
                                      'rft', 4, 512) for ids in ([1, 2], [3]))
    pruned_gradient = torch.autograd.grad(accumulated, logits)[0]
    torch.testing.assert_close(accumulated, full_loss)
    torch.testing.assert_close(pruned_gradient, full_gradient)


def test_standard_grpo_normalizes_each_response_then_original_batch():
    logits, targets = _loss_inputs()
    advantages = torch.tensor([-1., 1., .5, -.5])
    actual = loss_from_logits(logits, targets, advantages, 'grpo', 4, 512)
    nll = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                         ignore_index=-100, reduction='none').view_as(targets)
    lengths = (targets != -100).sum(dim=1)
    expected = ((nll.sum(dim=1) / lengths) * advantages).sum() / 4
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])


def test_rollout_commit_observes_complete_exact_verifier_records(tmp_path):
    records = [dict(question='What is 1 + 1?', ground_truth='2',
                    prompt='User: What is 1 + 1?\nAssistant: <think>',
                    completion='one plus one</think><answer>2</answer>',
                    prompt_token_ids=[1, 3], completion_token_ids=[4, 5, 2],
                    finish_reason='stop', reward=1., format_reward=1.,
                    answer_reward=1.)]
    observed = []

    def commit():
        files = list(tmp_path.rglob('*.json'))
        assert files, 'Volume commit must happen after the rollout file is written'
        assert not list(tmp_path.rglob('*.tmp'))
        observed.extend(json.loads(path.read_text()) for path in files)

    persist_rollouts(tmp_path, 'train', 3, records, commit)
    assert observed
    serialized = json.dumps(observed)
    assert records[0]['completion'] in serialized
    assert 'completion_token_ids' in serialized
    assert 'ground_truth' in serialized


def test_rollout_storage_failure_prevents_return_to_training(tmp_path):
    def commit():
        raise RuntimeError('volume commit failed')

    with pytest.raises(RuntimeError, match='volume commit failed'):
        persist_rollouts(tmp_path, 'train', 0, [{'completion': 'bad', 'reward': 0.}], commit)
    assert list(tmp_path.rglob('*.json')), 'Local record should remain recoverable'


def test_defaults_match_assignment_rft_batch_geometry():
    cfg = AssignmentConfig()
    assert cfg.mode == 'rft'
    assert cfg.group_size == 8
    assert cfg.rollout_batch_size == 256
    assert cfg.microbatch_size == 8
    assert cfg.normalization_constant == 512


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 4)
        self.output = torch.nn.Linear(4, 12)
        self.forward_calls = 0

    def forward(self, inputs, attention_mask=None, use_cache=False):
        self.forward_calls += 1
        return SimpleNamespace(logits=self.output(self.embedding(inputs)))


def test_zero_reward_batch_skips_forward_and_optimizer_even_with_momentum():
    from assignment_train import train_update
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
    # Populate optimizer state to distinguish skipping from stepping zero gradients.
    model(torch.tensor([[1, 2]])).logits.sum().backward()
    optimizer.step()
    before = [p.detach().clone() for p in model.parameters()]
    steps = [state['step'].clone() for state in optimizer.state.values()]
    model.forward_calls = 0
    records = [dict(prompt_token_ids=[1], completion_token_ids=[2], reward=0.)] * 4
    cfg = AssignmentConfig(rollout_batch_size=4, group_size=2, microbatch_size=2)
    result = train_update(model, optimizer, records, SimpleNamespace(pad_token_id=0), cfg)
    assert not result['optimizer_update_applied']
    assert result['trained_responses'] == 0
    assert result['loss'] == 0 and result['zero_advantage_fraction'] == 1
    assert model.forward_calls == 0
    for old, parameter in zip(before, model.parameters()):
        torch.testing.assert_close(parameter, old, rtol=0, atol=0)
    for old, state in zip(steps, optimizer.state.values()):
        torch.testing.assert_close(state['step'], old, rtol=0, atol=0)


def test_accepted_rft_responses_produce_one_optimizer_update():
    from assignment_train import train_update
    model = TinyPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
    records = [dict(prompt_token_ids=[1, 2], completion_token_ids=[3, 4], reward=r)
               for r in (0., 1., 1., 0.)]
    cfg = AssignmentConfig(rollout_batch_size=4, group_size=2, microbatch_size=1)
    result = train_update(model, optimizer, records, SimpleNamespace(pad_token_id=0), cfg)
    assert result['optimizer_update_applied']
    assert result['trained_responses'] == 2 and model.forward_calls == 2
    assert result['loss'] > 0 and result['grad_norm'] > 0
    assert result['entropy_token_count'] == 4
    assert all(state['step'].item() == 1 for state in optimizer.state.values())


def test_immutable_rollout_file_refuses_overwrite(tmp_path):
    persist_rollouts(tmp_path, 'train', 0, [{'text': 'original'}], lambda: None)
    with pytest.raises(FileExistsError):
        persist_rollouts(tmp_path, 'train', 0, [{'text': 'replacement'}], lambda: None)
    assert json.loads(next(tmp_path.glob('*.json')).read_text())['records'][0]['text'] == 'original'


def test_checkpoint_publishes_weights_state_and_completion_before_commit(tmp_path, monkeypatch):
    from assignment_train import checkpoint
    model = TinyPolicy()
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    model.save_pretrained = lambda path, **kw: (path / 'model.safetensors').write_bytes(b'weights')
    tokenizer = SimpleNamespace(save_pretrained=lambda path: (path / 'tokenizer.json').write_text('{}'))
    optimizer = torch.optim.AdamW(model.parameters())
    inspected = []
    def commit():
        latest = json.loads((tmp_path / 'latest.json').read_text())['checkpoint']
        directory = tmp_path / latest
        assert json.loads((directory / 'COMPLETE').read_text())['next_step'] == 10
        assert (directory / 'model.safetensors').read_bytes() == b'weights'
        assert (directory / 'tokenizer.json').exists()
        state = torch.load(directory / 'trainer.pt', weights_only=False)
        assert state['next_step'] == 10 and state['run_id'] == 'test-run'
        assert {'optimizer', 'torch_rng', 'cuda_rng', 'python_rng', 'config'} <= state.keys()
        inspected.append(latest)
    checkpoint(tmp_path, model, tokenizer, optimizer, AssignmentConfig(), 10, 'test-run', commit)
    assert len(inspected) == 1
    with pytest.raises(FileExistsError):
        checkpoint(tmp_path, model, tokenizer, optimizer, AssignmentConfig(), 10, 'test-run', commit)


def test_failed_checkpoint_does_not_replace_latest_pointer(tmp_path, monkeypatch):
    from assignment_train import checkpoint
    old = {'checkpoint': 'checkpoints/step-000010'}
    (tmp_path / 'latest.json').write_text(json.dumps(old))
    model = TinyPolicy()
    def fail_save(path, **kwargs):
        (path / 'partial-weights').write_bytes(b'partial')
        raise OSError('storage failure')
    model.save_pretrained = fail_save
    commits = []
    with pytest.raises(OSError, match='storage failure'):
        checkpoint(tmp_path, model, None, None, AssignmentConfig(), 20, 'test-run', lambda: commits.append(True))
    assert json.loads((tmp_path / 'latest.json').read_text()) == old
    assert not (tmp_path / 'checkpoints/step-000020').exists()
    assert not commits


def test_generation_records_exact_text_sampled_ids_and_separate_tokenization(monkeypatch):
    from assignment_train import generate_records
    from vendor import drgrpo_grader
    text = 'reasoning</think> <answer>2</answer>'
    encoded = []
    class Tokenizer:
        def encode(self, value, *, add_special_tokens):
            assert not add_special_tokens
            encoded.append(value)
            return [4, 5] if value == text else [1, 2, 3]
    requests = []
    class Server:
        def generate_completions(self, prompts, parameters):
            requests.append((prompts, parameters))
            return [SimpleNamespace(text=text, token_ids=[8, 9, 10], finish_reason='stop')]
    seen = []
    def grader(response, expected):
        seen.append((response, expected))
        return dict(reward=1., format_reward=1., answer_reward=1.)
    monkeypatch.setattr(drgrpo_grader, 'r1_zero_reward_fn', grader)
    rows = [dict(question='What is 1 + 1?', answer='Reasoning #### 2', dataset_index=17)]
    records = generate_records(Server(), Tokenizer(), rows, AssignmentConfig(), 1, 42, 3, 'eval')
    record = records[0]
    assert seen == [(text, '2')]
    assert record['text'] == text
    assert record['sampled_token_ids'] == [8, 9, 10]
    assert record['completion_token_ids'] == [4, 5]
    assert record['prompt_token_ids'] == [1, 2, 3]
    assert encoded == [record['prompt'], text]
    assert record['finish_reason'] == 'stop'
    params = requests[0][1]
    assert params['stop'] == ['</answer>'] and params['include_stop_str_in_output']
    assert params['top_k'] == -1 and params['top_p'] == 1. and params['max_tokens'] == 512
