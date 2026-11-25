import ast
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM
from train import Config, optimizer_groups, pack_sequences, policy_loss
from vendor.nanochat_optim import MuonAdamW


def test_rollouts_persist_verifier_inputs_before_volume_commit(tmp_path):
    import json
    from train import preserve_rollouts
    row = {'question': 'What is 1 + 1?', 'answer': 'Work #### 2'}
    texts = ['</think> <answer>2</answer>', 'unfinished reasoning']
    sequences = [[3, 4, 5], [3, 4, 6, 7]]
    grades = [dict(format_reward=1., answer_reward=1., reward=1.),
              dict(format_reward=0., answer_reward=0., reward=0.)]
    committed = []
    def commit():
        files = list(tmp_path.rglob('*.json'))
        assert not list(tmp_path.rglob('*.tmp'))
        committed.append([json.loads(f.read_text()) for f in files])
    for phase in ('train', 'eval'):
        preserve_rollouts(tmp_path / 'attempt-a', phase, 12, 7, row,
                          sequences, 2, texts, grades, Config(), commit)
    assert len(committed) == 2
    records = committed[-1]
    assert {r['phase'] for r in records} == {'train', 'eval'}
    for record in records:
        assert record['question'] == row['question']
        assert record['ground_truth'] == row['answer']
        assert record['step'] == 12 and record['dataset_index'] == 7
        assert [c['text'] for c in record['completions']] == texts
        assert [c['grade'] for c in record['completions']] == grades
        assert record['completions'][1]['completion_token_ids'] == [6, 7]
    preserve_rollouts(tmp_path / 'attempt-b', 'train', 12, 7, row,
                      sequences, 2, texts, grades, Config(), commit)
    assert len(list(tmp_path.rglob('*.json'))) == 3


def test_rollout_commit_failure_is_not_silently_ignored(tmp_path):
    import pytest
    from train import preserve_rollouts
    def fail_commit():
        raise RuntimeError('volume unavailable')
    with pytest.raises(RuntimeError, match='volume unavailable'):
        preserve_rollouts(tmp_path, 'train', 0, 0,
                          {'question': 'q', 'answer': '#### 2'}, [[1, 2]], 1,
                          ['2'], [{'reward': 0}], Config(), fail_commit)
    assert len(list(tmp_path.glob('*.json'))) == 1


def tiny_model():
    return Olmo2ForCausalLM(Olmo2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2,
                                      num_key_value_heads=2, max_position_embeddings=64,
                                      pad_token_id=0, eos_token_id=2))


def test_mask_excludes_prompt_and_padding_but_includes_sampled_stop_tokens():
    inputs, targets = pack_sequences([[9, 8, 5, 2], [9, 8, 4, 6, 2]], 2, 2, 'cpu')
    assert targets.tolist() == [[-1, 5, 2, -1], [-1, 4, 6, 2]]
    assert inputs.shape == targets.shape


def test_loss_and_gradient_equal_reference_with_unequal_lengths():
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    targets = torch.tensor([[-1, 3, 2, -1], [-1, 4, 5, 2]])
    rewards = torch.tensor([1., 0.])
    advantages = rewards - rewards.mean()
    actual = policy_loss(logits, targets, advantages, 2, 16)
    logp = -F.cross_entropy(logits.view(-1, 7), targets.view(-1), ignore_index=-1,
                           reduction='none').view_as(targets)
    reference = -(logp * advantages[:, None]).sum() / ((targets >= 0).sum() * 2 * 16)
    torch.testing.assert_close(actual, reference)
    grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.autograd.grad(reference, logits)[0])
    assert torch.count_nonzero(grad[targets == -1]) == 0
    # Winning tokens receive gradient increasing their probability, losers the inverse.
    assert grad[0, 1, 3] < 0 < grad[1, 1, 4]


def test_uniform_rewards_have_zero_policy_gradient():
    logits = torch.randn(2, 3, 7, requires_grad=True)
    rewards = torch.ones(2)
    loss = policy_loss(logits, torch.ones(2, 3, dtype=torch.long), rewards-rewards.mean(), 2, 16)
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_tiny_olmo_all_parameters_update_and_optimizer_roundtrip():
    model = tiny_model()
    cfg = Config()
    groups = optimizer_groups(model, cfg)
    assert groups[0]['betas'] == (0.8, 0.96)
    assert groups[0]['weight_decay'] == .01
    assert groups[1]['weight_decay'] == .001
    assert groups[1]['lr'] == cfg.embedding_lr * (16/768)**-.5 * cfg.init_lr_frac
    assert all(g['ns_steps'] == 5 for g in groups if g['kind'] == 'muon')
    optimizer = MuonAdamW(groups)
    ids = torch.tensor([[3, 5, 7, 8], [3, 6, 9, 10]])
    targets = torch.tensor([[-1, 7, 8, 2], [-1, 9, 10, 2]])
    loss = policy_loss(model(ids).logits, targets, torch.tensor([.5, -.5]), 1, 1)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))
    second = MuonAdamW(optimizer_groups(model, cfg))
    second.load_state_dict(optimizer.state_dict())
    assert len(second.state) == len(optimizer.state)


def test_assignment_grader_and_prompt():
    from train import prompt, grade_completion
    assert prompt('What is 1 + 1?').endswith('User: What is 1 + 1?\nAssistant: <think>')
    good = grade_completion('Work #### 2', '1+1=2.</think> <answer>2</answer>')
    assert good == dict(format_reward=1., answer_reward=1., reward=1.)
    wrong = grade_completion('Work #### 2', '1+1=3.</think> <answer>3</answer>')
    assert wrong == dict(format_reward=1., answer_reward=0., reward=0.)
    assert grade_completion('Work #### 2', '#### 2')['reward'] == 0
    # Equivalent numeric answers, not string equality.
    assert grade_completion('Work #### 0.5', '</think> <answer>1/2</answer>')['reward'] == 1


def test_generation_stop_is_per_sequence_and_keeps_answer_tag():
    from transformers import StopStringCriteria
    # Test HF stop machinery against the actual tokenizer; no model weights needed.
    from transformers import AutoTokenizer
    from train import MODEL, MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    stop = StopStringCriteria(tokenizer, ['</answer>'])
    rows = ['Some reasoning </think> <answer>2</answer>', 'Reasoning is still ongoing']
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    batch = tokenizer(rows, padding=True, return_tensors='pt')['input_ids']
    result = stop(batch, scores=None)
    assert result.tolist() == [True, False]


def test_all_shared_defaults_match_pinned_nanochat_script():
    tree = ast.parse((Path(__file__).parents[1] / 'vendor/reference/chat_rl.py.txt').read_text())
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            for kw in node.keywords:
                if kw.arg == 'default':
                    defaults[ast.literal_eval(node.args[0]).removeprefix('--').replace('-', '_')] = ast.literal_eval(kw.value)
    from dataclasses import asdict
    for key, value in asdict(Config()).items():
        if key != 'norm_lr':
            assert value == defaults[key], key


def test_sample_strips_eos_and_later_padding():
    from types import SimpleNamespace
    from train import sample
    class FakeTokenizer:
        eos_token_id = 2
        bos_token_id = 2
        def encode(self, text, add_special_tokens):
            return [3, 4]
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=4096)
        def generate(self, input_ids, **kwargs):
            assert kwargs['generation_config'].stop_strings == ['</answer>']
            return torch.tensor([[3, 4, 6, 2, 2], [3, 4, 5, 7, 2]])
    sequences, prefix = sample(FakeModel(), FakeTokenizer(), 'q', Config(), 2)
    assert prefix == 2
    assert sequences == [[3, 4, 6], [3, 4, 5, 7]]
