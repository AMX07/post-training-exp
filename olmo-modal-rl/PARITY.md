# Implementation provenance and deliberate adaptations

Reference: karpathy/nanochat commit `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`,
`scripts/chat_rl.py`, `nanochat/gpt.py::setup_optimizer`, `nanochat/optim.py`,
and `tasks/gsm8k.py`. This is a pinned snapshot of master, not a moving dependency.

CS336 Spring 2026 Assignment 5 (uploaded PDF, version 26.0.0) supplies the
OLMo-specific prompt, grading, and answer stop convention. Prompt and grader are
vendored from stanford-cs336/assignment5-alignment at
`c2734a26308710949fe13226960a1e8cece94b7e`.

## Preserved from nanochat

- On-policy, one optimizer update per batch; full model training, not LoRA.
- Binary correctness reward; per-question advantages `reward - mean(rewards)`.
- No reward standard-deviation division, reference model, KL, PPO ratio/clipping,
  gradient clipping, entropy bonus, or multiple updates on old rollouts.
- Loss is negative log-probability times advantage, divided by **each
  microbatch's** number of valid response tokens, then by passes per question
  and questions per step. This is the actual source behavior, not a global
  token denominator or per-sequence mean.
- One epoch, device batch 8, 16 questions per optimizer step, 16 samples per
  question (256 sequences per update), 256 maximum new tokens, temperature 1,
  top-k 50. GSM8K train and test both shuffled with seed 42. 7,473 training
  questions produce 467 steps; the last single question is not used.
- Evaluate first 400 shuffled test questions at steps 0,60,...,420; eight
  samples each, temperature 1, top-k 50. pass@k means success among the first
  k samples, not the combinatorial estimator. No extra final evaluation.
- Save after updates at step indices 60,120,...,420 and 466. Directory numbers
  count completed updates (61,121,...,421,467).
- Sampling seeds `hash((step, example_index, sampling_step)) & 0x7FFFFFFF`;
  evaluation seed 42 on each call. Different engines/tokenizers do not produce
  identical token streams even with identical seeds.
- Initial LR fraction .05, linear multiplier `1 - step / total_steps`.
  The final update has multiplier 1/467, rather than exactly zero.
- Vendored MuonAdamW optimizer, including Muon momentum .95, five Polar Express
  iterations, beta2 .9, shape-based scaling. Only its compute-dtype import was
  replaced with a standalone bf16-on-CUDA / fp32-on-CPU selection.
- Embedding base LR .2 and output-head LR .004, both scaled by
  `(hidden_size/768)**(-.5)`. Muon base LR .02. For OLMo hidden size 2048,
  initial effective LRs are .00612372436, .000122474487, and .001.
- Actual AdamW settings from GPT optimizer: embeddings betas (.8,.995),
  decay .001; output head betas (.8,.96), decay .01; epsilon 1e-10.
  The RL CLI's weight_decay=0 is passed to Muon; it does NOT override these
  hardcoded AdamW decays in the pinned upstream implementation.

## Explicit OLMo / assignment adaptations

- Start from the requested **base** OLMo, not nanochat's SFT checkpoint.
- Use assignment R1-Zero prompt ending `Assistant: <think>`. Plain text tokens;
  no added chat tokens, no calculator execution, and no initial SFT.
- Use the assignment's exact `r1_zero_reward_fn`, ground truth taken after
  `####`. Its strict format check requires `</think> <answer>` and `</answer>`.
  Format correctness alone gets no reward. Formatting is logged separately.
- Stop per sequence at `</answer>` (included) or EOS (excluded, matching
  nanochat's exclusion of terminal control tokens). Padding/prompt never
  contributes to the loss; sampled answer tags do.
- OLMo dense projections use Muon. Its learned normalization vectors have no
  exact nanochat counterpart: AdamW base LR .005, betas (.8,.95), epsilon 1e-10,
  decay 0, then the same .05 LR fraction and rampdown. All weights train.
  Nanochat-only value embeddings/residual gates do not exist in OLMo.
- Transformers SDPA and KV-cached generation replace nanochat's Engine.
  FP32 master weights, bf16 autocast, activation checkpointing; one H100 GPU.
  This uses the same current model for sampling/training, avoiding stale
  rollout weights. No separate inference server is required.
- The assignment's vLLM, 512-token limit, AdamW-only recipe, normalized
  advantages, alternative losses, and ablation homework are not adopted.
- W&B is enabled; optimizer/RNG state is saved for resumability in addition
  to HF model/tokenizer files. This is more than nanochat saves.
- `--smoke` is a distinct one-step diagnostic with reduced samples/lengths;
  it is clearly tagged and cannot resume as the full experiment.

This is algorithmic/default parity with documented architecture and prompt
adaptations, not a bit-for-bit reproduction or a guarantee of improving accuracy.
The 256-token cap and strict answer formatting can yield all-zero reward groups
for a base model; no reward-shaping fallback or silent hyperparameter change is made.
