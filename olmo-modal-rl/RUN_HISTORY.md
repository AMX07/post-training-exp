# Experiment run history

[Training charts and exported W&B metrics](charts/README.md)

Personal OLMo/GSM8K experiment log for `AMX07/post-training-exp`; not an upstream nanochat result or a completed CS336 assignment submission.
All trained runs below started from **OLMo-2-0425-1B base**, without prior SFT or RFT. Completed/stopped status last verified November 25, 2026.

| Run | Training method / algorithm | Optimizer | GPUs / rollout engine | Outcome |
| --- | --- | --- | --- | --- |
| `olmo-gsm8k-20261125-02` | On-policy RL; nanochat-style group-mean baseline, **no reward-std normalization**; not assignment-standard GRPO | Muon + AdamW | 1× H100 / Hugging Face generation | Stopped after 131 batches; 85 had zero correctness reward, including the final 65. Checkpoints at updates 61 and 121 retained. |
| `rft-bench-h100-01` | RFT smoke benchmark (intended) | AdamW | 2× H100 / vLLM | Startup failed: cache mount conflict; no training. |
| `rft-bench-h100-02` | RFT smoke benchmark (intended) | AdamW | 2× H100 / vLLM | Canceled before training while fixing cache configuration. |
| `rft-bench-b200-01` | RFT smoke benchmark (intended) | AdamW | 2× B200 / vLLM | Failed: uv cache locking on Modal Volume; no training. |
| `rft-bench-b200-02` | RFT smoke benchmark (intended) | AdamW | 2× B200 / vLLM | Failed: missing CUDA compiler for FlashInfer; no training. |
| `rft-bench-h100-03` | On-policy RFT / expert iteration; correctness-weighted response NLL, constant normalization | AdamW | 2× H100 / vLLM | Completed 3 steps; 159.24 s; 832 completions and final checkpoint saved. |
| `rft-bench-h100-04` | Same RFT; corrected CUDA development image, seed 0 | AdamW | 2× H100 / vLLM | Completed 3 steps; 104.65 s; 832 completions and final checkpoint saved. |
| `rft-bench-b200-03` | Same RFT and image as H100-04, seed 0 | AdamW | 2× B200 / vLLM | Completed 3 steps; 166.86 s; 832 completions and final checkpoint saved. |
| `rft-sanity-h100-seed0-20261125-01` | Assignment-style on-policy RFT; seed 0; 256 responses/step (32 questions × 8), max 512 tokens | AdamW, LR `1e-5` | 2× H100 / vLLM | Completed **50 steps in 12.98 min**. Validation correctness **0.29% → 33.50%** on 1,024 questions. Step-50 checkpoint and rollout files saved. |

RFT trains on correct responses only, with loss denominator `256 × 512`; rejected completions are still saved. The three-step benchmarks use only 32 validation questions and are infrastructure checks, not learning-quality comparisons. Runtime includes model startup, evaluation, and checkpoint saving; it excludes image building and queue time.

The earlier single-GPU experiment did **not** preserve original rollout text. Its aggregate history and checkpoints remain available. New RFT runs save train/evaluation completions, verifier results, and optimizer/model checkpoints in Modal Volume `olmo-assignment-artifacts`, under each run name.

**Not run yet:** assignment-standard GRPO (mean baseline + reward-std normalization), GRPO initialized from our RFT checkpoint, standalone SFT, or the full 200-step/four-seed comparison. Early canceled single-GPU setup launches are not counted as completed training runs.

Results: [RFT sanity run](https://wandb.ai/alephnott/olmo-gsm8k-assignment/runs/p2wswixp) · [earlier nanochat-style RL run](https://wandb.ai/alephnott/olmo-gsm8k-nanochat-rl/runs/w9vc0c3v).
