# CS336 RFT and on-policy GRPO

Measured GPU timings, saved-artifact verification, and benchmark limitations are
recorded in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

This is a separate implementation from `train.py` (the stopped nanochat-style
experiment). It uses the supplied Spring 2026 assignment, version 26.0.0, and
reference commit `c2734a26308710949fe13226960a1e8cece94b7e`.

## Algorithm and defaults

| Setting | Value |
| --- | --- |
| Initial model | `allenai/OLMo-2-0425-1B`, pinned base revision |
| Prompt | Assignment zero-shot `r1_zero`, no demonstrations |
| Train / validation subset | 6,400 / 1,024 GSM8K questions |
| Rollout steps | 200 |
| Questions × responses per question | 32 × 8 = 256 responses per step |
| Training microbatch | 8 responses; 32 passes before filtering |
| Generation | Temperature 1, top-p 1, top-k disabled, max 512 tokens |
| Stop | `</answer>`, included in completion text |
| Optimizer | AdamW, LR 1e-5, betas (0.9, 0.95), decay 0 |
| Gradient clipping | Global norm 1 |
| Evaluation / checkpoint interval | Every 10 steps; final checkpoint and post-update evaluation |

RFT uses the **raw correctness reward**, no baseline, no advantage normalization,
and no importance weights. It minimizes reward-weighted response-token negative
log likelihood with the fixed denominator **256 × 512**. Zero-reward responses
are omitted from training, but the denominator never changes to the accepted count.
All-zero batches are preserved and logged; the optimizer is skipped, preventing
Adam's existing momentum from moving weights without new supervision.

`--mode grpo` uses a per-question mean baseline and sample-standard-deviation
normalization (epsilon 1e-6), per-sequence token normalization and original-batch
averaging. This is standard **on-policy** GRPO, not the stopped nanochat objective.
It is prepared for later controlled comparisons; no full RFT/GRPO suite is launched
automatically. The assignment asks for an initial roughly 50-step sanity run,
then four seeds and variance plots. Preparing the scripts is not completing those
experiments. Compare modes from the same base model rather than silently using an
RFT checkpoint as the GRPO baseline.

## Two-GPU design and compatibility

Both GPUs are on the same Modal machine. GPU 0 holds the BF16 training policy and
optimizer. GPU 1 holds vLLM's inference replica and KV cache (tensor parallel size
1). Current policy weights are synchronized through NCCL and inference caches
reset before generation. The phases alternate; two GPUs do not imply simultaneous
training and inference. See `vendor/ASSIGNMENT_VLLM_NOTICE.md` for helper provenance.

Python **3.13** is used everywhere at the user's request. The assignment publishes
a Python 3.12 environment and a cp312-only standalone FlashAttention wheel. This
implementation uses PyTorch **SDPA** for training instead; vLLM retains its own
optimized inference kernels. The GPU stack uses vLLM 0.19.1 and Torch 2.10.0+cu129.
It must not be installed over the old Torch 2.9.1 experiment environment.

The Modal image includes the CUDA 12.9.1 development toolkit: B200's native
FlashInfer backend needs `nvcc` to compile kernels at startup. Both GPU types
were successfully smoke-tested on this image. H100 remains the default based
on the measured comparison in `BENCHMARK_RESULTS.md`.

## Persistent evidence

Artifacts are stored in Modal Volume `olmo-assignment-artifacts`, separate from
the stopped run's `olmo-gsm8k-checkpoints` Volume. Each new/resumed invocation has
a unique attempt directory. Nothing overwrites the old run or its checkpoints.

- All train/evaluation completions, including rejected answers: prompt, ground
  truth, exact verifier text, raw sampled IDs, training token IDs, finish reason,
  reward/format scores, seed, question index, and step.
- Atomic JSON files and a Volume commit **before optimizer updates**. A commit
  failure aborts training. A process crash before commit can lose the in-flight
  batch, but not advance training beyond uncommitted rollouts.
- Local metrics plus W&B: rewards, formatting, lengths, truncation, zero-reward
  groups, loss, gradient norm, update-applied flag, phase timings and GPU memory.
- Token entropy is labeled as covering nonzero-advantage training responses;
  `null` means no such responses, not zero model entropy.
- Model/tokenizer/optimizer/RNG checkpoints every 10 steps and at completion,
  with completion markers and a latest pointer. Up to nine updates can be lost
  on abrupt interruption; the earlier rollouts remain saved. Resume requires
  matching config and source fingerprints and does not overwrite prior attempts.
- Source snapshots, model/data revisions, package versions, GPU names and prompt.

## Local validation

Run from `/workspace/post-training-exp/olmo-modal-rl`:

```bash
export UV_CACHE_DIR=/workspace/.cache/uv
uv venv --python 3.13 .venv-assignment
uv pip install --python .venv-assignment/bin/python torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-assignment/bin/python -r requirements-assignment.txt
OMP_NUM_THREADS=2 uv run --no-project --python .venv-assignment/bin/python python -m pytest -q tests/test_assignment_training.py tests/test_assignment_vllm.py
```

These tests cover mathematical and persistence contracts, including pruned RFT
gradient equivalence and official GRPO sample-std behavior. Mocked NCCL tests do
not establish GPU runtime compatibility. GPU benchmark results are recorded
separately when available.

## Modal commands

`bash modal_cloud.sh` uses uv, Modal API proxy support, and system CA trust with
TLS verification enabled. It reads credentials from runtime settings. The Modal
secret `wandb_sec` must contain `WANDB_API_KEY`; never put keys in source or commands.

Bounded diagnostic runs (three full-sized rollout batches, 32 validation questions,
15-minute execution timeout, no automatic retries):

```bash
bash modal_cloud.sh run --detach assignment_modal.py --run-name rft-bench-h100-01 --gpu H100 --benchmark
bash modal_cloud.sh run --detach assignment_modal.py --run-name rft-bench-b200-01 --gpu B200 --benchmark
```

Do not rerun those names if their jobs already exist. Check app logs first. Full
runs require a separate decision after validation; example commands:

```bash
bash modal_cloud.sh run --detach assignment_modal.py --run-name rft-sanity-seed0 --mode rft --seed 0 --steps 50
bash modal_cloud.sh run --detach assignment_modal.py --run-name rft-seed0 --mode rft --seed 0
bash modal_cloud.sh run --detach assignment_modal.py --run-name grpo-seed0 --mode grpo --seed 0
```

The local launcher submits asynchronously and prints a function-call ID. Its
successful exit means submission, not experiment completion. Training is on H100:2
by default; `--gpu B200` selects B200:2. `--resume` requires the original run name,
mode, seed, step limit and source/configuration.

## Hardware cost comparison

Modal listed rates checked November 24, 2026 (America/Los_Angeles):

| Pair | GPU-only hourly cost | Required speedup over 2×H100 to be cheaper |
| --- | ---: | ---: |
| 2×H100 | $7.8984 | baseline |
| 2×B200 | $12.4992 | greater than 1.5825× |
| 2×B300 | $14.1984 | greater than 1.7976× |

Sources: https://modal.com/pricing and https://modal.com/docs/guide/gpu .
The assignment labels budgets in **B200 hours**, not B300. B300 needs CUDA 13.1+
and is not configured with this CUDA 12.9 stack. H100 may be substituted with H200
by Modal; record actual GPU names. Both GPUs are billed while one phase is idle.
CPU/RAM, storage, startup, checkpoint commits and model downloads are additional.
Benchmark the same workload: compare generation, update, sync, persistence and
total time separately. A short benchmark with zero accepted answers does not
measure training-backward throughput, and three steps are not a convergence test.
