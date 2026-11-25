# OLMo GSM8K RL on Modal

For the new **CS336-aligned RFT / GRPO pipeline with two GPUs and vLLM**, see
[ASSIGNMENT_RFT.md](ASSIGNMENT_RFT.md). The remainder of this page describes the
earlier single-GPU nanochat-style experiment; its stopped checkpoints are retained.

Full-parameter post-training of `allenai/OLMo-2-0425-1B` on GSM8K. Uses the
CS336 Spring 2026 R1-Zero prompt/grader and nanochat's simplified GRPO objective
and numeric defaults. See [PARITY.md](PARITY.md) for the precise contract.

One Modal H100 runs both rollouts and optimization. No Tinker account or separate
inference server is needed. W&B logs rewards, answer format rate, pass@1–8,
loss, learning rates, sequence length, wall time, and peak GPU memory.

## Credentials

From the repository root, create a local environment for the Modal launcher:

```bash
cd olmo-modal-rl
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python modal==1.2.4
uv run --no-project --python .venv/bin/python modal app list
```

Supply `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` through secure runtime environment
settings. Never put credential values in the repository or command arguments.
In the Modal dashboard, create a secret named `wandb_sec` containing
`WANDB_API_KEY`; optionally add `WANDB_ENTITY` and `WANDB_PROJECT`.
The default W&B project is `olmo-gsm8k-nanochat-rl`. Do not paste API keys into chat.
The model/dataset are public; Hugging Face credentials are not required.

## Connection status and launching from another machine

This cloud workspace requires Modal's API proxy support and the system certificate
store. Authentication and app listing succeeded with Modal 1.6.4,
`modal[api-proxy-support]`, and `truststore==0.10.5`. TLS verification remains enabled.
Use the following launcher in this cloud environment (append Modal CLI arguments):

```bash
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_TOOL_DIR=/workspace/.local/share/uv/tools
export UV_TOOL_BIN_DIR=/workspace/.local/bin
uv tool install --python 3.13 --with truststore==0.10.5 'modal[api-proxy-support]==1.6.4'
uv run --no-project --python /workspace/.local/share/uv/tools/modal/bin/python python -c 'import truststore; truststore.inject_into_ssl(); import runpy; runpy.run_module("modal", run_name="__main__")' app list
```

The remote image uses Python 3.13 and uv. The `wandb_sec` secret is configured in
workspace `mittalansh9`, environment `main`. The launcher submits with `.spawn()`
and prints the function-call ID; use `--detach` to keep the remote job running
after the launcher exits. A successful launcher exit confirms submission, not
completion of training. Inspect the Modal app logs and W&B for progress.

If that connection limitation persists, clone this repository on your own
machine and install only the launcher dependency locally:

```bash
uv tool install --python 3.13 modal==1.2.4
uv tool run --python 3.13 --from modal==1.2.4 modal token new
```

Run the commands below from the `olmo-modal-rl` directory. Modal builds
and installs all training dependencies remotely; your machine does not need a GPU.
Create `wandb_sec` in the same Modal workspace/environment that you launch into.

## Launch and resume

First run a separate smoke test (real base model, one tiny update):

```bash
uv tool run --python 3.13 --from modal==1.2.4 modal run modal_app.py --run-name olmo-smoke-001 --smoke
```

After that succeeds, launch the full 467-step experiment:

```bash
uv tool run --python 3.13 --from modal==1.2.4 modal run --detach modal_app.py --run-name olmo-gsm8k-001
```

This incurs Modal GPU charges. The function has a 24-hour timeout; runtime and
GPU memory have not yet been measured for this workload. Detached execution
continues if this cloud workspace disconnects. For interruptions, resume from
the most recent completed checkpoint:

```bash
uv tool run --python 3.13 --from modal==1.2.4 modal run --detach modal_app.py --run-name olmo-gsm8k-001 --resume
```

Don't launch concurrent writers with the same run name. Resume restores model,
optimizer, RNG, dataset position, and W&B run ID. Work since the last checkpoint
may be repeated. W&B may retain logs from that repeated interval; use the
explicit `step` metric to identify it. Resume is intended for trusted checkpoints
created by this code (trainer.pt contains Python optimizer state).

## Outputs

Future runs preserve every training and evaluation completion under
`/<run-name>/rollouts/<attempt-id>/`. Each JSON file contains one question's
samples, exact text passed to the verifier, prompt, ground truth, post-trim token
IDs, verifier scores, step/index, configuration, and source revisions. Files are
flushed, atomically renamed, and committed to the Modal Volume after each question
before any subsequent training update. Commit errors stop the run. Abrupt failure
before a commit completes can still lose the in-flight group. Attempt IDs prevent
resume from overwriting earlier records. Per-question commits add I/O overhead.

The stopped run `olmo-gsm8k-20261125-02` predates this logging: its original
completion texts cannot be recovered from aggregate metrics. Checkpoints at steps
61 and 121 remain in the Volume. Its complete W&B history (steps 0–130) was
downloaded to ignored `outputs/audit/`, with parsed `training-history.json` and
`summary.json`. No training was restarted to collect this data.

Modal Volume `olmo-gsm8k-checkpoints`, path `/<run-name>/`:
- `config.json`: full configuration and source revisions
- `step-000061/`, etc.: HF weights/tokenizer, optimizer/RNG state, COMPLETE marker
- `latest`: most recent complete checkpoint directory

```bash
uv tool run --python 3.13 --from modal==1.2.4 modal volume get olmo-gsm8k-checkpoints /olmo-gsm8k-001 ./outputs
```

HF download cache lives in Modal Volume `olmo-gsm8k-cache`.

## Local tests

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
TORCHDYNAMO_DISABLE=1 uv run --no-project --python .venv/bin/python python -m pytest -q tests
```

CPU tests check masking, exact policy-gradient loss/gradients against the reference,
optimizer mapping and update on a tiny OLMo, prompt/grader behavior, and generation
stop semantics. CPU tests disable optimizer compilation; the real GPU smoke test
must additionally validate CUDA compilation, memory, downloads, W&B, and persistence.
