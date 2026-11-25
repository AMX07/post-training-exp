# Two-GPU RFT benchmark evidence

These are short infrastructure benchmarks, not completed assignment experiments or
evidence that RFT improves model quality. No long training run is implied by a
successful benchmark. The three-step benchmark uses the actual OLMo-2-0425-1B
model, 256 training rollouts per step (32 questions × 8 samples), and a 512-token
generation cap, but only 32 validation questions rather than the full assignment
evaluation set. Seed is 0; RFT learning rate is 1e-5.

## Final comparison: recommend H100 for this workload

Both benchmarks completed on the same CUDA 12.9 development image after adding
the CUDA compiler required by B200's native FlashInfer backend. Configurations,
package versions, and source fingerprints match. Both Modal apps are now
**stopped, with zero tasks**. No full assignment training run was launched.

| Measurement | 2× H100, `rft-bench-h100-04` | 2× B200, `rft-bench-b200-03` |
|---|---:|---:|
| Full measured workload | 104.65 s | 166.86 s |
| Startup total / vLLM startup | 59.19 / 49.46 s | 92.86 / 80.23 s |
| Mean generation + grading, 256 responses | 1.860 s | 1.835 s |
| First optimizer update (cold) | 0.862 s | 17.165 s |
| Mean subsequent optimizer update, steps 1–2 | 0.126 s | 0.149 s |
| Mean warm step excluding checkpoint, steps 1–2 | 3.212 s | 4.597 s |
| Mean warm rollout persistence, steps 1–2 | 1.122 s | 2.492 s |
| Final checkpoint write + commit | 26.53 s | 29.47 s |
| Peak training-GPU allocated memory | 14.92 GB | 14.87 GB |
| GPU-only cost for measured workload | **$0.230** | **$0.579** |
| Actual mean sampled tokens/response | 107.53 | 105.37 |
| Training responses accepted and used | 5/768 | 8/768 |
| Final pass@1, 32 questions | 1/32 | 1/32 |

B200 generation was only **1.014× as fast** in this sample, below its **1.582×**
GPU-price break-even speedup. H100 is the lower-cost choice supported by these
measurements. The higher B200 total includes longer startup and a cold first
update; later updates should not be extrapolated from that cold cost. Persistence
latency also differs and is not a GPU-compute comparison. This is only one
three-step run per device with 1–5 accepted responses per update: it cannot
establish mature-policy throughput, full-run cost, or statistical quality parity.
Even with the same seed, sampled outputs differ between the runs; they are the
same workload configuration, not identical token sequences.

- [H100 Modal app](https://modal.com/apps/mittalansh9/main/ap-ffntAm7hpPaRlev3QLDgrs),
  [W&B](https://wandb.ai/alephnott/olmo-gsm8k-assignment/runs/fgkoyxj4),
  attempt `ff3f68271dde4122af4ee56a5bc4d069`.
- [B200 Modal app](https://modal.com/apps/mittalansh9/main/ap-VDubz9EKogtHARk4Nfaf8Z),
  [W&B](https://wandb.ai/alephnott/olmo-gsm8k-assignment/runs/hnlrh0lb),
  attempt `5f9e049f239a4b39b490735ad6cd6f85`.

Each final comparison run preserves **832 completion records**: 768 training,
32 initial evaluation, and 32 final evaluation. Each has 756 training responses
with `stop` finish reason and 12 with `length`; the cap is 512 tokens, not the
actual typical response length. Both final checkpoints at `step-000003` have
verified `COMPLETE` markers, model weights, tokenizer, optimizer/RNG state, and
latest pointers. The checkpoint files were listed and markers read; no restore
test was performed. The downloaded records, configs, provenance, metrics,
manifests, and checkpoint verification are under ignored
`outputs/assignment-benchmarks/h100-04/` and `b200-03/`. Original weights remain
in the `olmo-assignment-artifacts` Volume.

## Earlier H100 validation run (before the shared development image)

- Run: `rft-bench-h100-03`
- Attempt: `13df879dfe64470692cad7324db7e698`
- [Modal app](https://modal.com/apps/mittalansh9/main/ap-N6Eizu9rJ2PUbpIHTetvWK)
- [W&B run](https://wandb.ai/alephnott/olmo-gsm8k-assignment/runs/fy5uymrl)
- Provenance reports **two NVIDIA H100 80GB HBM3 GPUs**.
- Python 3.13 image; Torch 2.10.0+cu129, vLLM 0.19.1, Transformers 5.7.0,
  datasets 4.8.5, W&B 0.26.1.
- GPU 0 performs training; GPU 1 runs vLLM with NCCL weight synchronization.

| Measurement | Result |
|---|---:|
| Total measured workload, including startup and final evaluation | 159.24 s |
| Startup total / vLLM startup | 84.23 s / 67.46 s |
| Average generation + grading per 256 responses | 2.58 s |
| Average rollout persistence per step | 2.11 s |
| Average weight synchronization per step | 0.147 s |
| Average step excluding evaluation and checkpoint | 5.23 s |
| Final checkpoint write + commit | 45.73 s |
| Peak PyTorch allocated memory on training GPU | 14.92 GB |

Step durations were 9.70 s, 4.99 s, and 50.61 s. Step 0 includes 3.90 s
evaluation, and step 2 includes the 45.73 s checkpoint. The training-memory
metric is not the combined allocation of both GPUs and excludes vLLM's GPU 1.

### What the fast generation number actually represents

All **768 training completions** are saved, plus **32 initial and 32 final
evaluation completions**: **832 total records**. Each record includes question,
ground truth, prompt, text, sampled and training token IDs, finish reason,
sampling parameters, seed, verifier outcomes, and training-selection flags.

The training responses averaged **105.26 actual sampled tokens**, totaling
80,838 tokens; the shortest had 6 tokens and the longest 512. **761 stopped**
and **7 hit the length limit**. Therefore this is not a benchmark of 256 responses
all reaching 512 tokens. Token counts after text re-tokenization differ slightly
from original sampled token counts; both representations are preserved.

Only **5 of 768 responses passed the verifier** and were used for RFT updates:
1, 2, and 2 by step. All three steps applied an optimizer update. Low update times
partly reflect skipping the 763 rejected responses. Final pass@1 on 32 questions
was 0%, with 75% format acceptance. This small benchmark does not demonstrate
learning quality or predict mature-policy training throughput.

### Artifact verification

Remote Volume: `olmo-assignment-artifacts`.
Run path: `/rft-bench-h100-03/`.
The final checkpoint `checkpoints/step-000003` contains a `COMPLETE` marker,
`model.safetensors` (2,969,854,224 bytes), `trainer.pt` (5,939,828,927 bytes),
tokenizer and configuration files; `latest.json` points to it. File existence
and sizes were checked, not a full restore/reload test.

Downloaded evidence is preserved in ignored `outputs/assignment-benchmarks/h100-03/`:
provenance/config/latest metadata, the remote file manifest, all five rollout JSON
files, startup/step/final metrics. Original checkpoint weights remain in Modal
storage. Additional Modal application logs and app-lifetime metadata are retained
in `outputs/assignment-benchmarks/`.

## Earlier failures and comparison limits

`rft-bench-b200-02` failed during
vLLM startup: the native FlashInfer backend required `nvcc`, absent from the
slim image. The fix uses a CUDA 12.9 development image while preserving the
native backend. The final paired benchmarks above use that corrected image;
failed attempts are excluded from speed comparisons but included in setup cost.

The old single-H100 nanochat-style run is also not a controlled comparison: it
used another objective/optimizer, 16 samples per question, different generation
and validation settings, and far more responses in its update. Its roughly
90-second steps cannot support a claim that vLLM alone caused the whole speedup.

## Cost comparison

[Modal public prices](https://modal.com/pricing), checked during this task:

| GPU | Per GPU-second | Two GPUs per hour |
|---|---:|---:|
| H100 SXM5 | $0.001097 | $7.8984 |
| B200 | $0.001736 | $12.4992 |
| B300 | $0.001972 | $14.1984 |

At these GPU-only prices, B200 needs **more than 1.58× speedup** over H100 to
cost less for the same work. B300 needs **more than 1.80×**. Both GPUs are billed
while generation/training alternate; the idle counterpart is not free. CPU,
host memory, image-building overhead, storage, and egress are additional.

The measured 159.24-second H100 workload corresponds to **$0.349 GPU-only**.
This is not an invoice or the total cost of debugging the setup. Conservatively
charging both requested GPUs for each entire stopped application lifetime gives:

| Attempt | Whole-app seconds | GPU-only upper estimate |
|---|---:|---:|
| H100-01 (failed) | 144 | $0.316 |
| H100-02 (failed) | 102 | $0.224 |
| B200-01 (failed) | 64 | $0.222 |
| H100-03 (successful) | 199 | $0.437 |
| B200-02 (failed) | 116 | $0.403 |
| Total of these five attempts | | **$1.601** |

These bounds overcount intervals such as image building when the two GPUs may
not have been allocated. The final H100-04 app lifetime was 139 seconds, giving
$0.305 GPU-only. B200-03's whole app lifetime was 607 seconds, giving $2.107,
but this includes the long image build before its function was submitted at
07:33:50 UTC. Counting only the 193 seconds from that submission to app stop
gives a conservative B200-03 function GPU estimate of $0.670. Thus the cumulative
seven-attempt estimate is **$4.014 GPU-only even if all B200 image-build time is
charged as two GPUs**, or **$2.576** when excluding that known pre-submission
interval. These are conservative timing-based estimates, not bills. CPU, RAM,
build costs, storage, and egress are additional; Modal billing is authoritative.

The assignment budgets name B200, not B300. H100 is a reasonable capacity and
compatibility baseline for this model. B300 is not a drop-in choice for this
CUDA 12.9 stack: [Modal's GPU guide](https://modal.com/docs/guide/gpu) requires
CUDA 13.1+ for B300. Benchmarking equal workloads is necessary before claiming
which GPU is cheaper overall.
