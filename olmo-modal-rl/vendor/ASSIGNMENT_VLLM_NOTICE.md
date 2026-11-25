# Assignment-compatible vLLM transport

`assignment_vllm_utils.py` is a local implementation of the documented server
interface and vLLM NCCL transport used by Stanford CS336 Spring 2026 Assignment 5.
It is not a verbatim vendoring of the upstream helper. The upstream repository's
root tree at the pinned revision did not contain a license file; no claim of an
upstream software license is made here.

Reference implementation inspected:
https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/cs336_alignment/vllm_utils.py

Assignment source revision: `c2734a26308710949fe13226960a1e8cece94b7e`.
Backend protocol target: `vllm==0.19.1` with `torch==2.10.0`.

Preserved interface: `VLLMServer.start`, `stop`, `init_weight_sync`,
`sync_policy_weights`, and `generate_completions`; result fields `text`,
`token_ids`, `finish_reason`. Trainer uses GPU 0, inference uses GPU 1, tensor
parallelism is one, weights transfer through NCCL, and prefix caches are reset
before the server resumes generation.

Local choices: loopback-only port 18080; use `uv run --no-project` with the current
interpreter; pin model/tokenizer revisions; constrain maximum context; propagate
top-p and top-k explicitly; optionally retain server logs; never kill unrelated
processes; reject incomplete rollout groups or missing sampled token IDs.

Upstream pins Python >=3.12,<3.13, including a cp312-only optional flash-attn wheel.
vLLM 0.19.1 itself advertises Python >=3.10,<3.14 and abi3 Linux wheels. Python
3.13 with SDPA training therefore differs from the assignment environment and
requires separate GPU compatibility validation. This transport has not by itself
validated an image build, GPU kernel execution, or NCCL transfer on Modal.

## CUDA build toolkit

The initial Debian slim image failed on B200 because vLLM selected FlashInfer's
TRTLLM attention path, which attempted to compile a kernel but could not find
`nvcc`. The remote image now uses the official NVIDIA
`nvidia/cuda:12.9.1-devel-ubuntu24.04` base, manifest digest
`sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c`,
with Python 3.13 added by Modal and `CUDA_HOME=/usr/local/cuda`. An image-build
check runs `nvcc --version`. The default attention backend remains automatic so
B200 can use its native FlashInfer path and H100 its native FlashAttention path.
This correction is a prerequisite fix, not proof of a successful B200 benchmark.
