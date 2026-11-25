"""CS336 RFT/GRPO on two GPUs. Importing this file launches no compute."""
from dataclasses import asdict, replace
from pathlib import Path
import re

import modal

ROOT = Path(__file__).parent
app = modal.App('olmo-cs336-two-gpu')
image = (
    # B200 selects FlashInfer/TRTLLM, whose kernels require nvcc at runtime.
    # Official CUDA 12.9.1 devel Ubuntu 24.04 image, pinned by manifest digest.
    modal.Image.from_registry(
        'nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c',
        add_python='3.13',
    )
    .apt_install('libnuma1')
    .env({'CUDA_HOME': '/usr/local/cuda'})
    .run_commands('/usr/local/cuda/bin/nvcc --version')
    .uv_pip_install('torch==2.10.0+cu129', 'torchvision==0.25.0+cu129',
                    'torchaudio==2.10.0+cu129',
                    index_url='https://download.pytorch.org/whl/cu129')
    .uv_pip_install('vllm==0.19.1', requirements=[str(ROOT / 'requirements-assignment.txt')])
    .env({'HF_HOME': '/cache/huggingface', 'TOKENIZERS_PARALLELISM': 'false',
          'OMP_NUM_THREADS': '4'})
    .add_local_file(ROOT / 'assignment_train.py', '/root/project/assignment_train.py')
    .add_local_file(ROOT / 'requirements-assignment.txt', '/root/project/requirements-assignment.txt')
    .add_local_dir(ROOT / 'vendor', '/root/project/vendor', ignore=['__pycache__', '*.pyc'])
)
artifacts = modal.Volume.from_name('olmo-assignment-artifacts', create_if_missing=True)
cache = modal.Volume.from_name('olmo-gsm8k-cache', create_if_missing=True)
options = dict(image=image, cpu=8, memory=65536,
               volumes={'/runs': artifacts, '/cache': cache},
               secrets=[modal.Secret.from_name('wandb_sec', required_keys=['WANDB_API_KEY'])],
               max_containers=1, scaledown_window=2, retries=0)


def execute(run_name, mode, seed, resume=False, benchmark=False, steps=200):
    import os
    import sys
    # uv's cache locks/atomic linking require a local POSIX filesystem, not a Volume.
    os.environ['UV_CACHE_DIR'] = '/tmp/uv-cache'
    sys.path.insert(0, '/root/project')
    from assignment_train import AssignmentConfig, run
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', run_name):
        raise ValueError('Use a unique alphanumeric run name with hyphens/underscores')
    config = AssignmentConfig(mode=mode, seed=seed, num_rollout_steps=steps)
    if benchmark:
        # Same real model, full rollout batch and generation length on both GPUs.
        # Reduced evaluation/step counts are diagnostic, not assignment results.
        config = replace(config, n_val_examples=32, num_rollout_steps=3,
                         checkpoint_every=10)
    print(f'Configuration: {asdict(config)}', flush=True)
    return run(run_name, f'/runs/{run_name}', config, resume=resume, commit=artifacts.commit)


@app.function(gpu='H100:2', timeout=900, **options)
def benchmark_h100(run_name: str, seed: int = 0):
    return execute(run_name, 'rft', seed, benchmark=True)


@app.function(gpu='B200:2', timeout=900, **options)
def benchmark_b200(run_name: str, seed: int = 0):
    return execute(run_name, 'rft', seed, benchmark=True)


@app.function(gpu='H100:2', timeout=86400, **options)
def train_h100(run_name: str, mode: str = 'rft', seed: int = 0,
               resume: bool = False, steps: int = 200):
    return execute(run_name, mode, seed, resume=resume, steps=steps)


@app.function(gpu='B200:2', timeout=86400, **options)
def train_b200(run_name: str, mode: str = 'rft', seed: int = 0,
               resume: bool = False, steps: int = 200):
    return execute(run_name, mode, seed, resume=resume, steps=steps)


@app.local_entrypoint()
def main(run_name: str, gpu: str = 'H100', benchmark: bool = False,
         mode: str = 'rft', seed: int = 0, resume: bool = False, steps: int = 200):
    if gpu not in ('H100', 'B200') or mode not in ('rft', 'grpo'):
        raise ValueError('gpu must be H100/B200 and mode must be rft/grpo')
    if benchmark:
        if resume or mode != 'rft':
            raise ValueError('Benchmarks use fresh RFT runs')
        function = benchmark_h100 if gpu == 'H100' else benchmark_b200
        call = function.spawn(run_name, seed)
    else:
        function = train_h100 if gpu == 'H100' else train_b200
        call = function.spawn(run_name, mode, seed, resume, steps)
    print(f'Submitted {gpu}:2 call {call.object_id}; completion is tracked remotely.', flush=True)
