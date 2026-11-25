"""Run: modal run --detach modal_app.py --run-name olmo-gsm8k-001"""
import re
from pathlib import Path
import modal

app = modal.App('olmo-gsm8k-nanochat-rl')
image = (modal.Image.debian_slim(python_version='3.13')
         .uv_pip_install('torch==2.9.1', 'transformers==4.57.3', 'datasets==4.4.1', 'wandb==0.23.0',
                      'math-verify[antlr4-13-2]==0.8.0', 'pylatexenc==2.10')
         .env({'HF_HOME': '/cache/huggingface', 'TOKENIZERS_PARALLELISM': 'false'})
         .add_local_file(Path(__file__).parent / 'train.py', '/root/project/train.py')
         .add_local_dir(Path(__file__).parent / 'vendor', '/root/project/vendor'))
checkpoints = modal.Volume.from_name('olmo-gsm8k-checkpoints', create_if_missing=True)
cache = modal.Volume.from_name('olmo-gsm8k-cache', create_if_missing=True)

@app.function(image=image, gpu='H100', cpu=8, memory=65536, timeout=86400,
              volumes={'/runs': checkpoints, '/cache': cache},
              secrets=[modal.Secret.from_name('wandb_sec', required_keys=['WANDB_API_KEY'])])
def train_remote(run_name: str, smoke: bool = False, resume: bool = False):
    import sys
    sys.path.insert(0, '/root/project')
    from train import run
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_name):
        raise ValueError('Use a simple unique run name (letters, digits, underscore, hyphen)')
    return run(run_name, f'/runs/{run_name}', smoke=smoke, resume=resume, commit=checkpoints.commit)

@app.local_entrypoint()
def main(run_name: str = 'olmo-gsm8k-001', smoke: bool = False, resume: bool = False):
    call = train_remote.spawn(run_name, smoke, resume)
    print(f'Submitted training call: {call.object_id}', flush=True)
