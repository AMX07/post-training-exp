"""Two-GPU vLLM transport matching the CS336 Spring 2026 assignment interface.

Local implementation; provenance and intentional differences are documented in
ASSIGNMENT_VLLM_NOTICE.md. The development weight-transfer API is loopback-only.
"""
from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import urllib.request

import torch


def _assert_loopback(host: str) -> None:
    if host == "localhost":
        return
    if not ipaddress.ip_address(host).is_loopback:
        raise ValueError("The unauthenticated vLLM development server must bind to loopback")


def _http_json(method: str, url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    # Calls in this module are to our loopback server, never external services.
    from urllib.parse import urlsplit
    _assert_loopback(urlsplit(url).hostname or "")
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        result = response.read()
    return json.loads(result) if result else {}


@dataclass
class VLLMCompletion:
    text: str
    token_ids: list[int]
    finish_reason: str | None


@dataclass
class VLLMServer:
    model_id: str
    host: str = "127.0.0.1"
    port: int = 18080
    gpu: int = 1
    seed: int = 0
    load_format: str = "auto"
    logging_level: str = "INFO"
    gpu_memory_utilization: float = 0.85
    launch_server: bool = True
    startup_timeout: int = 1200
    shutdown_timeout: int = 30
    revision: str | None = None
    tokenizer_revision: str | None = None
    max_model_len: int = 2048
    server_log_path: str | None = None

    def __post_init__(self) -> None:
        _assert_loopback(self.host)
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None
        self.weight_sync_group = None
        self._log_file = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("This server instance has already been started")
        if self.launch_server:
            env = dict(os.environ)
            env.update(CUDA_VISIBLE_DEVICES=str(self.gpu), VLLM_SERVER_DEV_MODE="1",
                       VLLM_LOGGING_LEVEL=self.logging_level)
            uv_command = shutil.which('uv') or ('/.uv/uv' if Path('/.uv/uv').is_file() else None)
            if uv_command is None:
                raise RuntimeError('uv is required to launch the vLLM subprocess')
            command = [uv_command, "run", "--no-project", "--python", sys.executable,
                       "vllm", "serve", self.model_id, "--host", self.host,
                       "--port", str(self.port), "--dtype", "bfloat16",
                       "--enable-prefix-caching", "--gpu-memory-utilization",
                       str(self.gpu_memory_utilization), "--seed", str(self.seed),
                       "--tensor-parallel-size", "1", "--max-model-len", str(self.max_model_len),
                       "--weight-transfer-config", json.dumps({"backend": "nccl"}),
                       "--load-format", self.load_format]
            if self.revision:
                command += ["--revision", self.revision]
            if self.tokenizer_revision:
                command += ["--tokenizer-revision", self.tokenizer_revision]
            if self.server_log_path:
                log_path = Path(self.server_log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_file = log_path.open("a", buffering=1)
            try:
                self.process = subprocess.Popen(command, env=env, start_new_session=True,
                                                stdout=self._log_file, stderr=subprocess.STDOUT)
            except BaseException:
                if self._log_file:
                    self._log_file.close()
                raise
            atexit.register(self.stop)
        deadline = time.monotonic() + self.startup_timeout
        try:
            while time.monotonic() < deadline:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError(f"vLLM exited during startup: {self.process.returncode}; inspect server logs")
                try:
                    _http_json("GET", self.base_url + "/health", timeout=5)
                    return
                except OSError:
                    time.sleep(2)
            raise TimeoutError("vLLM health check timed out")
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        process = self.process
        try:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
        finally:
            if self._log_file is not None and not self._log_file.closed:
                self._log_file.close()

    def init_weight_sync(self, policy_device: str):
        """Connect trainer rank 0 to vLLM rank 1 using its supported NCCL API."""
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
        from vllm.utils.network_utils import get_ip, get_open_port
        world_size = _http_json("GET", self.base_url + "/get_world_size", timeout=10)["world_size"] + 1
        if world_size != 2:
            raise ValueError("Expected one vLLM GPU and one trainer GPU")
        info = dict(master_address=get_ip(), master_port=get_open_port(), world_size=world_size)
        torch.cuda.set_device(torch.device(policy_device))
        with ThreadPoolExecutor(max_workers=1) as pool:
            remote = pool.submit(_http_json, "POST", self.base_url + "/init_weight_transfer_engine",
                                 {"init_info": dict(info, rank_offset=1)}, 60)
            self.weight_sync_group = NCCLWeightTransferEngine.trainer_init(info)
            remote.result()
        return self.weight_sync_group

    def sync_policy_weights(self, policy: torch.nn.Module) -> None:
        """Synchronize current policy before generating another rollout batch."""
        if self.weight_sync_group is None:
            raise RuntimeError("init_weight_sync must precede sync_policy_weights")
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
        weights = list(policy.named_parameters())
        update = dict(names=[name for name, _ in weights],
                      dtype_names=[str(value.dtype).split(".")[-1] for _, value in weights],
                      shapes=[list(value.shape) for _, value in weights], packed=True)
        torch.cuda.set_device(next(policy.parameters()).device)
        _http_json("POST", self.base_url + "/pause")
        with ThreadPoolExecutor(max_workers=1) as pool:
            remote = pool.submit(_http_json, "POST", self.base_url + "/update_weights", {"update_info": update}, 300)
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(weights),
                trainer_args=NCCLTrainerSendWeightsArgs(group=self.weight_sync_group, packed=True))
            remote.result()
        # On any sync failure, propagate instead of generating from partly copied weights.
        _http_json("POST", self.base_url + "/reset_prefix_cache")
        _http_json("POST", self.base_url + "/resume")

    def generate_completions(self, prompts: list[str], sampling_params: dict,
                             batch_size: int | None = None) -> list[VLLMCompletion]:
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not prompts:
            return []
        width = len(prompts) if batch_size is None else batch_size
        results = []
        for start in range(0, len(prompts), width):
            chunk = prompts[start:start + width]
            request = dict(model=self.model_id, prompt=chunk, return_token_ids=True,
                           add_special_tokens=False,
                           temperature=sampling_params["temperature"], max_tokens=sampling_params["max_tokens"],
                           n=sampling_params["n"], seed=sampling_params["seed"],
                           top_p=sampling_params.get("top_p", 1.0), top_k=sampling_params.get("top_k", -1))
            if "stop" in sampling_params:
                request.update(stop=sampling_params["stop"],
                               include_stop_str_in_output=sampling_params.get("include_stop_str_in_output", False))
            response = _http_json("POST", self.base_url + "/v1/completions", request, timeout=3600)
            choices = sorted(response["choices"], key=lambda choice: choice["index"])
            if len(choices) != len(chunk) * request["n"]:
                raise RuntimeError("vLLM returned an incomplete rollout group")
            for choice in choices:
                if "token_ids" not in choice or choice["token_ids"] is None:
                    raise RuntimeError("vLLM did not return token_ids; refusing to retokenize sampled actions")
                results.append(VLLMCompletion(choice["text"], choice["token_ids"], choice.get("finish_reason")))
        return results
