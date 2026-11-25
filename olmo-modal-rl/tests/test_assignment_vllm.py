import sys
from types import SimpleNamespace, ModuleType
import pytest
import torch
from vendor import assignment_vllm_utils as backend


def test_generation_preserves_order_tokens_and_sampling(monkeypatch):
    requests=[]
    def http(method,url,payload,timeout):
        requests.append(payload)
        n=len(payload['prompt'])*payload['n']
        return {'choices':[{'index':i,'text':str(i),'token_ids':[i+1],'finish_reason':'stop'} for i in reversed(range(n))]}
    monkeypatch.setattr(backend,'_http_json',http)
    server=backend.VLLMServer('model')
    values=server.generate_completions(['a','b','c'],dict(temperature=1.,max_tokens=512,n=2,seed=7,stop=['</answer>'],include_stop_str_in_output=True),batch_size=2)
    assert len(values)==6
    assert [v.token_ids for v in values]==[[1],[2],[3],[4],[1],[2]]
    assert all(p['top_p']==1. and p['top_k']==-1 and p['return_token_ids'] and p['include_stop_str_in_output'] for p in requests)


def test_generation_missing_token_ids_fails(monkeypatch):
    monkeypatch.setattr(backend,'_http_json',lambda *args,**kw:{'choices':[{'index':0,'text':'a'}]})
    with pytest.raises(RuntimeError,match='token_ids'):
        backend.VLLMServer('model').generate_completions(['q'],dict(temperature=1,max_tokens=512,n=1,seed=0))


def test_untrusted_dev_bind_rejected():
    with pytest.raises(ValueError,match='loopback'):
        backend.VLLMServer('model',host='0.0.0.0')


def test_start_uses_uv_current_interpreter_and_gpu1(monkeypatch):
    captured={}
    monkeypatch.setattr(backend.shutil, 'which', lambda name: 'uv')
    # Separate fake running process prevents startup early exit; no real process.
    process=SimpleNamespace(poll=lambda:None,returncode=None)
    def popen(command,**kwargs):
        captured.update(command=command,**kwargs)
        return process
    monkeypatch.setattr(backend.subprocess,'Popen',popen)
    monkeypatch.setattr(backend,'_http_json',lambda *args,**kwargs:{})
    monkeypatch.setattr(backend.atexit,'register',lambda *args:None)
    s=backend.VLLMServer('model',revision='sha',tokenizer_revision='sha')
    s.start()
    assert captured['command'][:5]==['uv','run','--no-project','--python',sys.executable]
    assert captured['env']['CUDA_VISIBLE_DEVICES']=='1'
    assert '--revision' in captured['command']
    assert '18080' in captured['command']


def test_weight_sync_pause_transfer_cache_reset_resume(monkeypatch):
    calls=[]
    def http(method,url,*args,**kwargs):
        calls.append(url.rsplit('/',1)[-1]);return {}
    monkeypatch.setattr(backend,'_http_json',http)
    monkeypatch.setattr(torch.cuda,'set_device',lambda device:None)
    mod=ModuleType('vllm.distributed.weight_transfer.nccl_engine')
    mod.NCCLTrainerSendWeightsArgs=lambda **kwargs:kwargs
    mod.NCCLWeightTransferEngine=SimpleNamespace(trainer_send_weights=lambda **kwargs:list(kwargs['iterator']))
    monkeypatch.setitem(sys.modules,'vllm.distributed.weight_transfer.nccl_engine',mod)
    s=backend.VLLMServer('model');s.weight_sync_group=object()
    s.sync_policy_weights(torch.nn.Linear(2,2))
    assert calls==['pause','update_weights','reset_prefix_cache','resume']
