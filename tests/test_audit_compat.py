"""CPU numerical checks of CUDA-12 fallback dispatch, not GPU certification."""
from types import SimpleNamespace
import pytest
import torch
from workbench import cuda_compat as c


@pytest.mark.parametrize('cuda,driver,policy,expected', [
    ('12.8',12080,'auto',True),('12.8',13000,'auto',True),
    ('13.0',12080,'auto',True),('13.0',13000,'auto',False),
    ('13.1',None,'auto',True),(None,None,'auto',False),
    ('13.0',13000,'safe',True),('12.8',12080,'off',False),
    ('broken',13000,'auto',True)])
def test_decision(cuda,driver,policy,expected):
    assert c.needs_fallback(cuda,driver,policy) is expected


def test_invalid_policy_is_not_silent():
    with pytest.raises(ValueError):c.needs_fallback('12.8',12080,'maybe')


@pytest.mark.parametrize('batch',[1,2])
@pytest.mark.parametrize('seq',[1,2,5])
@pytest.mark.parametrize('ratio',[1,2,4])
def test_fixed_cache_matches_explicit_causal_gqa_reference(batch,seq,ratio):
    torch.manual_seed(22)
    h,d,capacity=2,8,16
    q=torch.randn(batch,h*ratio,seq,d,dtype=torch.float64)
    k=torch.randn(batch,h,capacity,d,dtype=torch.float64)
    v=torch.randn_like(k)
    seqlen=torch.arange(batch)+8
    def mask(n):
        rel=torch.arange(capacity)-seqlen.view(-1,1,1,1)
        visible=rel<torch.arange(1-n,1).view(n,1)
        return torch.zeros(visible.shape,dtype=k.dtype).masked_fill(~visible,float('-inf'))
    cache=SimpleNamespace(key=k,value=v,mask=mask)
    actual=c.fixed_cache_sdpa(cache,q)
    kk=k.repeat_interleave(ratio,1);vv=v.repeat_interleave(ratio,1)
    scores=q@kk.transpose(-1,-2)/(d**.5)+mask(seq)
    expected=(scores.softmax(-1)@vv).transpose(1,2).reshape(batch,seq,-1)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    assert actual.shape==(batch,seq,h*ratio*d)


def fake_modules():
    class Attn:
        def _load_from_state_dict(self,*a,**k):
            self.method=a[0];self.function='UNSAFE'
    class Cache:
        mask=lambda self,n: None
        attend=lambda self,*a: 'UNSAFE'
    class Gated:
        kv_cache_type=lambda self,*a: 'UNSAFE'
    ck=SimpleNamespace(int8_attention_is_available=lambda *a:True,
        flash_attention_decode_is_available=lambda *a:True,
        sol_attn_is_available=lambda *a:True)
    attention=SimpleNamespace(ComfyAttention=Attn,attention_pytorch=object(),
        COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE=True)
    return ck,attention,SimpleNamespace(FixedKVCache=Cache),SimpleNamespace(GatedAttention=Gated),SimpleNamespace(args=SimpleNamespace(use_ck_attention=False))


def test_cpu_runtime_dispatch_and_idempotence(monkeypatch):
    monkeypatch.setattr(c,'_INSTALLED',False)
    modules=fake_modules();ck,a,ll,qw,mm=modules
    report=c.install_compat(modules=modules,cuda_version='12.8',driver_version=12080)
    assert report['active']
    assert not ck.int8_attention_is_available()
    assert not ck.flash_attention_decode_is_available()
    obj=a.ComfyAttention();obj._load_from_state_dict('comfy_kitchen_int8')
    assert obj.method is None and obj.function is None
    obj._load_from_state_dict('comfy_kitchen_sol');assert obj.function is None
    obj._load_from_state_dict('unrelated');assert obj.method=='unrelated'
    assert qw.GatedAttention().kv_cache_type('cuda',torch.bfloat16) is ll.FixedKVCache
    assert ll.FixedKVCache.attend is c.fixed_cache_sdpa
    assert c.install_compat(modules=modules)['already_installed']


def test_modern_runtime_untouched(monkeypatch):
    monkeypatch.setattr(c,'_INSTALLED',False)
    modules=fake_modules();original=modules[2].FixedKVCache.attend
    report=c.install_compat(modules=modules,cuda_version='13.0',driver_version=13000)
    assert not report['active']
    assert modules[2].FixedKVCache.attend is original


def test_force_ck_flag_rejected_before_mutation(monkeypatch):
    monkeypatch.setattr(c,'_INSTALLED',False)
    modules=fake_modules();modules[4].args.use_ck_attention=True
    original=modules[1].ComfyAttention._load_from_state_dict
    with pytest.raises(RuntimeError):c.install_compat(modules=modules,cuda_version='12.8',driver_version=12080)
    assert modules[1].ComfyAttention._load_from_state_dict is original
    assert modules[0].int8_attention_is_available()
