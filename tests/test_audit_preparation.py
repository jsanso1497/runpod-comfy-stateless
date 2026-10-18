"""Exercise state publication, async races, hashing and streaming with fakes."""
import asyncio
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import pytest
from workbench import audit_preparation as p
from workbench.audit_defaults import configure_qwen


@pytest.mark.parametrize('policy,expected',[('0',False),('1',True)])
def test_moss_auto_preparation_without_changing_task_selection(monkeypatch,policy,expected):
    monkeypatch.setenv('WB_MOSS_AUTO_PREPARE',policy)
    m={'id':'qwen','default_tasks':['SQ'],'default_groups':['qwen','ollama'],'local_nodes':['qwen_people']}
    configure_qwen(m);configure_qwen(m)
    assert ('moss_image' in m['default_groups']) is expected
    assert m['default_tasks']==['SQ']
    assert m['local_nodes'].count('runtime_compat')==1
    other={'id':'krea'};configure_qwen(other);assert other=={'id':'krea'}


def test_bad_auto_prep_setting_rejected(monkeypatch):
    monkeypatch.setenv('WB_MOSS_AUTO_PREPARE','false')
    with pytest.raises(ValueError):configure_qwen({'id':'qwen'})


def test_tracked_hash_real_bytes_git_and_cache(tmp_path,capsys):
    path=tmp_path/'private-token-never-print.bin';path.write_bytes(b'hello')
    tracker=p.Progress();token=p._CURRENT.set(tracker)
    try:
        assert p.tracked_hash(path)==hashlib.sha256(b'hello').hexdigest()
        assert p.tracked_hash(path,'git')==hashlib.sha1(b'blob 5\0hello').hexdigest()
        before=dict(tracker.hashes)
        assert p.tracked_hash(path)==hashlib.sha256(b'hello').hexdigest()
        assert before==tracker.hashes
        path.write_bytes(b'bye')
        assert p.tracked_hash(path)==hashlib.sha256(b'bye').hexdigest()
        assert tracker.snapshot()['bytes_done']==3
    finally:p._CURRENT.reset(token)
    assert 'private-token' not in capsys.readouterr().out


def fake_assets(tmp_path,fail_second=False):
    model=tmp_path/'models';model.mkdir();state=tmp_path/'state';state.mkdir()
    assets=[{'id':name,'group':'moss_image','destination':name+'.bin','revision':'fixed'} for name in ('first','second')]
    class AssetError(ValueError):pass
    class LinkError(ValueError):pass
    class Transport:
        def close(self):pass
    def atomic(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
    observed=[]
    def transfer(row,target,transport,high_precision):
        observed.append(json.loads((state/'asset-status.json').read_text()))
        if fail_second and target.name=='second.bin':raise AssetError('Transfer failed.')
        target.write_bytes(b'fixture');return {'bytes':7}
    ns={'read_catalog':lambda name:{'assets':assets},'AssetError':AssetError,
        'pd':SimpleNamespace(Transport=Transport,LinkError=LinkError,file_hash=p.tracked_hash),
        'state_root':lambda:state,'model_root':lambda:model,
        'safe_path':lambda root,dest:root/dest,'atomic_json':atomic,
        '_standard_asset_high_precision':lambda asset:True,
        'resolve_standard':lambda a,t:a,'transfer':transfer,
        'inspect_safetensors':lambda path,high_precision:{'bytes':path.stat().st_size}}
    return ns,observed,state,model


def test_success_is_published_before_next_asset(tmp_path):
    ns,observed,state,model=fake_assets(tmp_path)
    result=p.prepare_standard_catalog(ns,['moss_image'])
    assert result['failures']==[]
    assert observed[0]['installed']==[]
    assert [r['id'] for r in observed[1]['installed']]==['first']
    assert len(json.loads((state/'asset-status.json').read_text())['installed'])==2


def test_failed_recheck_cannot_leave_stale_ready_and_retains_other_groups(tmp_path):
    ns,observed,state,model=fake_assets(tmp_path,True)
    previous={'installed':[{'id':'second','destination':'second.bin'}, {'id':'unrelated','destination':'old.bin'}],
              'failures':[{'id':'unrelated_failure','error':'Earlier error.'}]}
    (state/'asset-status.json').write_text(json.dumps(previous))
    result=p.prepare_standard_catalog(ns,['moss_image'])
    assert [f['id'] for f in result['failures']]==['second']
    stored=json.loads((state/'asset-status.json').read_text())
    assert {r['id'] for r in stored['installed']}=={'first','unrelated'}
    assert {r['id'] for r in stored['failures']}=={'second','unrelated_failure'}


def test_corrupt_readiness_not_trusted(tmp_path):
    ns,_,state,_=fake_assets(tmp_path)
    (state/'asset-status.json').write_text('broken')
    assert p.prepare_standard_catalog(ns,['moss_image'])['failures']==[]


def test_unknown_group_never_opens_transport(tmp_path):
    ns,observed,_,_=fake_assets(tmp_path)
    with pytest.raises(ValueError):p.prepare_standard_catalog(ns,['typo'])
    assert observed==[]


def test_cancelled_worker_finishes_before_lock_can_be_released():
    async def run():
        entered=threading.Event();release=threading.Event();finished=threading.Event()
        def work():
            entered.set();release.wait(3);finished.set()
        task=asyncio.create_task(p.wait_worker(work))
        await asyncio.to_thread(entered.wait,1)
        task.cancel();await asyncio.sleep(.02)
        assert not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert finished.is_set()
    asyncio.run(run())


class Response:
    def __init__(self,events,status=200):self.events=events;self.status=status;self.content=self
    async def __aenter__(self):return self
    async def __aexit__(self,*a):pass
    def __aiter__(self):
        async def gen():
            for event in self.events:yield json.dumps(event).encode()+b'\n'
        return gen()


def test_ollama_streams_and_confirms_success():
    called={}
    def post(url,**kwargs):
        called.update(kwargs)
        return Response([{'status':'pulling','completed':20,'total':100},{'status':'success'}])
    asyncio.run(p.pull_ollama(SimpleNamespace(post=post),'public-model',100))
    assert called['json']['stream'] is True


@pytest.mark.parametrize('events',[[{'status':'pulling','completed':2,'total':5}], [{'error':'sensitive provider body'}]])
def test_incomplete_or_failed_ollama_not_marked_success(events):
    with pytest.raises(ValueError) as exc:
        asyncio.run(p.pull_ollama(SimpleNamespace(post=lambda *a,**k:Response(events)),'model',100))
    assert 'sensitive provider body' not in str(exc.value)


def request_env():
    return {'read_catalog':lambda name:{'tasks':[{'id':'QMOSS','workspaces':['qwen'],'requires':['moss_image']}]},
            'model_config':lambda:{'toolbox':['moss_image']},'workspace':lambda:'qwen'}


def test_racing_prepare_requests_only_launch_once():
    async def run():
        ready=asyncio.Event();stop=asyncio.Event();arrivals=[];calls=[]
        async def body():
            arrivals.append(1)
            if len(arrivals)==2:ready.set()
            await ready.wait()
            return {'tasks':['QMOSS'],'private':False}
        async def prepare(groups,private):calls.append(groups);await stop.wait()
        runtime=SimpleNamespace(job=None,prepare=prepare)
        responses=await asyncio.gather(p.request_preparation(runtime,SimpleNamespace(json=body),request_env()),
                                       p.request_preparation(runtime,SimpleNamespace(json=body),request_env()))
        assert sorted(r.status for r in responses)==[202,409]
        await asyncio.sleep(0);assert len(calls)==1
        stop.set();await runtime.job
    asyncio.run(run())


@pytest.mark.parametrize('data',[[],{'tasks':'QMOSS'}, {'tasks':[{}]}, {'private':'false'}, {}, {'tasks':['UNKNOWN']}])
def test_invalid_preparation_schema_never_starts_job(data):
    async def run():
        async def body():return data
        runtime=SimpleNamespace(job=None)
        with pytest.raises(ValueError):await p.request_preparation(runtime,SimpleNamespace(json=body),request_env())
        assert runtime.job is None
    asyncio.run(run())


def test_repeated_cancellation_keeps_worker_alive_until_finished():
    async def run():
        entered=threading.Event();release=threading.Event();finished=threading.Event()
        def work():
            entered.set();release.wait(3);finished.set()
        task=asyncio.create_task(p.wait_worker(work))
        await asyncio.to_thread(entered.wait,1)
        task.cancel();await asyncio.sleep(.01)
        task.cancel();await asyncio.sleep(.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):await task
        assert finished.is_set()
    asyncio.run(run())


@pytest.mark.parametrize('bad',[{'installed':None,'failures':8}, {'installed':'not a list','models':[1,None,{}]}])
def test_corrupt_status_collections_do_not_break_readiness(tmp_path,bad):
    path=tmp_path/'state.json';path.write_text(json.dumps(bad))
    result=p._read_status(path)
    for key in ('installed','failures','models'):
        if key in result:assert isinstance(result[key],list) and all(isinstance(x,dict) for x in result[key])


def test_orchestrator_reports_finish_and_redacts_private_detail(tmp_path,capsys):
    async def run():
        state=tmp_path/'state';state.mkdir()
        calls=[]
        def atomic(path,data):path.write_text(json.dumps(data))
        def standard(groups):
            calls.append(('standard',groups));p.update(stage='verifying checksum',current_asset='public-model',bytes_done=5,bytes_total=10)
            return {'failures':[]}
        def private():calls.append(('private',True));return {'failures':[]}
        ns={'state_root':lambda:state,'atomic_json':atomic,'prepare_standard':standard,'prepare_private':private}
        runtime=SimpleNamespace(job_state={})
        await p.run_preparation(runtime,['moss_image'],True,ns)
        assert calls==[('standard',['moss_image']),('private',True)]
        assert runtime.job_state['state']=='finished' and runtime.job_state['failures']==0
        stored=json.loads((state/'preparation.json').read_text())
        assert stored['progress']['stage']=='finished'
        assert p._CURRENT.get() is None
    asyncio.run(run())
    assert '[PREP] finished' in capsys.readouterr().out


def test_orchestrator_failure_does_not_echo_exception_body(tmp_path,capsys):
    async def run():
        state=tmp_path/'state';state.mkdir()
        def standard(groups):raise RuntimeError('PRIVATE_URL?token=never-log-this')
        ns={'state_root':lambda:state,'atomic_json':lambda path,data:path.write_text(json.dumps(data)),
            'prepare_standard':standard,'prepare_private':lambda:{}}
        runtime=SimpleNamespace(job_state={})
        await p.run_preparation(runtime,['moss_image'],False,ns)
        assert runtime.job_state['state']=='failed'
        assert 'PRIVATE_URL' not in json.dumps(runtime.job_state)
    asyncio.run(run())
    assert 'PRIVATE_URL' not in capsys.readouterr().out
