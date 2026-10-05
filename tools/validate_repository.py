#!/usr/bin/env python3
"""Offline source/layout, Dockerfile, shell, workflow and widget checks.

Run locally from the repository root with Python, PyYAML, requests, and the
package's existing test dependencies installed. This does not build Docker.
"""
from __future__ import annotations
import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import yaml

ROOT=Path(__file__).resolve().parents[1]
DIRECTIVES={'ARG','FROM','SHELL','ENV','COPY','ADD','RUN','LABEL','EXPOSE','WORKDIR',
            'HEALTHCHECK','CMD','ENTRYPOINT','USER','VOLUME','STOPSIGNAL','ONBUILD'}


def docker_commands(path):
    pending=''
    for number,line in enumerate(path.read_text().splitlines(),1):
        if not pending and (not line.strip() or line.lstrip().startswith('#')):continue
        part=line.rstrip()
        pending+=part[:-1]+' ' if part.endswith('\\') else part
        if part.endswith('\\'):continue
        word,_,body=pending.strip().partition(' ')
        if word not in DIRECTIVES:
            raise ValueError(f'{path}:{number}: invalid Docker directive {word!r}')
        yield word,body
        pending=''
    if pending:raise ValueError(f'{path}: incomplete continuation')


def main():
    counts={}
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p),feature_version=(3,10))
        elif p.suffix=='.json':json.loads(p.read_text())
        elif p.suffix=='.sh':subprocess.run(['bash','-n',str(p)],check=True)
        if p.suffix in ('.py','.sh','.json','.yml','.js','.txt') and b'\r\n' in p.read_bytes():
            raise ValueError('Unexpected CRLF source: '+str(p))
    counts['python_sources']=sum(p.suffix=='.py' for p in files)
    counts['json_files']=sum(p.suffix=='.json' for p in files)
    counts['shell_scripts']=sum(p.suffix=='.sh' for p in files)
    for name in ('Dockerfile','h3_portrait/Dockerfile'):
        cmds=list(docker_commands(ROOT/name))
        for word,body in cmds:
            if word=='RUN':subprocess.run(['bash','-n'],input=body,text=True,check=True)
            if word=='COPY' and not body.startswith('--'):
                for src in body.split()[:-1]:
                    if not (ROOT/src).exists():raise ValueError('Missing Docker COPY source: '+src)
        assert not any(word=='ARG' and 'BASE_IMAGE=' in body and ':latest' in body for word,body in cmds)
    counts['dockerfiles_checked']=2
    for name,target in [('build-image.yml','general'),('build-h3-portrait.yml','portrait')]:
        p=ROOT/'.github/workflows'/name
        data=yaml.load(p.read_text(),Loader=yaml.BaseLoader)
        assert {'push','workflow_dispatch'} <= set(data['on'])
        assert 'shared_loras/**' in data['on']['push']['paths']
        steps=data['jobs']['build']['steps']
        build=next(s for s in steps if s.get('uses')=='docker/build-push-action@v6')
        assert build['with']['context']=='.'
        if target=='portrait':assert build['with']['file']=='./h3_portrait/Dockerfile'
        assert 'SOURCE_REVISION=${{ github.sha }}' in build['with']['build-args']
        assert 'cache-from' not in build['with'] and 'cache-to' not in build['with']
        for i,step in enumerate(steps):
            if 'run' in step:
                # Replace only GitHub expression placeholders before shell parsing.
                script=re.sub(r'\$\{\{.*?\}\}','PLACEHOLDER',step['run'])
                subprocess.run(['bash','-n'],input=script,text=True,check=True)
        assert any('tools/check_repository.py --target '+target in s.get('run','') for s in steps)
    counts['actions_workflows_checked']=2
    # Same static graph implementation used by the packaged build.
    spec=importlib.util.spec_from_file_location('general_graph_checks',ROOT/'scripts/check_workflows.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for p in (ROOT/'config/workflows').glob('*.json'):
        # Inspect the function's actual signature instead of assuming it.
        import inspect
        sig=inspect.signature(module.validate_graph)
        data=json.loads(p.read_text())
        if len(sig.parameters)==1:module.validate_graph(data)
        else:module.validate_graph(data,p.name)
    spec=importlib.util.spec_from_file_location('portrait_graph_checks',ROOT/'h3_portrait/smoke_check.py')
    portrait=importlib.util.module_from_spec(spec);spec.loader.exec_module(portrait)
    portrait.validate_graph(json.loads((ROOT/'h3_portrait/workflows/H3_Portrait_Auto.json').read_text()))
    counts['graphs_checked']=len(list((ROOT/'config/workflows').glob('*.json')))+1
    print(json.dumps(counts,indent=2))
    print('STATIC REPOSITORY CHECKS PASS. No Docker build or GPU inference performed.')


if __name__=='__main__':main()
