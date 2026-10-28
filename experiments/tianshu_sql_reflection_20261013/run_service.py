# SPDX-License-Identifier: Apache-2.0
"""Short service workload with isolated SQL reflection and preserved compute."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();assert not root.exists();root.mkdir(parents=True)
    trial=Path('/workspace/vllm-plugin-FL/experiments/tianshu_native_prefill_20261012/run_service_trial.py')
    hook=Path(__file__).resolve().parent/'hooks/sitecustomize.py'
    snapshot=root/'sql_source_snapshot';snapshot.mkdir()
    for p in [Path(__file__),hook,hook.parent.parent/'reflect_one.py',trial]:
        (snapshot/p.name if p.name!='sitecustomize.py' else snapshot/'sql_sitecustomize.py').write_bytes(p.read_bytes())
    original=subprocess.Popen
    def popen(command,*positional,**kwargs):
        if isinstance(command,list) and command[:2]==['vllm','serve']:
            env=dict(kwargs['env']);env.update(PYTHONPATH=str(hook.parent),TIANSHU_SQL_REFLECTION_PROBE='1',
                                             TIANSHU_SQL_REFLECTION_RECORD_DIR=str(root/'sql_records'))
            kwargs['env']=env
        return original(command,*positional,**kwargs)
    spec=importlib.util.spec_from_file_location('_tianshu_sql_service_trial',trial)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    subprocess.Popen=popen
    sys.argv=[str(trial),'--output',str(root/'candidate'),'--buckets','1','--gather','1','--bucket-trace','2',
              '--hybrid','1','--adaptive','1','--tile','1','--native-prefill','1','--decode-launch','1',
              '--max-batched-tokens','2048','--hist-count-bits','off','--address32','0','--requests','64','--rounds','3']
    try:code=module.main()
    finally:subprocess.Popen=original
    result=json.loads((root/'candidate/result.json').read_text())
    routes=[json.loads(p.read_text()) for p in (root/'sql_records').glob('route_*.json')]
    state=dict(phase=result['phase'],official_benchmark=False,adopted=False,compute_profile_unmodified=True,
               sql_reflection_routes=routes,workload=result['workload'])
    if code==0:assert routes,'SQL override did not run'
    (root/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    return code


if __name__=='__main__':raise SystemExit(main())
