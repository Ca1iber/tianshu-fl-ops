# SPDX-License-Identifier: Apache-2.0
"""Serialize the three short service probes and verify cleanup between them."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('run_service_probe.py')
    for mode in ('original', 'row', 'packed'):
        out = root / ('service_' + mode)
        out.mkdir(exist_ok=True)
        with (out / 'runner.log').open('w') as log:
            result = subprocess.run([sys.executable, '-u', str(script), '--output', str(out),
                                     '--buckets', '0', '--gather', '0', '--mode', mode],
                                    stdout=log, stderr=subprocess.STDOUT)
        data = json.loads((out / 'result.json').read_text())
        print(json.dumps({'mode': mode, 'exit_code': result.returncode, 'phase': data['phase'],
                          'rounds': len(data['short_4k_rounds']),
                          'later_four': data.get('later_four_stability')}, ensure_ascii=False), flush=True)
        if result.returncode != 0 or data['phase'] != 'complete':
            raise RuntimeError(f'{mode} failed; refusing to start next trial')
        hardware = subprocess.check_output(['ixsmi'], text=True)
        if 'No running processes found' not in hardware:
            raise RuntimeError('GPU is not idle after owned server cleanup')
        if mode != 'original':
            hooks = [json.loads(p.read_text()) for p in out.glob('hook_worker_*.json')]
            active = [d for d in hooks if d['registered_layers']]
            if len(active) != 1 or len(active[0]['live_validations']) != 3:
                raise RuntimeError('Actual live packing path was not validated at all three depths')
            if not all(v['bitwise_equal'] for v in active[0]['live_validations']):
                raise RuntimeError('Actual live packing path failed numerical validation')


if __name__ == '__main__':
    main()
