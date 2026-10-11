# SPDX-License-Identifier: Apache-2.0
"""Cycle 42 actual weight matrices to check isolated hot-weight bias."""

import argparse
import gc
import json
from pathlib import Path
import signal
import sys
import time

import torch
import flag_gems
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tianshu_packing.matched_gemm import launch
from tianshu_packing.packed_bundle import pack_linear_weight
from tianshu_packing.run_packing_probe import (
    agreement, captured, cpu_reference_samples, interleaved, write_json,
)


def load_layers(kind):
    result = []
    with safe_open('/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors',
                   framework='pt', device='cpu') as handle:
        for layer in range(42):
            prefix = f'model.layers.{layer}.'
            if kind == 'qkv':
                weight = torch.cat([handle.get_tensor(prefix + 'self_attn.' + name + '_proj.weight')
                                    for name in ('q', 'k', 'v')])
            else:
                suffix = 'mlp.down_proj.weight' if kind == 'down' else 'self_attn.o_proj.weight'
                weight = handle.get_tensor(prefix + suffix)
            result.append(weight.cuda())
    return result


def test(kind, m, config):
    weights = load_layers(kind)
    n, k = weights[0].shape
    x = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    outs = {key: torch.empty((m, n), dtype=x.dtype, device=x.device)
            for key in ('native', 'row', 'packed')}
    packed = [pack_linear_weight(w, panel_n=config[1], block_k=config[2]) for w in weights]
    validations = []
    for index in (0, 21, 41):
        w = weights[index]
        a = flag_gems.mm_out(x, w.T, out=outs['native']).clone()
        row = launch(x, w, outs['row'], config, False)
        p = launch(x, packed[index], outs['packed'], config, True)
        rows, cols, cpu = cpu_reference_samples(x, w)
        validations.append({'layer': index, 'bitwise_matched': bool(torch.equal(row, p)),
                            'vs_native': agreement(p, a),
                            'vs_cpu_fp32_samples': agreement(p.cpu()[rows][:, cols], cpu)})

    def native():
        for w in weights:
            flag_gems.mm_out(x, w.T, out=outs['native'])
        return outs['native']

    def row():
        for w in weights:
            launch(x, w, outs['row'], config, False)
        return outs['row']

    def pack():
        for w in packed:
            launch(x, w, outs['packed'], config, True)
        return outs['packed']

    graphs = {name: captured(fn)[0] for name, fn in (('native', native), ('row', row), ('packed', pack))}
    timing = interleaved(graphs, repetitions=10, trials=9)
    return {'kind': kind, 'shape': [m, n, k], 'layers_per_replay': 42,
            'config': list(config), 'timing_42_matrices': timing,
            'row_vs_native_speedup': timing['native']['median_ms'] / timing['row']['median_ms'],
            'packed_vs_native_speedup': timing['native']['median_ms'] / timing['packed']['median_ms'],
            'packed_vs_row_speedup': timing['row']['median_ms'] / timing['packed']['median_ms'],
            'additional_weight_bytes': sum(w.numel() * w.element_size() for w in packed),
            'validation': validations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    signal.alarm(1800)
    torch.set_num_threads(8)
    torch.manual_seed(2026101121)
    torch.backends.cuda.matmul.allow_tf32 = False
    data = {'official_benchmark': False,
            'note': '42 independent real weight GEMMs per replay, with the same input. No attention, KV or layer-to-layer activations.',
            'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'cases': []}
    with torch.inference_mode():
        for kind, m, config in [('down', 2048, (128, 128, 32, 8, 2)),
                                ('qkv', 2048, (128, 128, 32, 8, 2)),
                                ('o_proj', 2048, (128, 128, 32, 8, 2)),
                                ('qkv', 64, (64, 64, 64, 4, 2))]:
            case = test(kind, m, config)
            data['cases'].append(case)
            write_json(args.output / 'result.json', data)
            print(json.dumps(case), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    data['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write_json(args.output / 'result.json', data)


if __name__ == '__main__':
    main()
