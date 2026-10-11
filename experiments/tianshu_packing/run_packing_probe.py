# SPDX-License-Identifier: Apache-2.0
"""Short, serialized BI-V150 packing experiments using actual model weights.

This does not change installed packages, model weights or serving dispatch.
All times are isolated CUDA Graph replay measurements, not serving scores.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

import torch
import flag_gems
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tianshu_packing import packed_bundle as bundle
from tianshu_packing import packed_swiglu as swiglu
from tianshu_packing.matched_gemm import launch

ATOL, RTOL = 0.02, 0.01
CONFIGS = [
    (128, 128, 32, 8, 2),  # exact MetaX projection configuration
    (64, 128, 64, 4, 2),
    (64, 64, 64, 4, 2),
    (128, 128, 64, 8, 2),
    (32, 128, 32, 4, 2),
]


def write_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def agreement(actual, expected):
    a, b = actual.detach().float(), expected.detach().float()
    error = (a - b).abs()
    failures = error > (ATOL + RTOL * b.abs())
    return {
        'passed': bool(torch.isfinite(a).all() and torch.isfinite(b).all() and not failures.any()),
        'max_abs': error.max().item(),
        'rms_error': error.square().mean().sqrt().item(),
        'failed_elements': failures.sum().item(),
        'atol': ATOL, 'rtol': RTOL,
    }


def captured(fn):
    # Tune/compile before entering capture. Graphs keep all allocations alive.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def measure(graph, repetitions=30, trials=5):
    values = []
    for _ in range(trials):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / repetitions)
    return {'median_ms': statistics.median(values), 'samples_ms': values,
            'min_ms': min(values), 'max_ms': max(values)}


def interleaved(graphs, repetitions=30, trials=7):
    values = {name: [] for name in graphs}
    names = list(graphs)
    for trial in range(trials):
        order = names[trial % len(names):] + names[:trial % len(names)]
        for name in order:
            timing = measure(graphs[name], repetitions=repetitions, trials=1)
            values[name].append(timing['median_ms'])
    return {name: {'median_ms': statistics.median(v), 'samples_ms': v}
            for name, v in values.items()}


def load_weights():
    shard = '/workspace/MiniCPM5-2B/model-00000-of-00001.safetensors'
    with safe_open(shard, framework='pt', device='cpu') as handle:
        def get(suffix):
            return handle.get_tensor('model.layers.0.' + suffix).cuda()
        qkv = torch.cat([get('self_attn.' + name + '_proj.weight') for name in ('q', 'k', 'v')])
        gate_up = torch.cat([get('mlp.' + name + '_proj.weight') for name in ('gate', 'up')])
        return {'qkv': qkv, 'o_proj': get('self_attn.o_proj.weight'),
                'gate_up': gate_up, 'down': get('mlp.down_proj.weight'),
                'lm_head': handle.get_tensor('lm_head.weight').cuda()}


def cpu_reference_samples(x, weight):
    # CPU FP32 matmul uses an independent implementation and selected rows/columns.
    rows = torch.linspace(0, x.shape[0] - 1, min(8, x.shape[0])).long()
    cols = torch.linspace(0, weight.shape[0] - 1, min(128, weight.shape[0])).long()
    xs = x.detach().cpu()[rows].float()
    ws = weight.detach().cpu()[cols].float()
    return rows, cols, xs @ ws.T


def baseline_gemm(x, weight, out, head=False):
    if head:
        return flag_gems.linear(x, weight)
    return flag_gems.mm_out(x, weight.T, out=out)


def matrix_case(name, m, weight, sweep):
    n, k = weight.shape
    x = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    out = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    head = name.startswith('lm_head')
    rows, cols, cpu = cpu_reference_samples(x, weight)
    baseline = lambda: baseline_gemm(x, weight, out, head=head)
    native_graph, native = captured(baseline)
    record = {'name': name, 'shape': [m, n, k],
              'native_vs_cpu_fp32_samples': agreement(native.cpu()[rows][:, cols], cpu),
              'native': measure(native_graph), 'variants': []}

    # Unchanged MetaX kernel/configuration first: evaluate the direct port.
    if m == 2048 and name in ('qkv', 'o_proj'):
        packed = bundle.pack_linear_weight(weight, panel_n=128, block_k=32)
        port = lambda: bundle.packed_projection(x, packed, out=out)
    elif head and m in (1, 8, 32, 64):
        packed = bundle.pack_linear_weight(weight, panel_n=128, block_k=32)
        port = lambda: bundle.packed_lm_head(x, packed, out=out)
    else:
        port = None
    if port is not None:
        try:
            native_copy = baseline().clone()
            candidate_graph, candidate = captured(port)
            result = {'native': measure(native_graph), 'packed': measure(candidate_graph),
                      'vs_native': agreement(candidate, native_copy),
                      'vs_cpu_fp32_samples': agreement(candidate.cpu()[rows][:, cols], cpu)}
            record['direct_metax_port'] = result
            print('DIRECT', name, json.dumps(result), flush=True)
            del candidate_graph, candidate
        except Exception:
            record['direct_metax_port'] = {'error': traceback.format_exc()}
            print('DIRECT_ERROR', name, record['direct_metax_port']['error'], flush=True)

    if sweep:
        for cfg in CONFIGS:
            packed = bundle.pack_linear_weight(weight, panel_n=cfg[1], block_k=cfg[2])
            row_out, packed_out = torch.empty_like(out), torch.empty_like(out)
            entry = {'config': list(cfg), 'packed_weight_bytes': packed.numel() * packed.element_size()}
            try:
                row_graph, row = captured(lambda: launch(x, weight, row_out, cfg, False))
                pack_graph, pack = captured(lambda: launch(x, packed, packed_out, cfg, True))
                # The matched paths should be bitwise identical, including BF16 stores.
                entry['bitwise_matched'] = bool(torch.equal(row, pack))
                entry['vs_native'] = agreement(pack, baseline().clone())
                entry['vs_cpu_fp32_samples'] = agreement(pack.cpu()[rows][:, cols], cpu)
                entry['timing'] = interleaved({'native': native_graph, 'row': row_graph, 'packed': pack_graph})
                entry['packed_vs_row_speedup'] = entry['timing']['row']['median_ms'] / entry['timing']['packed']['median_ms']
                entry['packed_vs_native_speedup'] = entry['timing']['native']['median_ms'] / entry['timing']['packed']['median_ms']
                del row_graph, pack_graph, row, pack
            except Exception:
                entry['error'] = traceback.format_exc()
            record['variants'].append(entry)
            print('MATCHED', name, json.dumps(entry), flush=True)
    return record


def mlp_case(weights):
    x = torch.randn((2048, 2048), dtype=torch.bfloat16, device='cuda')
    gu, dw = weights['gate_up'], weights['down']
    gate_packed = swiglu.pack_gate_up_weight(gu)
    down_packed = bundle.pack_linear_weight(dw, panel_n=64, block_k=64)
    xp = torch.empty(swiglu.INPUT_PACKED_SHAPE, dtype=x.dtype, device=x.device)
    zp = torch.empty(bundle.ACTIVATION_PACKED_SHAPE, dtype=x.dtype, device=x.device)
    gu_out = torch.empty((2048, 12288), dtype=x.dtype, device=x.device)
    ref_out = torch.empty_like(x)
    cand_out = torch.empty_like(x)

    def baseline():
        flag_gems.mm_out(x, gu.T, out=gu_out)
        z = flag_gems.silu_and_mul(gu_out[:, :6144], gu_out[:, 6144:])
        return flag_gems.mm_out(z, dw.T, out=ref_out)

    def candidate():
        return bundle.packed_mlp(x, gate_packed, down_packed,
                                 input_workspace=xp, activation_workspace=zp, out=cand_out)

    native_graph, native = captured(baseline)
    native_copy = native.clone()
    graph, result = captured(candidate)
    record = {'name': 'mlp_direct_metax_port', 'shape': [2048, 2048, 6144],
              'vs_native': agreement(result, native_copy),
              'timing': interleaved({'native': native_graph, 'packed_fused': graph}),
              'extra_weight_bytes': (gate_packed.numel() + down_packed.numel()) * 2,
              'workspace_bytes': (xp.numel() + zp.numel()) * 2}
    # Independent FP32 GEMMs, preserving both BF16 boundaries of the MLP.
    u = (x.float() @ gu.float().T).bfloat16().float()
    g, v = u.chunk(2, dim=1)
    z = (g * torch.sigmoid(g) * v).bfloat16().float()
    fp32 = z @ dw.float().T
    record['vs_fp32_reference_with_bf16_boundaries'] = agreement(result, fp32)
    record['native_vs_same_reference'] = agreement(native_copy, fp32)
    record['speedup'] = record['timing']['native']['median_ms'] / record['timing']['packed_fused']['median_ms']
    print('MLP', json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['direct', 'sweep'], default='direct')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # This probe owns no server. Bound its GPU work even if SSH disconnects.
    signal.alarm(5400)
    torch.set_num_threads(8)
    torch.manual_seed(20261011)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = {
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'phase': args.phase, 'official_benchmark': False,
        'weights': 'MiniCPM5-2B layer 0 and full lm_head',
        'torch': torch.__version__, 'flag_gems_import': flag_gems.__file__,
        'vendor': flag_gems.runtime.device.vendor_name,
        'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.glob('*.py')},
        'gpu_before': subprocess.check_output(['ixsmi'], text=True),
        'cases': [],
    }
    path = args.output / 'result.json'
    write_json(path, manifest)
    if manifest['vendor'] != 'iluvatar':
        raise RuntimeError('This probe is scoped to Iluvatar')
    with torch.inference_mode():
        weights = load_weights()
        if args.phase == 'direct':
            cases = [('qkv', 2048), ('o_proj', 2048), ('lm_head_64', 64), ('lm_head_1', 1)]
        else:
            cases = [('qkv', 2048), ('gate_up', 2048), ('down', 2048),
                     ('qkv_decode', 64), ('gate_up_decode', 64), ('down_decode', 64),
                     ('lm_head_64', 64)]
        for name, m in cases:
            try:
                key = 'lm_head' if name.startswith('lm_head') else name.removesuffix('_decode')
                case = matrix_case(name, m, weights[key], sweep=args.phase == 'sweep')
            except Exception:
                case = {'name': name, 'error': traceback.format_exc()}
                print('CASE_ERROR', name, case['error'], flush=True)
            manifest['cases'].append(case)
            write_json(path, manifest)
            gc.collect()
            torch.cuda.empty_cache()
        if args.phase == 'direct':
            try:
                manifest['cases'].append(mlp_case(weights))
            except Exception:
                manifest['cases'].append({'name': 'mlp_direct_metax_port', 'error': traceback.format_exc()})
                print('MLP_ERROR', manifest['cases'][-1]['error'], flush=True)
        manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write_json(path, manifest)
        print('COMPLETE', path, flush=True)


if __name__ == '__main__':
    main()
