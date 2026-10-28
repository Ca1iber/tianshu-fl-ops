# SPDX-License-Identifier: Apache-2.0
"""Exact stable sort and graph timings for private integer buffers."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import signal
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    assert not root.exists()
    root.mkdir(parents=True)
    stop = datetime.fromisoformat(json.loads(Path(
        '/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json'
    ).read_text())['stop_at_utc'])
    remaining = int((stop - datetime.now(timezone.utc)).total_seconds())
    assert remaining >= 1200
    signal.alarm(min(3600, remaining))
    hardware = subprocess.check_output(['ixsmi'], text=True)
    assert 'No running processes found' in hardware
    (root / 'hardware_before.txt').write_text(hardware)
    import torch
    import flag_gems
    from buffer_variants import create_variants
    module = importlib.import_module('flag_gems.ops.sort')
    original = module.radix_sort
    variants, provenance = create_variants(module, root / 'generated')
    snapshot = root / 'source_snapshot'
    snapshot.mkdir()
    for path in (Path(module.__file__), Path(__file__), Path(__file__).with_name('buffer_variants.py')):
        (snapshot / path.name).write_bytes(path.read_bytes())
    data = dict(complete=False, numerical_checks_passed=False, official_benchmark=False,
                adopted=False, installed_sources_modified=False,
                original_sort_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                variants=provenance, cases=[], started_utc=datetime.now(timezone.utc).isoformat())
    torch.set_num_threads(8)

    def save():
        p = root / 'result.tmp'
        p.write_text(json.dumps(data, indent=2) + '\n')
        p.replace(root / 'result.json')

    save()
    try:
        with torch.inference_mode():
            shapes = [(1, 8191), (2, 8193), (1, 130560), (32, 130560), (64, 130560)]
            for case, (m, n) in enumerate(shapes):
                for pattern in ('random', 'ties', 'masked'):
                    torch.manual_seed(2026101330 + case)
                    x = (torch.randint(-8, 8, (m, n), device='cuda').float()
                         if pattern == 'ties' else torch.randn((m, n), device='cuda'))
                    if pattern == 'masked':
                        x[:, ::4] = -float('inf')
                    for descending in (False, True):
                        cpu_values, cpu_indices = torch.sort(x.cpu(), stable=True, descending=descending)
                        case_row = dict(m=m, n=n, pattern=pattern, descending=descending, variants=[])
                        graphs = []
                        for label, function in variants.items():
                            row = dict(label=label)
                            try:
                                module.radix_sort = function
                                with flag_gems.use_gems():
                                    values, indices = torch.sort(x, stable=True, descending=descending)
                                row.update(values_exact=torch.equal(values.cpu(), cpu_values),
                                           stable_indices_exact=torch.equal(indices.cpu(), cpu_indices),
                                           public_index_dtype=str(indices.dtype))
                                assert row['values_exact'] and row['stable_indices_exact']
                                assert indices.dtype == torch.int64
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    with flag_gems.use_gems():
                                        gv, gi = torch.sort(x, stable=True, descending=descending)
                                graph.replay()
                                torch.cuda.synchronize()
                                row['graph_values_exact'] = torch.equal(gv.cpu(), cpu_values)
                                row['graph_indices_exact'] = torch.equal(gi.cpu(), cpu_indices)
                                assert row['graph_values_exact'] and row['graph_indices_exact']
                                graphs.append((row, graph, gv, gi))
                            except Exception as error:
                                row['error'] = str(error)
                            finally:
                                module.radix_sort = original
                            case_row['variants'].append(row)
                        timings = {id(row): [] for row, *_ in graphs}
                        for repeat in range(7):
                            order = (graphs[repeat % len(graphs):] + graphs[:repeat % len(graphs)]) if graphs else []
                            for row, graph, *_ in order:
                                start = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                start.record()
                                for _ in range(10):
                                    graph.replay()
                                end.record()
                                end.synchronize()
                                timings[id(row)].append(start.elapsed_time(end) / 10)
                        for row, *_ in graphs:
                            row['raw_ms'] = timings[id(row)]
                            row['median_ms'] = statistics.median(row['raw_ms'])
                        data['cases'].append(case_row)
                        save()
                        print(json.dumps(case_row), flush=True)
                        del graphs
                    del x
                    torch.cuda.empty_cache()
        data['numerical_checks_passed'] = all(
            'error' not in row and row['values_exact'] and row['stable_indices_exact']
            and row['graph_values_exact'] and row['graph_indices_exact']
            for case in data['cases'] for row in case['variants'])
        assert len(data['cases']) == 30
        data.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
        save()
    finally:
        module.radix_sort = original


if __name__ == '__main__':
    main()
