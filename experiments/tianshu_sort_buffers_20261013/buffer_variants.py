# SPDX-License-Identifier: Apache-2.0
"""Narrow private radix buffers while preserving public int64 indices."""
import hashlib
import inspect
from pathlib import Path


def create_variants(module, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    original = inspect.getsource(module.radix_sort)
    expected_index = 'torch.arange(0, n, dtype=torch.int64, device=arr_in.device)'
    expected_sum = 'torch.cumsum(global_hist, -1)'
    expected_cast = 'ex_cumsum_bins = ex_cumsum_bins.to(torch.uint32)'
    expected_return = 'return arr_in, indices_in'
    assert all(original.count(text) == 1 for text in (
        expected_index, expected_sum, expected_cast, expected_return))
    variants = {'original': module.radix_sort}
    provenance = {}
    for label, index32, scan32 in [('index32', True, False), ('scan32', False, True),
                                   ('both32', True, True)]:
        source = original
        # Original radix_sort bounds n < 2**30. Each index and histogram-prefix
        # value is therefore in 0..n and fits a signed int32. Addresses remain
        # Triton pointers; the public indices are widened exactly once.
        if index32:
            source = source.replace(expected_index, expected_index.replace('int64', 'int32'))
            source = source.replace('arr_out = torch.empty_like(arr)',
                                    'assert indices_in.dtype == torch.int32\n        arr_out = torch.empty_like(arr)')
            source = source.replace(expected_return, expected_return + '.to(torch.int64)')
        if scan32:
            source = source.replace(expected_sum, 'torch.cumsum(global_hist, -1, dtype=torch.int32)')
            source = source.replace(expected_cast,
                'assert ex_cumsum_bins.dtype == torch.int32\n        ex_cumsum_bins = ex_cumsum_bins.view(torch.uint32)')
        path = root / f'{label}.py'
        path.write_text(source)
        namespace = dict(vars(module))
        exec(compile(source, str(path), 'exec'), namespace)
        variants[label] = namespace['radix_sort']
        provenance[label] = dict(source=str(path), sha256=hashlib.sha256(source.encode()).hexdigest(),
                                 private_index_bits=32 if index32 else 64,
                                 prefix_accumulator_bits=32 if scan32 else 64,
                                 public_indices_bits=64, local_histogram_accumulator_unmodified=True)
    return variants, provenance
