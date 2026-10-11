# SPDX-License-Identifier: Apache-2.0
"""Ephemeral, opt-in service experiment for M2048 down projection only.

Patch the actual FlagGems launch global in this child process; never overwrite
installed files. Original weights remain intact. Include packed weights in
vLLM's model memory accounting before profiling. No logits/attention changes.
"""

import atexit
import hashlib
import json
import os
from pathlib import Path
import sys

import flag_gems
import torch
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm_fl.worker.worker import WorkerFL

from .matched_gemm import launch
from .packed_bundle import pack_linear_weight

MODE = os.environ['TIANSHU_PACK_PROBE_MODE']
CONFIG = (128, 128, 32, 8, 2)
WEIGHTS = {}
DATA = {'mode': MODE, 'pid': os.getpid(), 'python_callbacks': 0,
        'packed_bytes': 0, 'registered_layers': [], 'live_validations': [],
        'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.glob('*.py')}}


def save():
    root = Path(os.environ['TIANSHU_PACK_TRACE_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'hook_worker_{os.getpid()}.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(DATA, indent=2) + '\n')
    temp.replace(path)


def live():
    return (not torch.cuda.is_current_stream_capturing()
            and is_forward_context_available()
            and get_forward_context().attn_metadata is not None)


def prepare(worker):
    config = worker.vllm_config
    text, parallel = config.model_config.hf_config, config.parallel_config
    if (flag_gems.runtime.device.vendor_name != 'iluvatar'
            or config.model_config.dtype != torch.bfloat16
            or config.model_config.quantization is not None
            or config.lora_config is not None or config.speculative_config is not None
            or parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or getattr(text, 'hidden_size', None) != 2048
            or getattr(text, 'intermediate_size', None) != 6144
            or getattr(text, 'num_hidden_layers', None) != 42):
        raise RuntimeError('Unsupported packing experiment model/platform')
    if MODE == 'original':
        save()
        return
    before = torch.cuda.memory_allocated()
    with torch.inference_mode():
        for name, layer in worker.model_runner.model.named_modules():
            if not name.endswith('.mlp.down_proj'):
                continue
            weight = layer.weight
            if (tuple(weight.shape) != (2048, 6144) or weight.dtype != torch.bfloat16
                    or not weight.is_contiguous() or getattr(layer, 'bias', None) is not None):
                raise RuntimeError('Unexpected down projection: ' + name)
            tensor = (pack_linear_weight(weight, panel_n=128, block_k=32)
                      if MODE == 'packed' else weight)
            WEIGHTS[(weight.device, weight.data_ptr())] = (name, tensor, weight)
            DATA['registered_layers'].append(name)
            if MODE == 'packed':
                DATA['packed_bytes'] += tensor.numel() * tensor.element_size()
    if len(WEIGHTS) != 42:
        raise RuntimeError(f'Expected 42 down projections, found {len(WEIGHTS)}')
    worker.model_runner.model_memory_usage += torch.cuda.memory_allocated() - before
    DATA['memory_accounting_added_bytes'] = torch.cuda.memory_allocated() - before
    globals_ = flag_gems.mm.__globals__
    original = globals_['_launch_mm']
    DATA['actual_mm_module'] = flag_gems.mm.__module__
    DATA['actual_mm_source'] = flag_gems.mm.__code__.co_filename
    DATA['actual_mm_source_sha256'] = hashlib.sha256(Path(DATA['actual_mm_source']).read_bytes()).hexdigest()

    def patched(a, b, c, m, n, k):
        if ((m, n, k) != (2048, 2048, 6144) or a.dtype != torch.bfloat16
                or b.dtype != torch.bfloat16 or c.dtype != torch.bfloat16
                or not a.is_contiguous() or not c.is_contiguous()
                or b.stride() != (1, 6144)):
            return original(a, b, c, m, n, k)
        state = WEIGHTS.get((b.device, b.data_ptr()))
        if state is None:
            return original(a, b, c, m, n, k)
        name, tensor, source = state
        result = launch(a, tensor, c, CONFIG, MODE == 'packed')
        DATA['python_callbacks'] += 1
        # Validate actual live activations at three depths, outside graph capture.
        validate_names = [DATA['registered_layers'][i] for i in (0, 21, 41)]
        done = {v['name'] for v in DATA['live_validations']}
        if name in validate_names and name not in done and live():
            reference = torch.empty_like(c)
            original(a, b, reference, m, n, k)
            difference = (result.float() - reference.float()).abs()
            check = {'name': name, 'shape': [m, n, k],
                     'bitwise_equal': bool(torch.equal(result, reference)),
                     'max_abs': difference.max().item()}
            DATA['live_validations'].append(check)
            save()
            if not check['bitwise_equal']:
                failure = Path(os.environ['TIANSHU_PACK_TRACE_DIR']) / f'failure_{os.getpid()}.pt'
                torch.save({'name': name, 'input': a.cpu(), 'weight': source.cpu(),
                            'candidate': result.cpu(), 'reference': reference.cpu()}, failure)
                raise RuntimeError('Live down projection failed bitwise comparison: ' + name)
        elif DATA['python_callbacks'] == 1:
            save()
        return result

    globals_['_launch_mm'] = patched
    save()
    print(f'[TIANSHU-PACK-PROBE] mode={MODE} layers=42 packed_bytes={DATA["packed_bytes"]}',
          file=sys.stderr, flush=True)


def install():
    if MODE not in ('original', 'row', 'packed'):
        raise RuntimeError('Unknown experiment mode: ' + MODE)
    original = WorkerFL.load_model

    def load_model(self, *, load_dummy_weights=False):
        original(self, load_dummy_weights=load_dummy_weights)
        prepare(self)

    WorkerFL.load_model = load_model
    atexit.register(save)
