# SPDX-License-Identifier: Apache-2.0
"""Subprocess-only experiment hook. Inert unless explicitly enabled."""
import os

if os.environ.get('TIANSHU_PACK_PROBE_MODE'):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    try:
        from tianshu_packing.service_hook import install
        install()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise SystemExit(86)
