"""Fast numerical references; the original torch_ref.py remains the oracle."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]


def _fp32_gemm(a,b):
    return torch.matmul(a,b)


class CompiledGemmReference:
    backend_name='torch_compile'
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def __init__(self):
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(ROOT/'build/reference_torch_cache'))
        # This suite intentionally contains many static shapes.
        if hasattr(torch._dynamo.config,'recompile_limit'):
            torch._dynamo.config.recompile_limit=128
        else:
            torch._dynamo.config.cache_size_limit=128
        torch._inductor.config.compile_threads=2
        self.compiled=torch.compile(_fp32_gemm,fullgraph=True,dynamic=False,
                                    mode='max-autotune-no-cudagraphs')

    def __call__(self,a,b):
        return self.compiled(a,b)


def select_reference(part,mode,plain_reference):
    plain_reference.backend_name='torch'
    path=ROOT/('part1' if part=='gemm' else 'part2')/'torch_ref.py'
    plain_reference.source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    if mode=='torch':return plain_reference
    if part=='gemm':return CompiledGemmReference()
    try:
        path=ROOT/'instructor/fast_attention_reference.py'
        if not path.is_file():raise FileNotFoundError('Instructor CuTe DSL reference is not in this package')
        spec=importlib.util.spec_from_file_location('mlsys_fast_attention',path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module;spec.loader.exec_module(module)
        return module.CuteAttentionReference()
    except (ImportError,FileNotFoundError) as error:
        if mode=='fast':raise RuntimeError(f'CuTe DSL reference unavailable: {error}') from error
        print(f'REFERENCE attention: using explicit Torch fallback ({error})',flush=True)
        return plain_reference
