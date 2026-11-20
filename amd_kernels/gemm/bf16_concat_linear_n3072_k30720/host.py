# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import functools
import torch
from .kernel import compile_reduce
from .kernel_staged_a_eightwave import compile_concat_linear_staged_a_eightwave
from .kernel_lds import compile_concat_linear_lds

N, K, H = 3072, 30720, 6144
MAX_ROWS = 2048  # the tiers' row bound (a caller with more rows splits them)
# (max rows, (kernel, row tiles, column tiles per wave, K slabs)); cold-weight graph sweep:
# benchmarks/gemm/bf16_concat_linear_n3072_k30720/benchmark_concat_linear.py --sweep
TIERS = (
    (32, ("lds3", 1, 8, 10)),
    (64, ("lds3", 2, 8, 10)),
    (96, ("lds3", 3, 8, 10)),
    (128, ("lds3", 4, 8, 10)),
    (192, ("lds3", 6, 8, 10)),
    (256, ("lds3", 8, 8, 10)),
    (320, ("lds3", 5, 16, 10)),
    (448, ("lds", 7, 16, 10)),
    (512, ("lds", 8, 16, 10)),
    (768, ("lds", 8, 12, 5)),
    (1024, ("lds", 8, 16, 5)),
    (MAX_ROWS, ("eightwave", 8, 3, 2)),
)
_COMPILERS = {
    "eightwave": compile_concat_linear_staged_a_eightwave,
    "lds": lambda mt, nr, split, _: compile_concat_linear_lds(mt, nr, split, 2),
    "lds1": lambda mt, nr, split, _: compile_concat_linear_lds(mt, nr, split, 1),
    "lds3": lambda mt, nr, split, _: compile_concat_linear_lds(mt, nr, split, 2, 3),
}


@functools.cache
def get_kernel(kind, mt, nt, split):
    return _COMPILERS[kind](mt, nt, split, 2)


@functools.cache
def get_reduce(split):return compile_reduce(split)


def run(exe,*args):
    import flydsl.compiler as flyc
    if not hasattr(exe,'_cf'):exe._cf=flyc.compile(exe,*args)
    else:exe._cf(*args)


def _partials_config(taps,weight,kind,mt,nt,split):
    rows=taps[0].shape[0]
    partials=torch.empty((split,rows,N),device=weight.device,dtype=torch.float32)
    run(get_kernel(kind,mt,nt,split),*(x.data_ptr() for x in taps),weight.data_ptr(),partials.data_ptr(),rows,*(x.stride(0) for x in taps),torch.cuda.current_stream())
    return partials


def _linear_config(taps,weight,kind,mt,nt,split,out=None):
    rows=taps[0].shape[0]
    partials=_partials_config(taps,weight,kind,mt,nt,split)
    out=out if out is not None else torch.empty((rows,N),device=weight.device,dtype=torch.bfloat16)
    run(get_reduce(split),partials,out,rows,torch.cuda.current_stream())
    return out


def pack_weight(weight):
    """Lossless load-time layout; replace and release the row-major weight."""
    return weight.view(N//16,16,K//32,4,8).permute(0,2,3,1,4).contiguous().view(N,K)


def config_for(rows):
    """The row tier (kernel, row tiles, column tiles per wave, K slabs); the slab count sets the fp32 sum order."""
    assert 0 < rows <= MAX_ROWS, rows
    return next(cfg for top, cfg in TIERS if rows <= top)


def bf16_concat_linear(taps,weight,out=None,config=None):
    """Project five bf16 taps in place through the packed N-shard; ``config`` overrides the tier (sweeps)."""
    assert len(taps)==5 and weight.shape==(N,K) and weight.dtype==torch.bfloat16 and weight.is_contiguous()
    m=taps[0].shape[0]
    assert 0<m<=MAX_ROWS
    for x in taps:
        assert x.shape==(m,H) and x.dtype==torch.bfloat16 and x.stride(1)==1 and x.stride(0)%8==0 and x.device==weight.device
    if out is not None:assert out.shape==(m,N) and out.dtype==torch.bfloat16 and out.is_contiguous() and out.device==weight.device
    return _linear_config(taps,weight,*(config or config_for(m)),out)


def warmup_concat_linear(taps,weight,max_rows=MAX_ROWS):
    """Compile every row tier up to ``max_rows`` (the taps hold at least that many rows)."""
    for top, _ in TIERS:
        m=min(top,max_rows)
        bf16_concat_linear([x[:m] for x in taps],weight)
        if top>=max_rows:break
