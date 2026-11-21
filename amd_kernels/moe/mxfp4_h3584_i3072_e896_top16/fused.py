# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""GEMM1 and GEMM2 in one launch. Workgroups [0, n1) are GEMM1 tiles, then GEMM2 tiles; the dispatcher issues workgroups in
id order, so a GEMM2 tile only waits for GEMM1 tiles issued before it (no deadlock). GEMM2 tiles stream their weights
while GEMM1 finishes, then wait on their m-block's ready counter. Counters only order work: every element has one writer."""
import functools
import flydsl.compiler as flyc
from amd_kernels.warm_up import launcher
import flydsl.expr as fx

from .common import H, I, TOPK
from .gemm1 import compile_gemm1_tile
from .gemm2 import NB1, compile_gemm2_tile


@functools.cache
@launcher
def compile_fused(*, kw=4, pf1=3, tn2=256, bm=16, name_suffix=""):
    t1, nb1 = compile_gemm1_tile(kw=kw, pf=pf1, bm=bm, signal_ready=True)
    assert nb1 == NB1 or kw != 4
    t2, nb2 = compile_gemm2_tile(tn=tn2, bm=bm, wait_for=nb1)
    mr = bm // 16
    name = f"moe_experts_a4w4_situv2_h{H}_i{I}_top{TOPK}_bm{bm}_kw{kw}_pf{pf1}_tn{tn2}{name_suffix}"

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, 4 * mr * 4 * 64 * 16, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(A: fx.Tensor, AS: fx.Tensor, W13: fx.Tensor, S13: fx.Tensor, W2: fx.Tensor, S2: fx.Tensor,
               Q: fx.Tensor, QS: fx.Tensor, SL: fx.Tensor, IDS: fx.Tensor, WT: fx.Tensor, EIDS: fx.Tensor,
               NV: fx.Tensor, m: fx.Int32, rows: fx.Int32, ready: fx.Int64):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        bid = fx.Int32(fx.block_idx.x)
        n1 = rows // bm * nb1
        if bid < n1:
            t1(base, bid, A, AS, W13, S13, Q, QS, IDS, EIDS, NV, m, rows, ready)
        if bid >= n1:
            t2(bid - n1, Q, QS, W2, S2, SL, IDS, WT, EIDS, NV, m, rows, ready)

    @flyc.jit
    def launch(A: fx.Tensor, AS: fx.Tensor, W13: fx.Tensor, S13: fx.Tensor, W2: fx.Tensor, S2: fx.Tensor,
               Q: fx.Tensor, QS: fx.Tensor, SL: fx.Tensor, IDS: fx.Tensor, WT: fx.Tensor, EIDS: fx.Tensor,
               NV: fx.Tensor, m: fx.Int32, rows: fx.Int32, ready: fx.Int64, stream: fx.Stream):
        kernel(A, AS, W13, S13, W2, S2, Q, QS, SL, IDS, WT, EIDS, NV, m, rows, ready).launch(
            grid=(rows // bm * (nb1 + nb2), 1, 1), block=(256, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
