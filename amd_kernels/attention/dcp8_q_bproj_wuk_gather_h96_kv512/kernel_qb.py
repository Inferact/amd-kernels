# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""q_b projection [M,1536] x [2304,1536]^T to bf16 q [M,12,192] (fp32 K sum, one bf16 rounding)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from .ops import mfma, pk_bf16

N, K = 2304, 1536
KSTEPS = K // 32
PAD = 4
PF = 12
NT = 0x2  # non-temporal loads: for weights that one workgroup reads once (+7% streaming, measured)
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


@launcher
def compile_qb(mt, nt=4, nw=4, wpe=1):
    """mt: 16-token tiles and nt: 16-column tiles per workgroup; nw waves split K inside it."""
    mtok, cols, wsteps = 16 * mt, 16 * nt, KSTEPS // nw
    ncb = N // cols
    nthr = 64 * nw
    items = (mtok * (cols // 8) + nthr - 1) // nthr
    name = f"bf16_q_b_proj_n2304_k1536_mt{mt}_nt{nt}_nw{nw}"
    row4 = (cols + PAD) // 4

    @fx.struct
    class Shared:
        p: fx.Array[fx.Int32, nw * mtok * (cols + PAD), 16]

    @flyc.kernel(name=name, known_block_size=[nthr, 1, 1])
    def kernel(x: fx.Int64, sx: fx.Int32, weight: fx.Int64, out: fx.Int64, rows: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        cb = bid % ncb
        tb = bid // ncb
        sm = fx.SharedAllocator().allocate(Shared).peek()
        v16 = fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, sm.p.ptr), fx.make_layout(nw * mtok * (cols + PAD), 1)),
            fx.make_layout(4, 1),
        )
        atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        wr = buffer_ops.create_buffer_resource_from_addr(weight, num_records_bytes=N * K * 2)
        xr = buffer_ops.create_buffer_resource_from_addr(x, num_records_bytes=fx.Int64(rows) * fx.Int64(sx * 2))

        def operands(s):
            step = wave * wsteps + s
            av = [
                Vec(
                    buffer_ops.buffer_load(
                        wr, ((cb * nt + j) * KSTEPS + step) * 256 + lane * 4, vec_width=4, dtype=fx.Int32
                    )
                )
                for j in range_constexpr(nt)
            ]
            bv = []
            for i in range_constexpr(mt):
                tok = tb * mtok + i * 16 + lane % 16
                bv.append(
                    Vec(
                        buffer_ops.buffer_load(
                            xr,
                            tok * (sx // 2) + step * 16 + (lane // 16) * 4,
                            vec_width=4,
                            dtype=fx.Int32,
                            mask=tok < rows,
                        )
                    )
                )
            return av + bv

        acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(mt * nt)]
        ops = [operands(s) for s in range_constexpr(min(PF, wsteps))]
        fx.rocdl.sched_barrier(0)
        for s in range_constexpr(wsteps):
            if s + PF < wsteps:
                ops.append(operands(s + PF))
            cur = ops[s]
            for i in range_constexpr(mt):
                for j in range_constexpr(nt):
                    acc[i * nt + j] = mfma(cur[j], cur[nt + i], acc[i * nt + j])
        for i in range_constexpr(mt):
            for j in range_constexpr(nt):
                tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                tmp.store(acc[i * nt + j].bitcast(fx.Int32))
                fx.copy(
                    atom, tmp, fx.slice(v16, (None, (wave * mtok + i * 16 + lane % 16) * row4 + j * 4 + lane // 16))
                )
        fx.gpu.barrier()
        dst = buffer_ops.create_buffer_resource_from_addr(out, num_records_bytes=fx.Int64(rows) * (N * 2))
        for r in range_constexpr(items):
            item = tx + r * nthr
            tk = item // (cols // 8)
            ch = item % (cols // 8)
            tok = tb * mtok + tk
            live = (tk < mtok) & (tok < rows)
            sums = [None] * 8
            for w in range_constexpr(nw):
                for h in range_constexpr(2):
                    tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(atom, fx.slice(v16, (None, (w * mtok + tk) * row4 + ch * 2 + h)), tmp)
                    v = Vec(tmp.load()).bitcast(fx.Float32)
                    for e in range_constexpr(4):
                        a = fx.Float32(v[e])
                        sums[h * 4 + e] = a if w == 0 else sums[h * 4 + e] + a
            packed = Vec.from_elements([pk_bf16(sums[2 * q], sums[2 * q + 1]) for q in range_constexpr(4)], fx.Int32)
            buffer_ops.buffer_store(packed, dst, tok * (N // 2) + cb * (cols // 2) + ch * 4, mask=live)

    @flyc.jit
    def launch_q_b(x: fx.Int64, sx: fx.Int32, weight: fx.Int64, out: fx.Int64, rows: fx.Int32, stream: fx.Stream):
        kernel(
            x,
            sx,
            weight,
            out,
            rows,
            value_attrs={"rocdl.waves_per_eu": wpe, "rocdl.flat_work_group_size": f"{nthr},{nthr}"},
        ).launch(grid=(fx.Int64(((rows + mtok - 1) // mtok) * ncb), 1, 1), block=(nthr, 1, 1), stream=stream)

    launch_q_b.compile_hints = PRELOAD_HINTS
    launch_q_b.kernel_name = name
    return launch_q_b


def pack_q_b(w):
    """[2304, 1536] row-major -> the MFMA A-operand tiles [144, 48, 4 (k group), 16 (row), 8 (k)]."""
    return w.view(N // 16, 16, KSTEPS, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(N, K)
