# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""W_UK absorption of this rank's 12 heads: bf16 q [M,12,192] to the latent query [M,12,576].

``local``: the result is a local tensor. ``push``: the result goes to rows [M, 96, 576] of the eight ranks' gather
buffers, and the kernel returns once all eight ranks' heads arrived (flags.py)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from .flags import DATA_OFF, FREE_OFF, LINE, READY_OFF, WORDS, flag_store, sel8, spin, ticket_add
from .ops import mfma, pk_bf16

L, P, QW, OW = 512, 128, 192, 576
GROUPS = L // 64
KS = P // 32
SYS = 0x11
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def pack_w_uk(w_uk_t):
    """W_UK_T bf16 [H, 128, 512] -> A tiles [H, 8 (64-column group), 4 (tile), 4 (k step), 4 (k group), 16 (row), 8].

    Row g*4+e of tile jj holds latent column 64*group + 32*(jj // 2) + 8*g + 4*(jj % 2) + e, so a lane's accumulators
    of tiles 2z and 2z+1 are 8 consecutive columns and the four lanes of a token cover 32."""
    h = w_uk_t.shape[0]
    w = w_uk_t.transpose(1, 2).contiguous()
    w = w.view(h, GROUPS, 2, 4, 2, 4, P)
    w = w.permute(0, 1, 2, 4, 3, 5, 6)
    w = w.reshape(h, GROUPS, 4, 16, KS, 4, 8)
    return w.permute(0, 1, 2, 4, 5, 3, 6).contiguous()


def grid_wuk(rows, heads, mt, ns):
    return ((rows + 16 * mt - 1) // (16 * mt)) * heads * ns


@launcher
def compile_wuk(heads, mt, ns, mode="local", wpe=1):
    """heads: H of the call; mt: token tiles per workgroup; ns: workgroups per head along the latent columns."""
    assert mode in ("local", "push")
    mtok = 16 * mt
    gpw = max(1, GROUPS // ns // 4)
    nwaves = GROUPS // ns // gpw
    nthr = 64 * nwaves
    assert gpw * nwaves * ns == GROUPS
    items = (mtok * 8 + nthr - 1) // nthr
    row4 = 73
    lat4 = 64 // ns
    name = f"bf16_w_uk_absorb_h{heads}_kv512_mt{mt}_ns{ns}_{mode}"

    @fx.struct
    class Shared:
        t: fx.Array[fx.Int32, mtok * row4 * 4, 16]

    @flyc.kernel(name=name, known_block_size=[nthr, 1, 1])
    def kernel(
        q: fx.Int64,
        sq: fx.Int32,
        weight: fx.Int64,
        out: fx.Int64,
        rows: fx.Int32,
        b0: fx.Int64,
        b1: fx.Int64,
        b2: fx.Int64,
        b3: fx.Int64,
        b4: fx.Int64,
        b5: fx.Int64,
        b6: fx.Int64,
        b7: fx.Int64,
        me: fx.Int32,
        state: fx.Int64,
        g2: fx.Int32,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        sl = bid % ns
        h = (bid // ns) % heads
        tb = bid // (ns * heads)
        bases = (b0, b1, b2, b3, b4, b5, b6, b7)
        mine = sel8(me, bases)
        sm = fx.SharedAllocator().allocate(Shared).peek()
        v16 = fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, sm.t.ptr), fx.make_layout(mtok * row4 * 4, 1)), fx.make_layout(4, 1)
        )
        atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        sr = buffer_ops.create_buffer_resource_from_addr(state, num_records_bytes=16)
        epoch = fx.Int32(buffer_ops.buffer_load(sr, 0, vec_width=1, dtype=fx.Int32, is_scalar=True))
        if mode == "push":
            if (bid == 0) & (tx < 8):
                flag_store(sel8(tx, bases) + FREE_OFF + fx.Int64(me * (LINE * 4)), epoch + 1)
        wr = buffer_ops.create_buffer_resource_from_addr(weight, num_records_bytes=heads * L * P * 2)
        qr = buffer_ops.create_buffer_resource_from_addr(q, num_records_bytes=fx.Int64(rows) * fx.Int64(sq * 2))
        dst = buffer_ops.create_buffer_resource_from_addr(out, num_records_bytes=fx.Int64(rows) * (heads * OW * 2))
        aw = []
        for gi in range_constexpr(gpw):
            grp = sl * (GROUPS // ns) + wave * gpw + gi
            for jj in range_constexpr(4):
                for s in range_constexpr(KS):
                    aw.append(
                        Vec(
                            buffer_ops.buffer_load(
                                wr,
                                (((h * GROUPS + grp) * 4 + jj) * KS + s) * 256 + lane * 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                    )
        bq = []
        for i in range_constexpr(mt):
            tok = tb * mtok + i * 16 + lane % 16
            for s in range_constexpr(KS):
                bq.append(
                    Vec(
                        buffer_ops.buffer_load(
                            qr,
                            tok * (sq // 2) + h * (QW // 2) + s * 16 + (lane // 16) * 4,
                            vec_width=4,
                            dtype=fx.Int32,
                            mask=tok < rows,
                        )
                    )
                )
        if mode == "push":
            if tx < 8:
                spin(mine + FREE_OFF + fx.Int64(tx * (LINE * 4)), epoch + 1)
            fx.gpu.barrier()

        def emit(vec, tok, off, live):
            if mode == "push":
                for p in range_constexpr(8):
                    d = buffer_ops.create_buffer_resource_from_addr(
                        bases[p] + DATA_OFF, num_records_bytes=fx.Int64(rows) * (96 * OW * 2)
                    )
                    buffer_ops.buffer_store(
                        vec, d, tok * (96 * OW // 2) + me * (12 * OW // 2) + h * (OW // 2) + off, mask=live
                    )
            else:
                buffer_ops.buffer_store(vec, dst, tok * (heads * OW // 2) + h * (OW // 2) + off, mask=live)

        for gi in range_constexpr(gpw):
            grp = sl * (GROUPS // ns) + wave * gpw + gi
            for i in range_constexpr(mt):
                packed = []
                for jj in range_constexpr(4):
                    c = Vec.filled(4, 0.0, fx.Float32)
                    for s in range_constexpr(KS):
                        c = mfma(aw[(gi * 4 + jj) * KS + s], bq[i * KS + s], c)
                    packed += [pk_bf16(c[2 * z], c[2 * z + 1]) for z in range_constexpr(2)]
                for z in range_constexpr(2):
                    tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    tmp.store(Vec.from_elements(packed[4 * z : 4 * z + 4], fx.Int32))
                    fx.copy(
                        atom, tmp, fx.slice(v16, (None, (i * 16 + lane % 16) * row4 + grp * 8 + z * 4 + lane // 16))
                    )
        fx.gpu.barrier()
        for r in range_constexpr((mtok * lat4 + nthr - 1) // nthr):
            item = tx + r * nthr
            tk = item // lat4
            c = sl * lat4 + item % lat4
            tok = tb * mtok + tk
            live = (tk < mtok) & (tok < rows)
            tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(atom, fx.slice(v16, (None, tk * row4 + c)), tmp)
            emit(Vec(tmp.load()), tok, 4 * c, live)
        if sl == 0:
            for r in range_constexpr(items):
                item = tx + r * nthr
                tk = item // 8
                ch = item % 8
                tok = tb * mtok + tk
                live = (tk < mtok) & (tok < rows)
                v = Vec(
                    buffer_ops.buffer_load(
                        qr, tok * (sq // 2) + h * (QW // 2) + 64 + ch * 4, vec_width=4, dtype=fx.Int32, mask=live
                    )
                )
                emit(v, tok, 256 + ch * 4, live)
        if mode == "push":
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            if tx < 8:
                flag_store(sel8(tx, bases) + READY_OFF + fx.Int64((me * WORDS + bid) * 4), epoch + 1)
            if bid < 8:
                for jw in range_constexpr((WORDS + nthr - 1) // nthr):
                    w = tx + jw * nthr
                    if w < g2:
                        spin(mine + READY_OFF + fx.Int64((bid * WORDS + w) * 4), epoch + 1)
            fx.gpu.barrier()
            if tx == 0:
                old = ticket_add(state + 4)
                if old + 1 == g2:
                    flag_store(state + 4, 0)
                    flag_store(state, epoch + 1)

    @flyc.jit
    def launch_w_uk(
        q: fx.Int64,
        sq: fx.Int32,
        weight: fx.Int64,
        out: fx.Int64,
        rows: fx.Int32,
        b0: fx.Int64,
        b1: fx.Int64,
        b2: fx.Int64,
        b3: fx.Int64,
        b4: fx.Int64,
        b5: fx.Int64,
        b6: fx.Int64,
        b7: fx.Int64,
        me: fx.Int32,
        state: fx.Int64,
        g2: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q,
            sq,
            weight,
            out,
            rows,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            me,
            state,
            g2,
            value_attrs={"rocdl.waves_per_eu": wpe, "rocdl.flat_work_group_size": f"{nthr},{nthr}"},
        ).launch(grid=(fx.Int64(((rows + mtok - 1) // mtok) * heads * ns), 1, 1), block=(nthr, 1, 1), stream=stream)

    launch_w_uk.compile_hints = PRELOAD_HINTS
    launch_w_uk.kernel_name = name
    return launch_w_uk
