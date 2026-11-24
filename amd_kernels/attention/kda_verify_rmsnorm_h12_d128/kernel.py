# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One block owns a checkpoint head. Each group of eight lanes keeps two value rows in registers across the window."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.common import buffer_ops

from ..kda_decode_fb_gate_spec_h12_d128.common import (
    bf16x2_hi,
    bf16x2_lo,
    pk_bf16,
    round_bf16,
    rsqrt,
    sigmoid,
    sum_group,
    wave_sum_lane63,
)

H, D = 12, 128
WAVES = 8


@functools.cache
def build(window, norm_f32):
    name = f"kda_verify_rmsnorm_h12_d128_t{window}_nw{32 if norm_f32 else 16}"

    @fx.struct
    class Shared:
        values: fx.Array[fx.Float32, window * 5 * D]
        betas: fx.Array[fx.Float32, window]
        squares: fx.Array[fx.Float32, WAVES * window]

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        qkv: fx.Int64,
        gate: fx.Int64,
        beta: fx.Int64,
        out_gate: fx.Int64,
        alog: fx.Int64,
        bias: fx.Int64,
        state: fx.Int64,
        offsets: fx.Int64,
        slots: fx.Int64,
        weight: fx.Int64,
        correction: fx.Int64,
        key_gate: fx.Int64,
        out: fx.Int64,
        qkv_stride: fx.Int32,
        gate_stride: fx.Int32,
        beta_stride: fx.Int32,
        og_stride: fx.Int32,
        state_pitch: fx.Int64,
        slot_stride: fx.Int32,
        out_stride: fx.Int32,
        bound: fx.Float32,
        eps: fx.Float32,
    ):
        sh = fx.SharedAllocator().allocate(Shared).peek()
        squares = sh.squares.ptr
        values = sh.values.ptr
        b = fx.block_idx.x
        head = fx.block_idx.y
        tx = fx.thread_idx.x
        lane = tx % 64
        group_lane = tx % 8
        wave = tx // 64
        v0 = 2 * (tx // 8)
        resource = buffer_ops.create_buffer_resource_from_addr
        load = buffer_ops.buffer_load
        store = buffer_ops.buffer_store
        cu = resource(offsets + fx.Int64(b) * 4, num_records_bytes=8)
        start = fx.Int32(load(cu, 0, vec_width=1, is_scalar=True))
        length = fx.Int32(load(cu, 1, vec_width=1, is_scalar=True)) - start
        slot_ptr = resource(slots + fx.Int64(b) * fx.Int64(slot_stride) * 4, num_records_bytes=4)
        slot = fx.Int32(load(slot_ptr, 0, vec_width=1, is_scalar=True))

        def row(ptr, stride, index, head_offset, elements):
            return resource(
                ptr + (fx.Int64(index) * fx.Int64(stride) + fx.Int64(head_offset)) * 2,
                num_records_bytes=elements * 2,
            )

        if (slot > 0) & (length > 0):
            sr = resource(state + (fx.Int64(slot) * state_pitch + fx.Int64(head) * D * D) * 4)
            record = (fx.Int64(b) * H + fx.Int64(head)) * window * D
            cr = resource(correction + record * 4, num_records_bytes=window * D * 4)
            kg = resource(key_gate + record * 4, num_records_bytes=window * D * 4)
            A = fx.math.exp(fx.Float32(load(resource(alog), head, vec_width=1)))
            db = resource(bias + fx.Int64(head) * D * 4)
            columns = [group_lane * 4 + 32 * j for j in range_constexpr(4)]
            S = [
                [Vec(load(sr, (v0 + i) * D + c, vec_width=4, dtype=fx.Float32)) for c in columns]
                for i in range_constexpr(2)
            ]

            def pair(word):
                return [bf16x2_lo(word), bf16x2_hi(word)]

            def broadcast_sum(x):
                last = wave_sum_lane63(x).bitcast(fx.Int32)
                return fx.Int32(
                    fx.rocdl.readlane(T.i32, fx.as_ir_value(last), fx.as_ir_value(fx.Int32(63)))
                ).bitcast(fx.Float32)

            if wave < window:
                valid = wave < length
                token = valid.select(start + wave, start)
                qr = row(qkv, qkv_stride, token, head * D, (2 * H + 1) * D)
                gr = row(gate, gate_stride, token, head * D, D)
                qval = pair(load(qr, lane, vec_width=1, dtype=fx.Int32))
                kp = load(qr, H * D // 2 + lane, vec_width=1, dtype=fx.Int32)
                gp = load(gr, lane, vec_width=1, dtype=fx.Int32)
                kval, gval = pair(kp), pair(gp)
                store(kp, kg, wave * D + lane, mask=valid)
                store(gp, kg, wave * D + D // 2 + lane, mask=valid)
                vval = pair(load(qr, H * D + lane, vec_width=1, dtype=fx.Int32))
                og = pair(load(row(out_gate, og_stride, token, head * D, D), lane, vec_width=1, dtype=fx.Int32))
                if const_expr(norm_f32):
                    nw = Vec(load(resource(weight), 2 * lane, vec_width=2, dtype=fx.Float32))
                else:
                    nw = pair(load(resource(weight), lane, vec_width=1, dtype=fx.Int32))
                dt = Vec(load(db, 2 * lane, vec_width=2, dtype=fx.Float32))
                iq = rsqrt(broadcast_sum(qval[0] * qval[0] + qval[1] * qval[1]) + 1e-6) * (D**-0.5)
                ik = rsqrt(broadcast_sum(kval[0] * kval[0] + kval[1] * kval[1]) + 1e-6)
                for e in range_constexpr(2):
                    pos = wave * 5 * D + lane * 2 + e
                    values[pos] = qval[e] * iq
                    values[pos + D] = kval[e] * ik
                    values[pos + 2 * D] = fx.math.exp(bound * sigmoid(A * (gval[e] + dt[e])))
                    values[pos + 3 * D] = vval[e]
                    values[pos + 4 * D] = nw[e] * sigmoid(og[e])
                if lane == 0:
                    br = row(beta, beta_stride, token, 0, H)
                    bits = fx.Int32(load(br, head, vec_width=1, dtype=fx.Int16))
                    sh.betas.ptr[wave] = sigmoid((bits << 16).bitcast(fx.Float32))
            fx.gpu.barrier()

            view = fx.logical_divide(fx.make_view(values, fx.make_layout(window * 5 * D, 1)), fx.make_layout(4, 1))
            atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)

            def load4(index):
                tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                fx.copy(atom, fx.slice(view, (None, index // 4)), tmp)
                return Vec(tmp.load())

            def dot(x, y):
                partial = x[0] * y[0]
                for j in range_constexpr(1, 4):
                    partial = fx.math.fma(x[j], y[j], partial)
                return sum_group((partial[0] + partial[1]) + (partial[2] + partial[3]), 8)

            results = []
            for t in range_constexpr(window):
                base = t * 5 * D
                q = [load4(base + c) for c in columns]
                k = [load4(base + D + c) for c in columns]
                decay = [load4(base + 2 * D + c) for c in columns]
                bt = sh.betas.ptr[t]
                outputs, corrections = [], []
                for i in range_constexpr(2):
                    decayed = [S[i][j] * decay[j] for j in range_constexpr(4)]
                    u = (values[base + 3 * D + v0 + i] - dot(decayed, k)) * bt
                    corrections.append(u)
                    for j in range_constexpr(4):
                        S[i][j] = decayed[j] + k[j] * u
                    outputs.append(round_bf16(dot(S[i], q)))
                store(
                    Vec.from_elements(corrections, fx.Float32),
                    cr,
                    t * D + v0,
                    mask=(group_lane == 0) & (t < length),
                )
                ss = outputs[0] * outputs[0] + outputs[1] * outputs[1]
                wave_ss = wave_sum_lane63((group_lane == 0).select(ss, fx.Float32(0)))
                if lane == 63:
                    squares[t * WAVES + wave] = wave_ss
                results.append(outputs)

            fx.gpu.barrier()
            for t in range_constexpr(window):
                total = squares[t * WAVES]
                for j in range_constexpr(1, WAVES):
                    total = total + squares[t * WAVES + j]
                inv = rsqrt(total * (1.0 / D) + eps)
                scale = t * 5 * D + 4 * D + v0
                result = pk_bf16(results[t][0] * inv * values[scale], results[t][1] * inv * values[scale + 1])
                if t < length:
                    store(result, row(out, out_stride, start + t, head * D, D), v0 // 2, mask=group_lane == 0)
        else:
            for t in range_constexpr(window):
                if t < length:
                    store(fx.Int32(0), row(out, out_stride, start + t, head * D, D), v0 // 2, mask=group_lane == 0)

    @flyc.jit
    def launch(
        qkv: fx.Int64,
        gate: fx.Int64,
        beta: fx.Int64,
        out_gate: fx.Int64,
        alog: fx.Int64,
        bias: fx.Int64,
        state: fx.Int64,
        offsets: fx.Int64,
        slots: fx.Int64,
        weight: fx.Int64,
        correction: fx.Int64,
        key_gate: fx.Int64,
        out: fx.Int64,
        qkv_stride: fx.Int32,
        gate_stride: fx.Int32,
        beta_stride: fx.Int32,
        og_stride: fx.Int32,
        state_pitch: fx.Int64,
        slot_stride: fx.Int32,
        out_stride: fx.Int32,
        bound: fx.Float32,
        eps: fx.Float32,
        batch: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            qkv,
            gate,
            beta,
            out_gate,
            alog,
            bias,
            state,
            offsets,
            slots,
            weight,
            correction,
            key_gate,
            out,
            qkv_stride,
            gate_stride,
            beta_stride,
            og_stride,
            state_pitch,
            slot_stride,
            out_stride,
            bound,
            eps,
        ).launch(grid=(batch, H, 1), block=(512, 1, 1), stream=stream)

    return launch
