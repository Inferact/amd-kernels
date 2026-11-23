# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""N-gram hash and Engram row pull in one launch, rows written as the MXFP8 input of the ``wkv`` GEMM.

A workgroup of 256 threads takes one token and both layers. It finds the token's request (the one thread whose
``query_start_loc`` interval holds the token publishes it through LDS, so no search chain), reads the four lookback
sources (batch ids, else the request's window) through ``token_map``, and one lane per (layer, column) forms the 48 hash
columns' table rows (owner rank = row // peer_rows) into LDS. Then 16 lanes own one (layer, column) row: the 256 code
bytes are one 16-byte load per lane from the owner's table (a peer pointer: local HBM or xGMI, immutable data), the
lane's scale byte goes to its place in the shuffled scale layout. A lane issues all three of its rows' loads before
its first store, and the scale byte's load before the code store that would wait on the codes.

Codes are row-major, or with ``tile`` > 0 in the weight's (16, 16) layout (byte (m, k) at (m // 16) * 16 K + (k // 64)
* 1024 + ((k % 64) // 16) * 256 + (m % 16) * 16 + k % 16) for the GEMM's decode rows, the rows padded to a multiple of
``tile``: the rows past the last token write zero codes and the scale pad (127 = 1.0), up to a multiple of 32.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

DIM = 256
COLS = 24
HEADS = 8
SCALES = 8
K = COLS * DIM
KS = COLS * SCALES
RANKS = 8
LAYERS = 2
DEPTH = 3
NT = 256
ROWS_MAX = 1024
PAD_SCALE = 127


def compile_ngram_rows():
    name = "ngram_rows_mxfp8_d256_c24_ranks8"

    @fx.struct
    class Shared:
        req: fx.Array[fx.Int32, 4]
        row: fx.Array[fx.Int32, 2 * LAYERS * COLS]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        arg_ids: fx.Int64, arg_pos: fx.Int64, arg_qsl: fx.Int64, arg_lb: fx.Int64, arg_map: fx.Int64,
        mult: fx.Tensor, primes: fx.Tensor, offsets: fx.Tensor, peers: fx.Tensor,
        arg_codes: fx.Int64, arg_scales: fx.Int64,
        i32_m: fx.Int32, i32_slots: fx.Int32, i32_ids_dw: fx.Int32, i32_pos_dw: fx.Int32, i32_lb_stride: fx.Int32,
        i32_pad_id: fx.Int32, i32_image_id: fx.Int32, i64_peer_rows: fx.Int64,
        i32_codes_layer: fx.Int32, i32_scales_layer: fx.Int32, i32_tile: fx.Int32, i32_rows_padded: fx.Int32,
    ):
        tx = fx.thread_idx.x
        t = fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        slot = smem.req.ptr
        rows_lds = smem.row.ptr
        scale_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_scales, num_records_bytes=fx.Int64(i32_scales_layer) * LAYERS)
        codes_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arg_codes, num_records_bytes=fx.Int64(i32_codes_layer) * LAYERS)
        row = tx // 16
        part = tx % 16
        shuffled = i32_tile > 0

        def code_off(layer, c):
            """Byte offset of lane ``part``'s 16 code bytes of column ``c`` of row t of ``layer``."""
            flat = t * K + c * DIM + part * 16
            tiled = (t // 16) * (16 * K) + (c * 4 + part // 4) * 1024 + (part % 4) * 256 + (t % 16) * 16
            return shuffled.select(tiled, flat) + layer * i32_codes_layer

        def scale_off(layer, c, j):
            """Byte offset of scale ``j`` of column ``c`` of row t of ``layer`` (the shuffled layout)."""
            return ((((((t // 32) * COLS + c) * 4 + j % 4) * 16 + t % 16) * 2 + (j // 4) % 2) * 2 + (t // 16) % 2
                    + layer * i32_scales_layer)

        if t >= i32_m:
            if t < i32_rows_padded:
                zero = Vec.from_elements([fx.Int32(0)] * 4, fx.Int32)
                for i in range_constexpr(3):
                    r = row + 16 * i
                    layer = r // COLS
                    buffer_ops.buffer_store(zero, codes_rsrc, code_off(layer, r - layer * COLS), offset_is_bytes=True)
            if tx < KS:
                for layer in range_constexpr(LAYERS):
                    buffer_ops.buffer_store(
                        fx.Int32(PAD_SCALE).to(fx.Int8), scale_rsrc, scale_off(layer, tx // SCALES, tx % SCALES),
                        offset_is_bytes=True)
        else:
            qsl = buffer_ops.create_buffer_resource_from_addr(arg_qsl, num_records_bytes=(i32_slots + 1) * 4)
            ids = buffer_ops.create_buffer_resource_from_addr(arg_ids, num_records_bytes=i32_m * i32_ids_dw * 4)
            pos = buffer_ops.create_buffer_resource_from_addr(arg_pos, num_records_bytes=i32_m * i32_pos_dw * 4)
            tmap = buffer_ops.create_buffer_resource_from_addr(arg_map, num_records_bytes=0xFFFFFFFF)
            lb = buffer_ops.create_buffer_resource_from_addr(arg_lb, num_records_bytes=(i32_slots * i32_lb_stride) * 4)
            pad = fx.Int64(i32_pad_id)

            def source(tok_i32):
                """(compressed id, dead) of a token id."""
                safe = (tok_i32 >= 0).select(tok_i32, fx.Int32(0))
                src = fx.Int32(buffer_ops.buffer_load(tmap, safe, vec_width=1, dtype=fx.Int32))
                return fx.Int64(src), tok_i32 == i32_image_id

            # the batch tokens do not wait for the request search
            b_src, b_dead = [], []
            for s in range_constexpr(4):
                idx = t - s
                idx = (idx > 0).select(idx, fx.Int32(0))
                tok = fx.Int32(buffer_ops.buffer_load(ids, idx * i32_ids_dw, vec_width=1, dtype=fx.Int32))
                a, d = source(tok)
                b_src.append(a)
                b_dead.append(d)
            p = fx.Int64(fx.Int32(buffer_ops.buffer_load(pos, t * i32_pos_dw, vec_width=1, dtype=fx.Int32)))
            if tx == 0:
                slot[0] = i32_slots - 1
            fx.gpu.barrier()
            for i in range_constexpr(ROWS_MAX // NT):
                j = tx + i * NT
                lo = fx.Int32(buffer_ops.buffer_load(qsl, j, vec_width=1, dtype=fx.Int32, mask=j < i32_slots))
                hi = fx.Int32(buffer_ops.buffer_load(qsl, j + 1, vec_width=1, dtype=fx.Int32, mask=j < i32_slots))
                if (j < i32_slots) & (lo <= t) & (t < hi):
                    slot[0] = j
            fx.gpu.barrier()
            req = fx.Int32(slot[0])
            chunk_idx = fx.Int32(buffer_ops.buffer_load(qsl, req, vec_width=1, dtype=fx.Int32))
            chunk_idx = (chunk_idx < i32_m - 1).select(chunk_idx, i32_m - 1)
            chunk_start = fx.Int64(fx.Int32(buffer_ops.buffer_load(pos, chunk_idx * i32_pos_dw, vec_width=1, dtype=fx.Int32)))
            w_tok, w_src, w_dead = [], [], []
            for c in range_constexpr(DEPTH):
                tok = fx.Int32(buffer_ops.buffer_load(lb, req * i32_lb_stride + c, vec_width=1, dtype=fx.Int32))
                a, d = source(tok)
                w_tok.append(tok)
                w_src.append(a)
                w_dead.append(d)

            blocked = p < 0
            value = []
            for s in range_constexpr(4):
                look = p - s
                in_batch = look >= chunk_start
                col = chunk_start - 1 - look
                tok = (col == 0).select(w_tok[0], (col == 1).select(w_tok[1], w_tok[2]))
                wsrc = (col == 0).select(w_src[0], (col == 1).select(w_src[1], w_src[2]))
                wdead = (col == 0).select(w_dead[0], (col == 1).select(w_dead[1], w_dead[2]))
                known = (look < chunk_start) & (col < DEPTH) & (tok >= 0)
                src = in_batch.select(b_src[s], known.select(wsrc, pad))
                dead = in_batch.select(b_dead[s], known.select(wdead, fx.Boolean(False)))
                blocked = blocked | (look < 0) | dead
                value.append(blocked.select(pad, src))

            # one lane per (layer, column) forms the table row once; the 16 lanes of a row read it from LDS
            if tx < LAYERS * COLS:
                layer = tx // COLS
                c = tx - layer * COLS
                r1 = value[0] * fx.Int64(mult[layer * 4]) ^ value[1] * fx.Int64(mult[layer * 4 + 1])
                r2 = r1 ^ value[2] * fx.Int64(mult[layer * 4 + 2])
                r3 = r2 ^ value[3] * fx.Int64(mult[layer * 4 + 3])
                rolling = (c < HEADS).select(r1, (c < 2 * HEADS).select(r2, r3))
                index = rolling % fx.Int64(primes[tx]) + fx.Int64(offsets[tx])
                owner = index // i64_peer_rows
                rows_lds[2 * tx] = fx.Int32(owner)
                rows_lds[2 * tx + 1] = fx.Int32(index - owner * i64_peer_rows)
            fx.gpu.barrier()
            pulled = []
            for i in range_constexpr(3):
                r = row + 16 * i
                layer = r // COLS
                owner = fx.Int32(rows_lds[2 * r])
                local = fx.Int64(fx.Int32(rows_lds[2 * r + 1]))
                w_base = fx.Int64(peers[(layer * 2) * RANKS + owner])
                s_base = fx.Int64(peers[(layer * 2 + 1) * RANKS + owner])
                src = buffer_ops.create_buffer_resource_from_addr(w_base + local * DIM, num_records_bytes=DIM)
                ssrc = buffer_ops.create_buffer_resource_from_addr(s_base + local * SCALES, num_records_bytes=SCALES)
                vals = Vec(buffer_ops.buffer_load(src, part * 4, vec_width=4, dtype=fx.Int32))
                sb = buffer_ops.buffer_load(ssrc, part, vec_width=1, dtype=fx.Int8, mask=part < SCALES)
                pulled.append((layer, r - layer * COLS, vals, sb))
            for layer, c, vals, sb in pulled:
                buffer_ops.buffer_store(vals, codes_rsrc, code_off(layer, c), offset_is_bytes=True)
                buffer_ops.buffer_store(sb, scale_rsrc, scale_off(layer, c, part), mask=part < SCALES, offset_is_bytes=True)

    @flyc.jit
    def launch_ngram_rows_mxfp8_d256_c24_ranks8(
        arg_ids: fx.Int64, arg_pos: fx.Int64, arg_qsl: fx.Int64, arg_lb: fx.Int64, arg_map: fx.Int64,
        mult: fx.Tensor, primes: fx.Tensor, offsets: fx.Tensor, peers: fx.Tensor,
        arg_codes: fx.Int64, arg_scales: fx.Int64,
        i32_m: fx.Int32, i32_slots: fx.Int32, i32_ids_dw: fx.Int32, i32_pos_dw: fx.Int32, i32_lb_stride: fx.Int32,
        i32_pad_id: fx.Int32, i32_image_id: fx.Int32, i64_peer_rows: fx.Int64,
        i32_codes_layer: fx.Int32, i32_scales_layer: fx.Int32, i32_tile: fx.Int32, i32_rows_padded: fx.Int32,
        i32_grid: fx.Int32, stream: fx.Stream,
    ):
        kernel(arg_ids, arg_pos, arg_qsl, arg_lb, arg_map, mult, primes, offsets, peers, arg_codes, arg_scales,
               i32_m, i32_slots, i32_ids_dw, i32_pos_dw, i32_lb_stride, i32_pad_id, i32_image_id, i64_peer_rows,
               i32_codes_layer, i32_scales_layer, i32_tile, i32_rows_padded).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1), stream=stream)

    launch_ngram_rows_mxfp8_d256_c24_ranks8.kernel_name = name
    return launch_ngram_rows_mxfp8_d256_c24_ranks8
