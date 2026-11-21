# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA chunked prefill of one rank (12 heads, head dim 128, chunk 64): conv + gate + L2 norm prologue, delta-rule chunk
scan with a recurrent state, gated RMSNorm.

Contract (what vLLM's ``causal_conv1d_split_qkv`` + ``fused_kda_prologue`` + ``fused_kda_chunk`` + ``FusedRMSNormGated``
compute, kda_prefill_backend=fused):

- ``mixed`` [T, 4608] bf16 (any row stride): the in_proj q | k | v bands (channel = band * 1536 + head * 128 + d).
  ``conv_weight`` [3, 4, 1536] fp32 (band, tap, channel; tap 3 multiplies the current token). Per sequence the conv history is
  ``conv_state[slot, channel, 0:3]`` (oldest first; zeros when the sequence has no initial state);
  ``conv = sum_tap hist_tap * w_tap`` in fp32, ``silu`` in fp32, one rounding to bf16 (the stock conv's output dtype).
  Afterwards ``conv_state[slot, channel, 0:3]`` holds the last 3 raw inputs of the sequence (older history shifts in when
  the sequence is shorter than 3 tokens).
- ``q``, ``k`` are L2-normalized per head with eps 1e-6 in fp32 (``x * rsqrt(sum x^2 + 1e-6)``), ``beta = sigmoid(beta_logit)``.
- Gate: ``glog = lower_bound * sigmoid(exp(A_log[h]) * (raw_g + dt_bias[h, d]))`` (fp32, natural log, <= 0), summed down
  each 64-token chunk (fp32 cumulative sum, restarting at every chunk) and scaled by 1 / ln 2: ``g`` (log2 units).
- Chunk operands (fp32 math, per chunk and head, ``n`` valid rows, rows past ``n`` are zero)::

      Akk[i, j]  = beta_i * sum_d kn[i, d] kn[j, d] exp2(g[i, d] - g[j, d])          (j < i)
      A          = (I + Akk)^-1
      w          = A (beta * kn * exp2(g)),   u = A (beta * v)
      kg[j, d]   = kn[j, d] exp2(g[n-1, d] - g[j, d]),   decay[d] = exp2(g[n-1, d])
      Aqk[t, j]  = scale * sum_d qn[t, d] kn[j, d] exp2(g[t, d] - g[j, d])           (j <= t)
      qg[t, d]   = scale * qn[t, d] exp2(g[t, d])

- Recurrence per head, state ``S`` [V, K] fp32 (K = V = 128), read from ``state_cache[slot, h]`` (zeros when the sequence has
  no initial state), chunk by chunk::

      Vn = u - w S^T;   O = qg S^T + Aqk Vn;   S = S diag(decay) + Vn^T kg

  and written back to the same cache row after the last chunk. ``checkpoint_offsets[n]`` (a multiple of 64, 0 = off) also
  snapshots the state after that many tokens of sequence ``n`` into ``checkpoint_state[checkpoint_rows[n]]``.
- Output: ``o`` rounded to bf16, then the gated RMSNorm per head row (128 values):
  ``out = bf16(o * rsqrt(mean(o^2) + eps) * norm_weight * sigmoid(gate))``, fp32 math on the bf16 ``o``.

The kernels round the chunk operands (qg, kg, w, u, Aqk, the state fed to the MFMAs, Vn) to bf16 and accumulate in fp32; this
reference stays in fp32 (the inverse in fp64). Tolerance of the kernels against it: out rel-L2 <= 6e-3 (the stock path is
within 4e-3 of its own Triton twin), state rel-L2 <= 1e-3; the kernels are deterministic run to run.

The workspace between the prologue and the scan is one uint8 tensor [H, Nc, RECORD_BYTES] (head-major, Nc = total chunks; one
record per chunk and head: w | qg | kgt | aqk | decay | u at the RECORD_* byte offsets), its tiles bf16 fragment-major (``pack_frag``): a [R, C] tile is stored as
[R / 16][C / 32][64 lanes][8], lane ``l`` (row ``l % 16``, group ``q = l // 16``) holding columns
``32 ks + 16 (s // 4) + 4 q + s % 4`` of its row at slot ``s``, so one 1 KB fragment is one linear read and the MFMA's
reduction is permuted identically on both operands.

The one-launch path adds a state record per chunk and head (walk.py, 48 KB): the bf16 fragments of the state the chunk starts from
(``S^T`` of each 16-column value slice as the B operand of the first product) and of ``Vn = u - w S^T`` (the operand of the state
update and of the ``Aqk`` product); the output pass computes ``O = qg S^T + Aqk Vn`` from them and the record's ``qg`` / ``aqk``,
which is the reference's ``chunk_reference`` with the same bf16 operands.
"""

import math

import torch

NH = 12
D = 128
LP = NH * D
BT = 64
SCALE = D ** -0.5
EPS_QK = 1e-6
RCP_LN2 = 1.4426950408889634
RECORD_W, RECORD_QG, RECORD_KGT, RECORD_AQK, RECORD_DEC, RECORD_U = 0, 16384, 32768, 49152, 57344, 58368
RECORD_BYTES = 74752


def _f(x):
    return x.float()


def pack_frag(t, rows, cols):
    """[..., rows, cols] -> [..., rows // 16, cols // 32, 64, 8] (bf16 fragment-major tile)."""
    lead = t.shape[:-2]
    x = t.reshape(*lead, rows // 16, 16, cols // 32, 2, 4, 4)  # (mt, i, ks, s/4, q, s%4)
    n = len(lead)
    x = x.permute(*range(n), n, n + 2, n + 4, n + 1, n + 3, n + 5)
    return x.reshape(*lead, rows // 16, cols // 32, 64, 8).contiguous()


def unpack_frag(x, rows, cols):
    lead = x.shape[:-4]
    n = len(lead)
    y = x.reshape(*lead, rows // 16, cols // 32, 4, 16, 2, 4)  # (mt, ks, q, i, s/4, s%4)
    y = y.permute(*range(n), n, n + 3, n + 1, n + 4, n + 2, n + 5)  # (mt, i, ks, s/4, q, s%4)
    return y.reshape(*lead, rows, cols).contiguous()


def pack_u(u):
    """[..., 64, 128] -> [..., 8 (v slice), 4 (token tile), 64 lanes, 4]: lane (v % 16 + 16 q) holds token 16 mt + 4 q + e."""
    lead = u.shape[:-2]
    n = len(lead)
    x = u.reshape(*lead, 4, 4, 4, 8, 16)  # (mt, q, e, vs, v)
    x = x.permute(*range(n), n + 3, n, n + 1, n + 4, n + 2)  # (vs, mt, q, v, e)
    return x.reshape(*lead, 8, 4, 64, 4).contiguous()


def unpack_u(x):
    lead = x.shape[:-4]
    n = len(lead)
    y = x.reshape(*lead, 8, 4, 4, 16, 4)  # (vs, mt, q, v, e)
    y = y.permute(*range(n), n + 1, n + 2, n + 4, n, n + 3)  # (mt, q, e, vs, v)
    return y.reshape(*lead, 64, 128).contiguous()


def chunk_table(cu_seqlens):
    """(seq, chunk-in-seq, first token, rows) per chunk, and chunk_offsets [N + 1]."""
    cu = cu_seqlens.tolist()
    rows = []
    offs = [0]
    for n in range(len(cu) - 1):
        ln = cu[n + 1] - cu[n]
        nc = -(-ln // BT)
        for c in range(nc):
            rows.append((n, c, cu[n] + c * BT, min(BT, ln - c * BT)))
        offs.append(offs[-1] + nc)
    return rows, offs


def conv_reference(mixed, conv_weight, conv_state, state_indices, has_initial_state, cu_seqlens):
    """(conv outputs [T, 4608] bf16, new conv_state values [N, 4608, 3] bf16 of each sequence's slot)."""
    T = mixed.shape[0]
    wt = _f(conv_weight).reshape(3, 4, LP).permute(1, 0, 2).reshape(4, 3 * LP)  # [tap, channel]
    out = torch.empty(T, 3 * LP, dtype=torch.bfloat16, device=mixed.device)
    new_state = []
    cu = cu_seqlens.tolist()
    for n in range(len(cu) - 1):
        t0, t1 = cu[n], cu[n + 1]
        ln = t1 - t0
        x = _f(mixed[t0:t1])
        if bool(has_initial_state[n]):
            hist = _f(conv_state[int(state_indices[n])]).t().clone()  # [3, channel]
        else:
            hist = torch.zeros(3, 3 * LP, device=mixed.device)
        xe = torch.cat([hist, x], 0)  # [3 + ln, channel]
        acc = sum(xe[tp:tp + ln] * wt[tp] for tp in range(4))
        out[t0:t1] = (acc * torch.sigmoid(acc)).bfloat16()
        new_state.append(xe[-3:].t().bfloat16())
    return out, torch.stack(new_state) if new_state else None


def gate_reference(raw_g, A_log, dt_bias, lower_bound):
    """Natural-log gate [T, 12, 128] fp32."""
    a = _f(A_log).exp().view(1, NH, 1)
    return lower_bound * torch.sigmoid(a * (_f(raw_g).view(-1, NH, D) + _f(dt_bias).view(1, NH, D)))


def prologue_reference(q, k, v, raw_g, beta_logit, A_log, dt_bias, lower_bound, scale, cu_seqlens):
    """Chunk operands, fp32, padded to 64 rows: a dict of [Nc, H, ...] tensors (qg, nw = -w, u, kg, aqk, decay, g)."""
    dev = q.device
    glog = gate_reference(raw_g, A_log, dt_bias, lower_bound)
    beta = torch.sigmoid(_f(beta_logit).view(-1, NH))
    table, _ = chunk_table(cu_seqlens)
    nc = len(table)
    z = lambda *s: torch.zeros(nc, NH, *s, device=dev)
    qg, nw, u, kgt, aqk, dec, gg = z(BT, D), z(BT, D), z(BT, D), z(D, BT), z(BT, BT), z(D), z(BT, D)
    tri = torch.tril(torch.ones(BT, BT, device=dev), -1)
    eye = torch.eye(BT, device=dev, dtype=torch.float64)
    for ci, (_, _, t0, n) in enumerate(table):
        qc, kc, vc = (_f(x[t0:t0 + n]).view(n, NH, D).transpose(0, 1) for x in (q, k, v))  # [H, n, D]
        qn = qc * torch.rsqrt((qc * qc).sum(-1, keepdim=True) + EPS_QK)
        kn = kc * torch.rsqrt((kc * kc).sum(-1, keepdim=True) + EPS_QK)
        g = torch.cumsum(glog[t0:t0 + n].transpose(0, 1), dim=1) * RCP_LN2  # [H, n, D]
        b = beta[t0:t0 + n].t()  # [H, n]
        mask = torch.tril(torch.ones(n, n, device=dev, dtype=torch.bool))[None, :, :, None]
        dg = torch.exp2(torch.where(mask, g[:, :, None, :] - g[:, None, :, :], float("-inf")))  # [H, i, j, D], j <= i
        akk = torch.einsum("hid,hjd,hijd->hij", kn, kn, dg) * b[:, :, None]
        akk = akk * torch.tril(torch.ones(n, n, device=dev), -1)
        inv = torch.linalg.solve_triangular(
            (torch.eye(n, device=dev, dtype=torch.float64) + akk.double()), torch.eye(n, device=dev, dtype=torch.float64).expand(NH, n, n), upper=False
        ).float()
        eg = torch.exp2(g)
        w = inv @ (b[:, :, None] * kn * eg)
        uu = inv @ (b[:, :, None] * vc)
        kgc = kn * torch.exp2(g[:, n - 1:n, :] - g)
        aq = scale * torch.einsum("hid,hjd,hijd->hij", qn, kn, dg)
        aq = aq * torch.tril(torch.ones(n, n, device=dev))
        qgc = scale * qn * eg
        qg[ci, :, :n] = qgc
        nw[ci, :, :n] = -w
        u[ci, :, :n] = uu
        kgt[ci, :, :, :n] = kgc.transpose(1, 2)
        aqk[ci, :, :n, :n] = aq
        dec[ci] = torch.exp2(g[:, n - 1, :])
        gg[ci, :, :n] = g
    return dict(qg=qg, nw=nw, u=u, kgt=kgt, aqk=aqk, decay=dec, g=gg)


def pack_workspace(ws):
    """Fragment-major bf16 workspace of ``prologue_reference``'s dict, head-major: [H, Nc, ...] tensors."""
    bf = lambda x: x.bfloat16()
    return dict(
        w=pack_frag(bf(ws["nw"]).transpose(0, 1), BT, D).contiguous(),
        qg=pack_frag(bf(ws["qg"]).transpose(0, 1), BT, D).contiguous(),
        kgt=pack_frag(bf(ws["kgt"]).transpose(0, 1), D, BT).contiguous(),
        aqk=pack_frag(bf(ws["aqk"]).transpose(0, 1), BT, BT).contiguous(),
        u=pack_u(bf(ws["u"]).transpose(0, 1)).contiguous(),
        decay=ws["decay"].transpose(0, 1).contiguous(),
    )


def pack_record(ws):
    """The uint8 workspace [H, Nc, RECORD_BYTES] of ``prologue_reference``'s dict."""
    p = pack_workspace(ws)
    h, nc = p["w"].shape[:2]
    rec = torch.zeros(h, nc, RECORD_BYTES, dtype=torch.uint8, device=p["w"].device)
    for key, off, n in (("w", RECORD_W, 16384), ("qg", RECORD_QG, 16384), ("kgt", RECORD_KGT, 16384), ("aqk", RECORD_AQK, 8192),
                        ("u", RECORD_U, 16384)):
        rec[:, :, off:off + n] = p[key].reshape(h, nc, -1).view(torch.uint8).reshape(h, nc, n)
    rec[:, :, RECORD_DEC:RECORD_DEC + 512] = p["decay"].reshape(h, nc, 128).contiguous().view(torch.uint8).reshape(h, nc, 512)
    return rec


def unpack_record(rec):
    """Inverse of ``pack_record`` -> [Nc, H, ...] fp32 logical tiles (``unpack_workspace``'s dict)."""
    h, nc = rec.shape[:2]
    get = lambda off, n, shape: rec[:, :, off:off + n].contiguous().view(torch.bfloat16).reshape(h, nc, *shape)
    return unpack_workspace(dict(
        w=get(RECORD_W, 16384, (4, 4, 64, 8)), qg=get(RECORD_QG, 16384, (4, 4, 64, 8)), kgt=get(RECORD_KGT, 16384, (8, 2, 64, 8)),
        aqk=get(RECORD_AQK, 8192, (4, 2, 64, 8)), u=get(RECORD_U, 16384, (8, 4, 64, 4)),
        decay=rec[:, :, RECORD_DEC:RECORD_DEC + 512].contiguous().view(torch.float32).reshape(h, nc, 128)))


def unpack_workspace(w):
    """Inverse of ``pack_workspace`` (head-major tensors -> [Nc, H, ...] fp32 logical tiles)."""
    t = lambda x: x.transpose(0, 1).float()
    return dict(
        nw=t(unpack_frag(w["w"], BT, D)),
        qg=t(unpack_frag(w["qg"], BT, D)),
        kgt=t(unpack_frag(w["kgt"], D, BT)),
        aqk=t(unpack_frag(w["aqk"], BT, BT)),
        u=t(unpack_u(w["u"])),
        decay=w["decay"].transpose(0, 1).float(),
    )


def chunk_reference(ws, cu_seqlens, state_cache, state_indices, has_initial_state, checkpoint_state=None,
                    checkpoint_offsets=None, checkpoint_rows=None):
    """Delta-rule scan over the logical chunk operands; updates ``state_cache`` (and ``checkpoint_state``) in place.

    Returns o [T, 12, 128] fp32.
    """
    T = int(cu_seqlens[-1])
    o = torch.zeros(T, NH, D, device=ws["qg"].device)
    table, _ = chunk_table(cu_seqlens)
    S = None
    for ci, (n, c, t0, rows) in enumerate(table):
        if c == 0:
            slot = int(state_indices[n])
            S = state_cache[slot].float().clone() if bool(has_initial_state[n]) else torch.zeros(NH, D, D, device=o.device)
        nw, qg, kgt, aq, u, dec = (ws[k][ci] for k in ("nw", "qg", "kgt", "aqk", "u", "decay"))
        vn = u + nw @ S.transpose(1, 2)  # [H, 64, V]
        oc = qg @ S.transpose(1, 2) + aq @ vn
        S = S * dec[:, None, :] + vn.transpose(1, 2) @ kgt.transpose(1, 2)
        o[t0:t0 + rows] = oc[:, :rows].transpose(0, 1)
        nxt = table[ci + 1] if ci + 1 < len(table) else None
        if checkpoint_offsets is not None and int(checkpoint_offsets[n]) > 0 and (c + 1) * BT == int(checkpoint_offsets[n]):
            row = int(checkpoint_rows[n]) if checkpoint_rows is not None else n
            if row >= 0:
                checkpoint_state[row] = S.to(checkpoint_state.dtype)
        if nxt is None or nxt[0] != n:
            state_cache[int(state_indices[n])] = S.to(state_cache.dtype)
    return o


def gated_norm_reference(o, gate, norm_weight, eps):
    """o [T, 12, 128] (bf16 values), gate [T, 12, 128] -> bf16 [T, 12, 128]."""
    of = _f(o)
    rstd = torch.rsqrt((of * of).mean(-1, keepdim=True) + eps)
    return (of * rstd * _f(norm_weight) * torch.sigmoid(_f(gate))).bfloat16()


def kda_prefill_reference(mixed, raw_g, beta_logit, gate, A_log, dt_bias, lower_bound, conv_weight, conv_state, state_cache,
                          state_indices, has_initial_state, cu_seqlens, norm_weight, norm_eps, scale=SCALE,
                          checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None):
    """End to end: returns ``out`` [T, 12, 128] bf16; updates ``conv_state`` and ``state_cache`` (and the checkpoint) in place."""
    conv, new_cs = conv_reference(mixed, conv_weight, conv_state, state_indices, has_initial_state, cu_seqlens)
    q, k, v = conv[:, :LP], conv[:, LP:2 * LP], conv[:, 2 * LP:]
    ws = prologue_reference(q, k, v, raw_g, beta_logit, A_log, dt_bias, lower_bound, scale, cu_seqlens)
    o = chunk_reference(ws, cu_seqlens, state_cache, state_indices, has_initial_state, checkpoint_state, checkpoint_offsets,
                        checkpoint_rows)
    for n in range(len(cu_seqlens) - 1):
        conv_state[int(state_indices[n])] = new_cs[n]
    return gated_norm_reference(o.bfloat16(), gate, norm_weight, norm_eps)
