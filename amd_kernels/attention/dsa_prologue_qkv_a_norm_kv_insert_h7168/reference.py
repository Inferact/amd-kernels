# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``qkv_a_norm_kv_insert`` and ``indexer_prep`` (root README, rule 3): torch only.

``qkv_a_norm_kv_insert``: a row of the qkv_a projection is ``[q 1536 | kv 512 | k_pe 64]`` bf16 (no RoPE: the K3.1
DSA layers are NoPE, k_pe is stored as is).

1. ``y = RN_bf16(x * rsqrt(sum(x^2) / n + eps) * w)``, fp32 accumulation and one rounding (AITER ``rms_norm``), for
   q (n = 1536) and kv (n = 512).
2. fp8_ds_mla record of a token with slot >= 0 (this DCP rank owns it; ``slot_mapping`` is already localized) at
   ``cache[slot // 64, slot % 64]``, 656 bytes: 512 e4m3 codes of ``kv_y / scale`` (RNE), 4 fp32 tile scales
   (one per 128 columns: the smallest power of two >= max(amax / 448, 1e-4)), then the 64 bf16 of k_pe.
   Tokens past ``len(slot_mapping)`` and slots < 0 write nothing.

``indexer_prep``: the indexer's decode inputs of one layer.

1. Key (``k [n, 128]`` bf16, the wk_index output): ``y = RN_bf16(w * (rstd * (x - mean)) + b)`` (F.layer_norm in
   bf16 with fp32 statistics, ``w`` and ``b`` bf16), then MXFP4 and the insert into ``cache`` at ``slot``.
2. MXFP4 of a 32-value group: ``scale = 2^(e - 127)`` with ``e`` the biased exponent of the smallest power of two >=
   max(amax / 6, 1e-4), at most 254; codes are E2M1 of ``x / scale`` (RNE, ties to the even code), two per byte,
   the even column in the low nibble. Key page layout (physical page stride ``cache.stride(0)``): payload byte
   ``b`` of the 64 at ``(b // 16) * 1024 + slot_in_page * 16 + b % 16``, group scale ``g`` at ``4096 + g * 64 +
   (s % 16) * 4 + s // 16`` (``s`` the slot in the page).
3. Queries (``q [n, 32, 128]`` bf16): payload ``[n, 32, 64]`` u8; scales ``[n, 1, 4, 16, 4]`` u8 at
   ``(group, head % 16, head // 16)``, the entries of heads 32..63 zero.
4. ``weights = w_raw.float() * weight_scale`` (fp32), ``[n, 32]``.
5. Every ``topk_out`` row is set to -1.

Tolerances: the norm outputs are bit-equal except where the summation order moves a value one bf16 ulp (at least
``EXACT_MIN`` bit-equal, all within one ulp); the fp8 codes are within one step where the tile scale matches and at
least ``EXACT_MIN`` of the tile scales match; the key payload follows the key's rounding (at least ``EXACT_MIN`` of
its bytes equal); the query payload, scales and weights, which have no reduction, are bit-equal. Bytes of the caches
outside the written records stay unchanged.
"""

import torch

Q = 1536
KV = 512
PE = 64
RECORD = 656
D = 128
HEADS = 32
EXACT_MIN = 0.999
FP8_MAX = 448.0


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * r * w.float()).bfloat16()


def _pow2_exponent(s: torch.Tensor) -> torch.Tensor:
    b = s.contiguous().view(torch.int32)
    return ((b >> 23) & 255) + ((b & 0x7FFFFF) != 0).int()


def _pow2(e: torch.Tensor) -> torch.Tensor:
    return (e << 23).view(torch.float32)


def qkv_a_norm_kv_insert_reference(qkv_a, q_weight, kv_weight, eps, kv_cache=None, slot_mapping=None):
    """(q_out, kv_out, cache copy, written mask of the cache bytes)."""
    n = qkv_a.shape[0]
    q = rms_norm(qkv_a[:, :Q], q_weight, eps)
    kv = rms_norm(qkv_a[:, Q : Q + KV], kv_weight, eps)
    if kv_cache is None:
        return q, kv, None, None
    cache = kv_cache.clone()
    written = torch.zeros_like(kv_cache, dtype=torch.bool)
    pages, block = kv_cache.shape[0], kv_cache.shape[1]
    flat = cache.view(pages, block, RECORD)
    mask = written.view(pages, block, RECORD)
    m = min(n, slot_mapping.shape[0])
    tiles = kv[:m].float().view(m, 4, 128)
    s = (tiles.abs().amax(-1) / FP8_MAX).clamp_min(1e-4)
    e = _pow2_exponent(s)
    scale = _pow2(e)
    codes = (tiles / scale[..., None]).to(torch.float8_e4m3fn).view(torch.uint8).view(m, 512)
    scale_bytes = scale.contiguous().view(torch.uint8).view(m, 16)
    pe_bytes = qkv_a[:m, Q + KV : Q + KV + PE].contiguous().view(torch.uint8).view(m, 128)
    rec = torch.cat([codes, scale_bytes, pe_bytes], dim=1)
    for t in range(m):
        slot = int(slot_mapping[t])
        if slot >= 0:
            flat[slot // block, slot % block] = rec[t]
            mask[slot // block, slot % block] = True
    return q, kv, cache, written


def layer_norm(x, w, b, eps):
    xf = x.float()
    c = xf - xf.mean(-1, keepdim=True)
    r = torch.rsqrt(c.pow(2).mean(-1, keepdim=True) + eps)
    return (w.float() * (r * c) + b.float()).bfloat16()


def mxfp4_quantize(values: torch.Tensor):
    """``[..., 32k]`` fp32 -> (packed [..., 16k] u8, e8m0 exponent [..., k] u8)."""
    g = values.float().reshape(*values.shape[:-1], -1, 32)
    s = (g.abs().amax(-1) / 6.0).clamp_min(1.0e-4)
    e = _pow2_exponent(s).clamp_max(254)
    x = g / _pow2(e)[..., None]
    a = x.abs()
    code = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int()
            + (a > 2.5).int() + (a >= 3.5).int() + (a > 5.0).int())
    code = code | (((x < 0) & (code != 0)).int() << 3)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return packed.reshape(*values.shape[:-1], -1), e.to(torch.uint8)


def index_cache_views(cache: torch.Tensor):
    pages, stride, start = cache.shape[0], cache.stride(0), cache.storage_offset()
    payload = cache.as_strided((pages, 1, 4, 64, 16), (stride, 4096, 1024, 16, 1), start)
    scales = cache.as_strided((pages, 1, 4, 64), (stride, 256, 64, 1), start + 4096)
    return payload, scales


def indexer_prep_reference(k, ln_weight, ln_bias, ln_eps, slot_mapping, q, w_raw, weight_scale, topk_out, cache):
    """(q_payload, q_scales, weights, topk_out copy, cache copy, written mask of the cache bytes)."""
    n = q.shape[0]
    heads = q.shape[1]
    payload, e = mxfp4_quantize(q.float().reshape(n, heads, D))
    scales = torch.zeros((n, 1, 4, 16, 4), dtype=torch.uint8, device=q.device)
    for h in range(heads):
        scales[:, 0, :, h % 16, h // 16] = e[:, h, :]
    weights = w_raw.float() * weight_scale
    out = torch.full_like(topk_out, -1)
    cache_out = cache.clone()
    written = torch.zeros_like(cache, dtype=torch.bool)
    m = min(k.shape[0], slot_mapping.shape[0])
    if m:
        y = layer_norm(k[:m], ln_weight, ln_bias, ln_eps)
        kp, ke = mxfp4_quantize(y.float())
        pv, sv = index_cache_views(cache_out)
        pw, sw = index_cache_views(written)
        for t in range(m):
            slot = int(slot_mapping[t])
            if slot < 0:
                continue
            page, off = slot // 64, slot % 64
            pv[page, 0, :, off, :] = kp[t].view(4, 16)
            pw[page, 0, :, off, :] = True
            shuffled = (off % 16) * 4 + off // 16
            sv[page, 0, :, shuffled] = ke[t]
            sw[page, 0, :, shuffled] = True
    return payload.view(n, heads, 64), scales, weights, out, cache_out, written


def check_norm(got: torch.Tensor, ref: torch.Tensor, name: str) -> float:
    gb, rb = got.view(torch.int16).int(), ref.view(torch.int16).int()
    near = ((gb ^ rb) & 0x8000 == 0) & ((gb & 0x7FFF) - (rb & 0x7FFF)).abs().le(1)
    assert bool(near.all()), (name, int((~near).sum()))
    frac = (gb == rb).float().mean().item()
    assert frac >= EXACT_MIN, (name, frac)
    return frac


def check_kv_cache(got, ref, before, written, name: str) -> float:
    """Records written: k_pe bytes equal, tile scales equal in at least EXACT_MIN, codes within one e4m3 step where
    the scale matches; every other byte unchanged. Returns the fraction of equal scales."""
    assert torch.equal(got[~written], before[~written]), (name, "a byte the kernel must not write changed")
    g, r = got[written[..., 0]], ref[written[..., 0]]
    if not g.shape[0]:
        return 1.0
    assert torch.equal(g[:, 528:], r[:, 528:]), (name, "k_pe")
    gs, rs = g[:, 512:528].contiguous().view(torch.int32), r[:, 512:528].contiguous().view(torch.int32)
    same = gs == rs
    frac = same.float().mean().item()
    assert frac >= EXACT_MIN, (name, "scales", frac)
    gc = g[:, :512].view(g.shape[0], 4, 128)
    rc = r[:, :512].view(g.shape[0], 4, 128)
    near = (((gc ^ rc) & 0x80) == 0) & ((gc & 0x7F).int() - (rc & 0x7F).int()).abs().le(1)
    assert bool((near | ~same[..., None]).all()), (name, "codes")
    return frac


def check_index(got_cache, ref_cache, before, written, name: str) -> float:
    assert torch.equal(got_cache[~written], before[~written]), (name, "a byte the kernel must not write changed")
    g, r = got_cache[written], ref_cache[written]
    frac = (g == r).float().mean().item() if g.numel() else 1.0
    assert frac >= EXACT_MIN, (name, frac)
    return frac
