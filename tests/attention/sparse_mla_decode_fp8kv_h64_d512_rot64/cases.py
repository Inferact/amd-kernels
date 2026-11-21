# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Inputs of the ``sparse_mla_decode`` tests: ragged key lists over e4m3 pools, e4m3 NaN bytes in every cache row no
list names (the kernel must not read them), per-row lengths of the main list (the sliding window) and the extra list
(the top-k rows), padding rows with empty lists as CUDA-graph padding makes them."""

import math

import torch

H, D = 64, 512
SOFTMAX_SCALE = D**-0.5
NAN_BYTE = 0x7F


def require_gfx950():
    import pytest

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")


def e4m3(shape, std, gen):
    return (torch.randn(shape, generator=gen, device="cuda") * std).to(torch.float8_e4m3fn).view(torch.uint8)


def row_lengths(qlens, window=128, topk=512, ctx=(300, 100000), seed=0):
    """(n_main, n_extra, position) per query row: request r has ``qlens[r]`` rows at the last positions of a context
    of random length; the window is the keys up to the row's position, the top-k the (<= 512) compressed rows."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for ql in qlens:
        c = int(torch.randint(ctx[0], ctx[1], (1,), generator=gen))
        for j in range(ql):
            p = c + j
            out.append((min(window, p + 1), min(topk, p // 2), p))
    return out


def make_case(lens, *, pad=0, swa_block=32, comp_block=64, has_extra=True, q_std=1.5, k_std=1.0, seed=0, pitch=0,
              pool_rows=1 << 16):
    """``lens``: [(n_main, n_extra, position)] per query row. ``pad`` rows with empty lists follow. ``pitch`` extra
    rows per cache block (a padded page pitch). Returns the arguments of ``sparse_mla_decode`` (a dict)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed + 1)
    rows = len(lens) + pad
    n_swa = max(sum(n for n, _, _ in lens), 1)
    swa_rows = -(-max(2 * n_swa, 4096) // swa_block) * swa_block
    comp_rows = -(-pool_rows // comp_block) * comp_block
    swa = e4m3((swa_rows // swa_block, swa_block + pitch, D), k_std, gen)
    comp = e4m3((comp_rows // comp_block, comp_block + pitch, D), k_std, gen) if has_extra else None
    mp, xp, mi, xi = [0], [0], [], []
    for n_main, n_x, _ in lens:
        mi.append(torch.randperm(swa_rows, generator=cpu)[:n_main])
        mp.append(mp[-1] + n_main)
        if has_extra:
            xi.append(torch.randperm(comp_rows, generator=cpu)[:n_x])
            xp.append(xp[-1] + n_x)
    mp += [mp[-1]] * pad
    xp += [xp[-1]] * pad
    mi, xi = torch.cat(mi) if mi else torch.zeros(0, dtype=torch.long), (torch.cat(xi) if xi else torch.zeros(0, dtype=torch.long))
    poison(swa, mi, swa_block)
    swa = swa[:, :swa_block]
    if has_extra:
        poison(comp, xi, comp_block)
        comp = comp[:, :comp_block]
    pos = [p for _, _, p in lens] + [0] * pad
    ang = torch.rand((max(pos) + 8, 32), generator=gen, device="cuda") * 6.2832
    d = lambda x: x.int().cuda()
    return dict(
        q=e4m3((rows, H, D), q_std, gen),
        main_cache=swa, main_indptr=d(torch.tensor(mp)), main_indices=d(mi),
        extra_cache=comp, extra_indptr=d(torch.tensor(xp)) if has_extra else None,
        extra_indices=d(xi) if has_extra else None,
        attn_sink=torch.randn(H, generator=gen, device="cuda"),
        softmax_scale=SOFTMAX_SCALE,
        q_scale=torch.full((1,), 1.0, device="cuda"), kv_scale=torch.full((1,), 1.0, device="cuda"),
        positions=d(torch.tensor(pos)), cos_sin=torch.cat([ang.cos(), ang.sin()], dim=1).float(),
    )  # fmt: skip


def poison(pool, used, block):
    """e4m3 NaN bytes in every row of ``pool`` [blocks, block + pitch, 512] that ``used`` (slot ids) does not name."""
    keep = torch.zeros(pool.shape[0] * block, dtype=torch.bool)
    keep[used.long()] = True
    keep = keep.view(pool.shape[0], block).cuda()
    view = pool[:, :block]
    view[~keep] = NAN_BYTE
    if pool.shape[1] > block:
        pool[:, block:] = NAN_BYTE


def outputs(rows, device="cuda"):
    """The wo_a GEMM's input as the kernel fills it: e4m3 codes [rows, 8 * 4096], uint8 e8m0 scales [rows, 8 * 128]
    (NaN codes and 255 scales: a byte the kernel does not write shows)."""
    codes = torch.empty((rows, H * D), dtype=torch.float8_e4m3fn, device=device)
    codes.view(torch.uint8).fill_(NAN_BYTE)
    return codes, torch.full((rows, H * D // 32), 255, dtype=torch.uint8, device=device)


def log2(n):
    return int(math.log2(n))
