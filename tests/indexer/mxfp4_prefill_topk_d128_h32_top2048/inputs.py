# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch

CAP = 8192


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("needs gfx950")


def make_logits(kind: str, rows: int, width: int, seed: int = 0, device="cuda") -> torch.Tensor:
    """fp32 logits [rows, width] of a named distribution (the shapes the threshold estimate and the retry / fallback paths see)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(rows, width, device=device, generator=g)
    if kind == "gauss":
        return x
    if kind == "ties":  # quantized: the 2048th value is shared by many keys; includes -0.0
        return torch.round(x * 8) / 8
    if kind == "zeros":  # every key ties: exact fallback
        return torch.zeros_like(x)
    if kind == "blocky":  # page-correlated: the sample misses by a lot, rows are streamed again
        off = torch.randn(rows, (width + 63) // 64, device=device, generator=g) * 1.5
        return x * 0.5 + off.repeat_interleave(64, dim=1)[:, :width]
    if kind == "recency":
        x[:, -3000:] += 2.5
        return x
    if kind == "sink":
        x[:, :64] += 6
        x[:, -64:] += 3
        return x
    if kind == "spikes":
        idx = torch.randint(0, width, (rows, 100), device=device, generator=g)
        return x.scatter_(1, idx, x.gather(1, idx) + 50)
    if kind == "negative":
        return -torch.exp(x)
    if kind == "expo":
        return -torch.log(torch.rand(rows, width, device=device, generator=g) + 1e-9)
    raise ValueError(kind)


def make_windows(rows: int, width: int, kind: str = "causal", device="cuda"):
    """int32 (ks, ke) [rows]: ``causal`` one request ending at the row's position; ``requests`` several requests with
    unaligned starts; ``mixed`` window lengths around the candidate-list capacity."""
    if kind == "causal":
        ks = torch.zeros(rows, dtype=torch.int64)
        ke = torch.arange(width - rows + 1, width + 1, dtype=torch.int64)
    elif kind == "requests":
        bounds = torch.sort(torch.randint(1, width - 1, (5,), generator=torch.Generator().manual_seed(3))).values.tolist()
        bounds = [0] + bounds + [width]
        ks, ke = [], []
        for r in range(rows):
            j = r * (len(bounds) - 1) // rows
            lo, hi = bounds[j], bounds[j + 1]
            ks.append(lo + (r % 3) * 17 if hi - lo > 100 else lo)
            ke.append(min(hi, ks[-1] + 1 + (r * 7919) % max(1, hi - ks[-1])))
        ks, ke = torch.tensor(ks), torch.tensor(ke)
    elif kind == "mixed":
        lens = [0, 1, 2, 2047, 2048, 2049, CAP - 1, CAP, CAP + 1, CAP + 64, 3 * CAP, 4000, 9999]
        ks, ke = [], []
        for r in range(rows):
            n = min(lens[r % len(lens)], width - 70)
            lo = (r * 37) % (width - n - 64)
            ks.append(lo)
            ke.append(lo + n)
        ks, ke = torch.tensor(ks), torch.tensor(ke)
    else:
        raise ValueError(kind)
    return ks.to(torch.int32).to(device), ke.to(torch.int32).to(device)
