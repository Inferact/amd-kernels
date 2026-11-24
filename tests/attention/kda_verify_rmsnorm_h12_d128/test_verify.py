# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch

from amd_kernels.attention.kda_common import (
    kda_verify_rmsnorm,
    kda_verify_rmsnorm_unsupported_reason as unsupported_reason,
)
from amd_kernels.attention.kda_verify_rmsnorm_h12_d128.reference import kda_verify_rmsnorm_reference

H, D = 12, 128


def inputs(device, window, weight_dtype):
    rng = torch.Generator(device=device).manual_seed(29)

    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device=device, dtype=dtype, generator=rng)

    lengths = [window, 0, 1, window - 1]
    rows = sum(lengths)
    # Projection views have a common padded row pitch. State slots have a fractional head pitch.
    packed = rand((rows, 5 * H * D + 32))
    qkv = packed[:, : 3 * H * D]
    gate = packed[:, 3 * H * D : 4 * H * D].view(rows, H, D)
    og = packed[:, 4 * H * D : 5 * H * D].view(rows, H, D)
    beta = packed[:, -H:]
    pitch = H * D * D + 12
    state = rand((5, pitch), torch.float32)[:, : H * D * D].view(5, H, D, D)
    slots = torch.tensor([[3, 1], [1, 2], [0, 4], [2, 3]], dtype=torch.int32, device=device)[:, 0]
    cu = torch.tensor([0, *lengths], device=device).cumsum(0, dtype=torch.int32)
    record_shape = (len(lengths) + 1, H, window)
    return dict(
        qkv=qkv,
        gate=gate,
        beta=beta,
        out_gate=og,
        alog=rand((H,), torch.float32),
        bias=rand((H * D,), torch.float32),
        state=state,
        offsets=cu,
        slots=slots,
        weight=rand((D,), weight_dtype),
        correction=torch.full((*record_shape, D), -17.0, device=device),
        key_gate=torch.full((*record_shape, 2 * D), -17.0, device=device, dtype=torch.bfloat16),
        out=torch.full_like(og, -17.0),
        lower_bound=-5.0,
        eps=1e-5,
    )


def check_prefixes(a):
    """Replay every possible accepted prefix, including rejection after the first token."""
    cu = a["offsets"].tolist()
    for b, slot in enumerate(a["slots"].tolist()):
        if slot <= 0:
            assert torch.count_nonzero(a["out"][cu[b] : cu[b + 1]]) == 0
            continue
        recovered = expected = a["state"][slot]
        for pos, t in enumerate(range(cu[b], cu[b + 1])):
            _, k, v = a["qkv"][t].view(3, H, D).float()
            raw = a["gate"][t].float()
            kg = a["key_gate"][b, :, pos]
            torch.testing.assert_close(kg[:, :D].float(), k, rtol=0, atol=0)
            torch.testing.assert_close(kg[:, D:].float(), raw, rtol=0, atol=0)
            kn = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            decay = (
                a["lower_bound"] * torch.sigmoid(a["alog"].exp()[:, None] * (raw + a["bias"].view(H, D)))
            ).exp()
            decayed = expected * decay[:, None, :]
            u = (v - (decayed @ kn.unsqueeze(-1)).squeeze(-1)) * a["beta"][t].float().sigmoid()[:, None]
            expected = torch.baddbmm(decayed, u.unsqueeze(-1), kn.unsqueeze(1))
            recovered = recovered * decay[:, None, :] + a["correction"][b, :, pos, :, None] * kn[:, None, :]
            torch.testing.assert_close(recovered, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "window,weight_dtype", [(1, torch.float32), (4, torch.bfloat16), (5, torch.float32), (8, torch.bfloat16)]
)
def test_compact_records_and_gated_output(device, window, weight_dtype):
    if device == "cuda" and (
        not torch.cuda.is_available()
        or "gfx950" not in getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    ):
        pytest.skip("needs gfx950")
    a = inputs(device, window, weight_dtype)
    checkpoint = a["state"].clone()
    ref = a.copy()
    for name in ("correction", "key_gate", "out"):
        ref[name] = a[name].clone()
    kda_verify_rmsnorm_reference(**ref)
    if device == "cuda":
        kda_verify_rmsnorm(**a)
        torch.testing.assert_close(a["out"], ref["out"], atol=0.025, rtol=0.016)
        torch.testing.assert_close(a["correction"], ref["correction"], atol=2e-5, rtol=2e-4)
        assert torch.equal(a["key_gate"], ref["key_gate"])
        saved = [a[n].clone() for n in ("out", "correction", "key_gate")]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kda_verify_rmsnorm(**a)
        for n in ("out", "correction", "key_gate"):
            a[n].fill_(-17)
        graph.replay()
        for n, expected in zip(("out", "correction", "key_gate"), saved):
            assert torch.equal(a[n], expected)
        # Reuse the graph with live, null and ragged request rows in a different order.
        a["slots"].copy_(a["slots"].flip(0))
        lengths = a["offsets"].diff().flip(0)
        a["offsets"][1:].copy_(torch.cumsum(lengths, 0))
        a["gate"].neg_()
        for n in ("out", "correction", "key_gate"):
            a[n].fill_(-17)
            ref[n].fill_(-17)
        kda_verify_rmsnorm_reference(**ref)
        graph.replay()
        torch.testing.assert_close(a["out"], ref["out"], atol=0.025, rtol=0.016)
        torch.testing.assert_close(a["correction"], ref["correction"], atol=2e-5, rtol=2e-4)
        assert torch.equal(a["key_gate"], ref["key_gate"])
    else:
        a = ref
    assert torch.equal(a["state"], checkpoint)
    check_prefixes(a)
    cu = a["offsets"].tolist()
    for b in range(a["correction"].shape[0]):
        used = cu[b + 1] - cu[b] if b < len(cu) - 1 and int(a["slots"][b]) > 0 else 0
        assert (a["key_gate"][b, :, used:] == -17).all()
        assert (a["correction"][b, :, used:] == -17).all()


def test_declines_invalid_metadata_before_launch():
    a = inputs("cpu", 5, torch.bfloat16)
    a["correction"] = a["correction"][:, :, :0]
    assert "1..8" in unsupported_reason(**a)
    with pytest.raises(ValueError, match="1..8"):
        kda_verify_rmsnorm(**a)
