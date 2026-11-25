# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Send buffers of all 8 ranks and vLLM's unpack-combine (installed, or its copy in triton_combine.py) for the tests
and the benchmark."""
import torch

from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192.reference import D, PACK, SUB, WORLD


def make_sends(t, step, dev, empty_ranks=()):
    """The 8 ranks' send buffers [8, T, 48, 66] bf16 (the same on every rank): base-2 LSEs around 10 with -inf for
    empty shards (``empty_ranks`` and one source per row), a few NaN and +inf LSEs, NaN partials where the LSE is -inf."""
    g = torch.Generator(device=dev).manual_seed(4321 + t * 7 + step * 100003)
    sends = []
    for s in range(WORLD):
        x = (torch.randn((WORLD, t, SUB, D), generator=g, device=dev) * 0.5).bfloat16()
        lse = torch.randn((WORLD, t, SUB), generator=g, device=dev) * 3 + 10
        if s in empty_ranks:
            lse[:] = float("-inf")
        rows = torch.arange(t, device=dev)
        lse[:, (rows + step) % WORLD == s] = float("-inf")
        r = torch.rand((WORLD, t, SUB), generator=g, device=dev)
        lse[r < 0.01] = float("nan")
        lse[(r > 0.01) & (r < 0.02)] = float("inf")
        x[lse == float("-inf")] = float("nan")
        buf = torch.empty((WORLD, t, SUB, PACK), dtype=torch.bfloat16, device=dev)
        buf[..., :D] = x
        buf[..., D:] = lse.unsqueeze(-1).view(torch.bfloat16)
        sends.append(buf)
    return sends


def vllm_combine(sends, rank):
    """vLLM's ``_dcp_a2a_unpack_combine_kernel`` on the buffer ``all_to_all_single`` would deliver to ``rank``."""
    from vllm.v1.attention.ops.dcp import _dcp_a2a_unpack_combine_kernel

    recv = torch.stack([s[rank] for s in sends])
    t = recv.shape[1]
    out = torch.empty((t, SUB, D), dtype=torch.bfloat16, device=recv.device)
    _dcp_a2a_unpack_combine_kernel[(t, SUB)](
        recv, out, out, *recv.stride(), *out.stride(), 0, 0,
        N=WORLD, HEAD_DIM=D, IS_BASE_E=False, RETURN_LSE=False, LSE_PACK_DIM=2,
    )
    return out.view(t, 16, 192)


def triton_combine(sends, rank):
    """``vllm_combine`` with the copy of vLLM's kernel in triton_combine.py."""
    from .triton_combine import unpack_combine_kernel

    recv = torch.stack([s[rank] for s in sends])
    t = recv.shape[1]
    out = torch.empty((t, SUB, D), dtype=torch.bfloat16, device=recv.device)
    unpack_combine_kernel[(t, SUB)](recv, out, *recv.stride(), *out.stride(), N=WORLD, HEAD_DIM=D)
    return out.view(t, 16, 192)


def exact_combine(sends, rank):
    """The combine the kernel matches bit for bit: vLLM's when installed, else its copy."""
    return (vllm_combine if has_vllm() else triton_combine)(sends, rank)


def has_vllm():
    try:
        from vllm.v1.attention.ops.dcp import _dcp_a2a_unpack_combine_kernel  # noqa: F401
    except Exception:
        return False
    return True


def exact_name():
    return "vLLM" if has_vllm() else "the copy of vLLM's kernel"
