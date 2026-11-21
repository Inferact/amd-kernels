# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry of the gated Engram update."""
import functools
import struct

import torch
from amd_kernels.warm_up import entry

from .kernel import H, HC, compile_gated_post

NV = 5
from .reference import CLAMP, EPS


@functools.cache
def _kernel(has_mask):
    return compile_gated_post(has_mask, NV)


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _bits(x):
    return struct.unpack("<i", struct.pack("<f", x))[0]


def _gated_post(hidden, kv, q_weight, k_weight, token_mask=None, out=None, eps=EPS, clamp=CLAMP):
    t = hidden.shape[0]
    assert hidden.dtype == kv.dtype == q_weight.dtype == k_weight.dtype == torch.bfloat16
    assert hidden.shape == (t, HC, H) and kv.shape == (t, (HC + 1) * H) and q_weight.shape == k_weight.shape == (HC, H)
    for x in (hidden, kv, q_weight, k_weight):
        assert x.is_contiguous() and x.device == hidden.device
    if token_mask is not None:
        assert token_mask.shape == (t,) and token_mask.dtype == torch.bool and token_mask.is_contiguous()
    if out is None:
        out = torch.empty_like(hidden)
    assert out.shape == hidden.shape and out.dtype == hidden.dtype and out.is_contiguous()
    if t:
        _run_compiled(
            _kernel(token_mask is not None), hidden.data_ptr(), kv.data_ptr(), q_weight.data_ptr(), k_weight.data_ptr(),
            0 if token_mask is None else token_mask.data_ptr(), out.data_ptr(), t, _bits(eps), _bits(clamp),
            torch.cuda.current_stream())
    return out


def _mode(hidden, kv, q_weight, k_weight, token_mask=None, out=None, **kwargs):
    """One mode per mask / no mask: the variants do not depend on the token count."""
    return (token_mask is not None,)


def _sizes(mode):
    return (1,)


def _inputs(mode, n, hidden, kv, q_weight, k_weight, token_mask=None, out=None, **kwargs):
    dev = hidden.device
    args = (torch.zeros(n, HC, H, dtype=torch.bfloat16, device=dev), torch.zeros(n, (HC + 1) * H, dtype=torch.bfloat16, device=dev),
            q_weight, k_weight)
    return args, ({"token_mask": torch.ones(n, dtype=torch.bool, device=dev)} if mode[0] else {})


gated_post = entry("engram/gated_post_bf16_h5120_hc4", mode=_mode, sizes=_sizes, inputs=_inputs)(_gated_post)
gated_post.__doc__ = """vLLM's Engram gate and residual update (``reference.py``): ``hidden`` bf16 [T, 4, 5120] and the ``wkv``
output ``kv`` bf16 [T, 25600], the layer's ``q_weight`` / ``k_weight`` bf16 [4, 5120], an optional bool ``token_mask``
[T] (False: no update). ``out`` may be ``hidden`` (in place); else a new tensor. All contiguous. Returns the output."""
