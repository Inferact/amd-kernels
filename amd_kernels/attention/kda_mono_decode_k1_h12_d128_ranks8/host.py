# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry and workspace of the KDA decode kernel K1 (reference.py is the contract)."""

import functools

import torch

from amd_kernels import warm_up

from . import kernel as K
from .kernel import BLOCKS, MAX_S, KdaPreBuild

NAME = "attention/kda_mono_decode_k1_h12_d128_ranks8"
HIDDEN, PROJ, NPROJ, NH, HD = K.HIDDEN, K.PROJ, K.NPROJ, K.NH, K.HD


def _run_compiled(launcher, *args) -> None:
    import flydsl.compiler as flyc
    from flydsl.utils import env

    if env.compile.compile_only:
        flyc.compile(launcher, *args)
        return
    compiled = getattr(launcher, "_cf", None)
    if compiled is not None:
        compiled(*args)
        return
    launcher._cf = flyc.compile(launcher, *args)


@warm_up.launcher
def compile_kda_mono_decode_k1(key: KdaPreBuild):
    return K.build(key)


@functools.cache
def _launcher(key: KdaPreBuild):
    return compile_kda_mono_decode_k1(key)


class Workspace:
    """Per-device mailbox state shared by every launch of a step: the scratch the CTAs hand off through, the step
    epoch (``begin_step``) and the fail-stop words. Tags carry the epoch and the launch's ``layer`` slot, so nothing
    is zeroed between launches."""

    def __init__(self, device, max_tokens: int = MAX_S, max_blocks: int = 8):
        key = KdaPreBuild(tokens=max_tokens, qlen=max_tokens, nblocks=max_blocks, delta=True, state_len=K.CONV_W - 1 + max_tokens)
        self.scratch = torch.zeros(K.scratch_bytes(key), dtype=torch.uint8, device=device)
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.health = torch.zeros(K.HEALTH_WORDS, dtype=torch.int32, device=device)
        self.flags = torch.zeros(1, dtype=torch.int32, device=device)

    def begin_step(self) -> None:
        """Once a step before its first launch (a CUDA graph captures it)."""
        self.epoch.add_(1)

    def failed(self) -> bool:
        """A wait of an earlier launch gave up (10 s): its step's results are void."""
        return bool(self.flags.any() or self.health[K.FAILED].item())


def _mode(ws, layer, prefix, delta, blocks, ares_nw, ares_qk, in_nw, w_in, w_fb, conv_w, conv_state, a_log, dt_bias,
          on_w, rstate, st_idx, num_acc, *, qlen, nblocks, write_idx=-1, eps=1e-5, out_eps=1e-5, onorm_eps=1e-5,
          lower_bound=-5.0, timeline=None):
    """What a call fixes about its kernel besides the token count (``amd_kernels.warm_up``)."""
    return (qlen, nblocks, delta is not None, write_idx, conv_state.size(-1), eps, out_eps, onorm_eps, lower_bound,
            timeline is not None)


def _sizes(mode):
    qlen = mode[0]
    return tuple(range(qlen, MAX_S + 1, qlen))


def _inputs(mode, n, ws, layer, prefix, delta, blocks, ares_nw, ares_qk, in_nw, w_in, w_fb, conv_w, conv_state, a_log,
            dt_bias, on_w, rstate, st_idx, num_acc, *, qlen, nblocks, write_idx=-1, eps=1e-5, out_eps=1e-5,
            onorm_eps=1e-5, lower_bound=-5.0, timeline=None):
    dev = prefix.device
    z = lambda *s, dt=torch.bfloat16: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
    # FlyDSL's handle for a launcher whose arguments are raw addresses runs the kernel once: on a workspace and
    # null-slot requests of its own, it touches nothing a caller holds
    args = (Workspace(dev), layer, z(n, HIDDEN), None if delta is None else z(n, HIDDEN), z(n, blocks.size(1), HIDDEN),
            ares_nw, ares_qk, in_nw, w_in, w_fb, conv_w, conv_state, a_log, dt_bias, on_w, rstate,
            z(n // qlen, st_idx.size(1), dt=torch.int32), z(n // qlen, dt=torch.int32) + 1)
    return args, dict(qlen=qlen, nblocks=nblocks, write_idx=write_idx, eps=eps, out_eps=out_eps,
                      onorm_eps=onorm_eps, lower_bound=lower_bound, timeline=None if timeline is None else
                      torch.zeros(BLOCKS * K.TL_POINTS, dtype=torch.int64, device=dev))


@warm_up.entry(NAME, mode=_mode, sizes=_sizes, inputs=_inputs)
def kda_mono_decode_k1(ws, layer, prefix, delta, blocks, ares_nw, ares_qk, in_nw, w_in, w_fb, conv_w, conv_state, a_log,
                       dt_bias, on_w, rstate, st_idx, num_acc, *, qlen, nblocks, write_idx=-1, eps=1e-5, out_eps=1e-5,
                       onorm_eps=1e-5, lower_bound=-5.0, timeline=None):
    """One KDA layer of a spec-verify step from the AttnRes seam to the gated core output (see reference.py).

    ``layer``: this launch's mailbox slot (0 .. 255), distinct among a step's launches. ``prefix`` [S, 7168] bf16
    (updated in place, with ``delta`` added first when given), ``blocks`` [S, >= nblocks (+1 with ``write_idx``), 7168],
    ``w_in`` [6288, 7168], ``w_fb`` [1536, 128], ``conv_state`` [slots, 4608, state_len] or its transposed view,
    ``rstate`` [slots, 12, 128, 128] fp32, ``st_idx`` [S / qlen, >= qlen], ``num_acc`` [S / qlen] int32.
    Returns ``core`` [S, 1536] bf16.
    """
    s = prefix.size(0)
    sl = conv_state.size(-1)
    assert 1 <= s <= MAX_S and s % qlen == 0 and qlen <= sl, (s, qlen, sl)
    assert prefix.is_contiguous() and (delta is None or delta.is_contiguous())
    assert prefix.shape == (s, HIDDEN) and prefix.dtype == torch.bfloat16
    assert blocks.size(0) == s and blocks.size(2) == HIDDEN and blocks.stride(2) == 1
    assert blocks.stride(0) % 8 == 0 and blocks.stride(1) % 8 == 0 and blocks.data_ptr() % 16 == 0
    assert 1 <= nblocks <= min(8, blocks.size(1)) and (write_idx < 0 or nblocks <= write_idx < blocks.size(1))
    assert w_in.shape == (NPROJ, HIDDEN) and w_in.is_contiguous() and w_fb.shape == (PROJ, HD) and w_fb.is_contiguous()
    assert rstate.dtype == torch.float32 and rstate.shape[1:] == (NH, HD, HD) and rstate.stride()[1:] == (HD * HD, HD, 1)
    assert st_idx.dtype == torch.int32 and st_idx.stride(1) == 1 and num_acc.dtype == torch.int32
    key = KdaPreBuild(tokens=s, qlen=qlen, nblocks=nblocks, delta=delta is not None, state_len=sl, eps=eps,
                      out_eps=out_eps, onorm_eps=onorm_eps, lower_bound=lower_bound, write_idx=write_idx,
                      timeline=timeline is not None)
    assert ws.scratch.numel() >= K.scratch_bytes(key)
    core = torch.empty((s, PROJ), dtype=torch.bfloat16, device=prefix.device)
    _run_compiled(
        _launcher(key), prefix.data_ptr(), 0 if delta is None else delta.data_ptr(), blocks.data_ptr(), blocks.stride(0),
        blocks.stride(1), ares_nw.data_ptr(), ares_qk.data_ptr(), in_nw.data_ptr(), w_in.data_ptr(), w_fb.data_ptr(),
        conv_w.data_ptr(), conv_state.data_ptr(), conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        a_log.data_ptr(), dt_bias.data_ptr(), on_w.data_ptr(), rstate.data_ptr(), rstate.stride(0), st_idx.data_ptr(),
        st_idx.stride(0), num_acc.data_ptr(), core.data_ptr(), ws.scratch.data_ptr(), layer, ws.epoch.data_ptr(),
        ws.health.data_ptr(), ws.flags.data_ptr(), 0 if timeline is None else timeline.data_ptr(),
        torch.cuda.current_stream())
    return core
