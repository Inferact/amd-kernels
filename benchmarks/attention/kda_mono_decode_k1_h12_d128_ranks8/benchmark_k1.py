# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""K1 under CUDA graph replay: one graph of ``SETS`` launches over distinct weight sets (together larger than the
LLC, so every launch streams in_proj from HBM as a layer of the model does), median us a launch [min-max over
rounds], with its roofline: bytes read and written against the HBM ceilings measured on MI355X (6.5 TB/s
read, 6.0 TB/s write, 1.57 us graph launch gap).

    python benchmarks/attention/kda_mono_decode_k1_h12_d128_ranks8/benchmark_k1.py [--shapes 12x6 6x6 ...] [--nb 4]
"""

import argparse
import statistics

import torch

from amd_kernels.attention.kda_mono_decode_k1_h12_d128_ranks8 import Workspace, kda_mono_decode_k1

HIDDEN, NPROJ, PROJ, NH, HD = 7168, 6288, 1536, 12, 128
READ_TBS, WRITE_TBS, LAUNCH_US = 6.5, 6.0, 1.57
SETS, ROUNDS, REPLAYS = 4, 11, 10


def sets(S, L, nb):
    """SETS weight sets and their own state, sharing nothing larger than the norm vectors."""
    out = []
    sl = 3 + (L - 1)
    nreq = S // L
    slots = S + 4
    for i in range(SETS):
        g = torch.Generator(device="cuda").manual_seed(i)
        r = lambda *s, sc=1.0, dt=torch.bfloat16: (torch.randn(*s, generator=g, device="cuda") * sc).to(dt)  # noqa: E731
        w = dict(
            prefix=r(S, HIDDEN, sc=2.0), delta=r(S, HIDDEN, sc=0.5), blocks=r(S, nb, HIDDEN, sc=2.0),
            ares_nw=r(HIDDEN), ares_qk=r(HIDDEN, sc=0.05), in_nw=r(HIDDEN), w_in=r(NPROJ, HIDDEN, sc=0.02),
            w_fb=r(PROJ, HD, sc=0.08), conv_w=r(3 * PROJ, 4, sc=0.4, dt=torch.float32),
            conv_state=r(slots, 3 * PROJ, sl, sc=0.5), a_log=r(NH, sc=0.5, dt=torch.float32),
            dt_bias=r(PROJ, sc=0.5, dt=torch.float32), on_w=r(HD), rstate=r(slots, NH, HD, HD, sc=0.05, dt=torch.float32),
            st_idx=torch.arange(3, 3 + S, dtype=torch.int32, device="cuda").view(nreq, L),
            num_acc=torch.full((nreq,), L, dtype=torch.int32, device="cuda"),
        )  # fmt: skip
        w["w_in"][NPROJ - 4 :].zero_()
        out.append(w)
    return out


def launch(ws, i, w, L, nb):
    return kda_mono_decode_k1(
        ws, i, w["prefix"], w["delta"], w["blocks"], w["ares_nw"], w["ares_qk"], w["in_nw"], w["w_in"], w["w_fb"],
        w["conv_w"], w["conv_state"], w["a_log"], w["dt_bias"], w["on_w"], w["rstate"], w["st_idx"], w["num_acc"],
        qlen=L, nblocks=nb,
    )  # fmt: skip


def replay_us(S, L, nb):
    ws = Workspace("cuda")
    ws_sets = sets(S, L, nb)
    for i, w in enumerate(ws_sets):
        ws.begin_step()
        launch(ws, i, w, L, nb)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ws.begin_step()
        for i, w in enumerate(ws_sets):
            launch(ws, i, w, L, nb)
    for _ in range(3):
        g.replay()
    res = []
    for _ in range(ROUNDS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(REPLAYS):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        res.append(a.elapsed_time(b) * 1000 / REPLAYS / SETS)
    assert not ws.failed()
    return res


def traffic(S, L, nb):
    """(bytes read, bytes written) of a launch."""
    nreq = S // L
    state = NH * HD * HD * 4
    read = NPROJ * HIDDEN * 2 + PROJ * HD * 2 + S * (nb + 2) * HIDDEN * 2 + 3 * HIDDEN * 2 + nreq * state
    write = S * state + S * HIDDEN * 2 + S * PROJ * 2
    return read, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="*", default=["6x6", "12x6", "4x4", "8x4", "16x4", "8x8", "16x8"])
    ap.add_argument("--nb", type=int, default=4)
    args = ap.parse_args()
    from amd_kernels import warm_up

    warm_up.set_auto(False)
    print(f"K1 (nb={args.nb} AttnRes blocks), CUDA graph replay, {SETS} weight sets, us a launch [min-max], "
          f"ceilings read {READ_TBS} TB/s write {WRITE_TBS} TB/s launch {LAUNCH_US} us")  # fmt: skip
    print(f"{'rows x qlen':>12} {'us':>8} {'min-max':>14} {'GB/s':>8} {'floor us':>9} {'% of floor':>11}")
    for spec in args.shapes:
        S, L = map(int, spec.split("x"))
        res = replay_us(S, L, args.nb)
        us = statistics.median(res)
        rd, wr = traffic(S, L, args.nb)
        floor = rd / (READ_TBS * 1e6) + wr / (WRITE_TBS * 1e6) + LAUNCH_US
        print(f"{spec:>12} {us:8.2f} {min(res):6.2f}-{max(res):6.2f} {(rd + wr) / us / 1e3:8.0f} {floor:9.2f} "
              f"{100 * floor / us:10.1f}%")  # fmt: skip


if __name__ == "__main__":
    main()
