# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay latency for KDA verify and gated RMSNorm, optionally cycling checkpoint slots between calls."""

import argparse
import math
import statistics

import torch

from amd_kernels.attention.kda_common import kda_verify_rmsnorm


def sample(batch, window, state):
    h, d = state.shape[1:3]
    rows = batch * window
    dev = state.device

    def rand(*shape):
        return torch.randn(shape, device=dev, dtype=torch.bfloat16)

    proj = rand(rows, 5 * h * d + 16)
    return dict(
        qkv=proj[:, : 3 * h * d],
        gate=rand(rows, h, d),
        beta=proj[:, -h:],
        out_gate=proj[:, 3 * h * d : 4 * h * d].view(rows, h, d),
        alog=rand(h).float(),
        bias=rand(h * d).float(),
        state=state,
        offsets=torch.arange(batch + 1, device=dev, dtype=torch.int32) * window,
        slots=torch.arange(1, batch + 1, device=dev, dtype=torch.int32),
        weight=rand(d),
        correction=torch.empty(batch, h, window, d, device=dev),
        key_gate=rand(batch, h, window, 2 * d),
        out=rand(rows, h, d),
        lower_bound=-5.0,
        eps=1e-5,
    )


def vllm_baseline():
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMRecords, kda_recoverssm_verify
    from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated

    def verify(
        qkv,
        gate,
        beta,
        out_gate,
        alog,
        bias,
        state,
        offsets,
        slots,
        weight,
        correction,
        key_gate,
        out,
        lower_bound,
        eps,
    ):
        q, k, v = [x.unsqueeze(0) for x in qkv.view(-1, 3, 12, 128).unbind(1)]
        x = kda_recoverssm_verify(
            q,
            k,
            v,
            gate[None],
            beta[None],
            alog,
            bias,
            lower_bound,
            state,
            KDARecoverSSMRecords(correction, key_gate),
            offsets,
            slots,
            correction.shape[2],
            out=out[None],
        )
        out.copy_(rms_norm_gated(x, out_gate, weight, None, eps=eps, activation="sigmoid").view_as(out))

    return verify


def capture(fn, calls):
    fn(**calls[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for args in calls:
            fn(**args)
    graph.replay()
    return graph


def time_us(graph, ops_per_graph):
    repeats = max(1, math.ceil(200 / ops_per_graph))
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (repeats * ops_per_graph)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, choices=range(1, 9), default=5)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 96])
    parser.add_argument("--checkpoint-mib", type=int, default=0, help="minimum distinct checkpoint MiB per graph")
    parser.add_argument("--compare-vllm", action="store_true", help="requires vLLM with KDA RecoverSSM verify")
    args = parser.parse_args()
    torch.manual_seed(42)
    for batch in args.batches:
        if batch <= 0 or args.checkpoint_mib < 0:
            parser.error("batches must be positive and checkpoint MiB nonnegative")
        slot_bytes = 12 * 128 * 128 * 4
        count = max(1, math.ceil(args.checkpoint_mib * 2**20 / (batch * slot_bytes)))
        state = torch.randn(1 + count * batch, 12, 128, 128, device="cuda")
        a = sample(batch, args.window, state)
        calls = [dict(a, slots=a["slots"] + i * batch) for i in range(count)]
        if count == 1:
            calls *= 10
        ops = {"fused": kda_verify_rmsnorm}
        if args.compare_vllm:
            ops["vllm"] = vllm_baseline()
        graphs = [(name, capture(fn, calls)) for name, fn in ops.items()]
        timings = {name: [] for name in ops}
        for i in range(9):
            for name, graph in graphs if i % 2 == 0 else reversed(graphs):
                timings[name].append(time_us(graph, len(calls)))
        values = " ".join(f"{n}={statistics.median(v):.2f}us" for n, v in timings.items())
        print(f"batch={batch} window={args.window} checkpoints={count * batch} {values}", flush=True)


if __name__ == "__main__":
    main()
