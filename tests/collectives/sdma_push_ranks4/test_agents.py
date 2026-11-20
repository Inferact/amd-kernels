# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The HSA agent of every rank is that rank's GPU: keys unique over every GPU agent of the node (two PCI domains can
share bus / device numbers), checked against the agent UUID, which torch reports as the ASCII bytes of its suffix."""

import ctypes

import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_4_gpus

HSA_AMD_AGENT_INFO_UUID = 0xA011


def _agent_uuid(agent) -> str:
    from amd_kernels.collectives.sdma_push_ranks4 import hsa_runtime as rt

    buf = ctypes.create_string_buffer(64)
    assert rt.hsa().hsa_agent_get_info(agent, HSA_AMD_AGENT_INFO_UUID, buf) == 0
    return buf.value.decode()


def _device_uuid(device: torch.device) -> str:
    return "GPU-" + bytes.fromhex(str(torch.cuda.get_device_properties(device).uuid).replace("-", "")).decode()


def _all_gpu_agent_uuids() -> list[str]:
    from amd_kernels.collectives.sdma_push_ranks4 import hsa_runtime as rt

    lib, out = rt.hsa(), []

    def visit(agent, _):
        kind = ctypes.c_uint32()
        lib.hsa_agent_get_info(agent, rt.HSA_AGENT_INFO_DEVICE, ctypes.byref(kind))
        if kind.value == rt.HSA_DEVICE_TYPE_GPU:
            out.append(_agent_uuid(rt.Agent(agent.handle)))
        return 0

    assert lib.hsa_iterate_agents(lib._visit_type(visit), None) == 0
    return out


def test_agent_keys_unique_and_match_devices():
    from amd_kernels.collectives.sdma_push_ranks4 import hsa_runtime as rt

    agents = rt.gpu_agents()
    assert sorted(_agent_uuid(a) for a in agents.values()) == sorted(_all_gpu_agent_uuids())
    devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    keys = [rt.pci_key(d) for d in devices]
    assert len(set(keys)) == len(keys)
    for d, k in zip(devices, keys):
        assert _agent_uuid(agents[k]) == _device_uuid(d), (d, hex(k))


def _worker(rank, errors):
    from amd_kernels.collectives.sdma_push_ranks4 import WORLD, SdmaPush

    dev = torch.device("cuda", rank)
    t = SdmaPush(rank, dist.group.WORLD, dev, [4096])
    want = [None] * WORLD
    dist.all_gather_object(want, _device_uuid(dev))
    got = [_agent_uuid(a) for a in t.agents]
    if got != want:
        errors.append(f"rank {rank}: agents {got}, devices {want}")


def test_transport_agents_are_the_ranks_gpus():
    run_on_4_gpus(_worker, timeout=300)
