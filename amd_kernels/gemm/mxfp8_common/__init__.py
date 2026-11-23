# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP8 GEMM entries: the dense linears and their dispatcher (``gemm``, ``gemm._ROUTES``; shapes with kernels of
their own live in ``gemm/mxfp8_n<N>_k<K>``), gate_up + SwiGLU + MXFP8 (``gate_up_swiglu``), the mHC post-mix GEMMs
(``postmix``), the grouped GEMM (``grouped``) and the load-time binding of a weight (``prepared``)."""

from .gate_up_swiglu import (mxfp8_gate_up_swiglu_mxfp8, mxfp8_gate_up_swiglu_mxfp8_blocker,
                             mxfp8_gate_up_swiglu_mxfp8_prequant, mxfp8_gate_up_swiglu_mxfp8_prequant_blocker,
                             mxfp8_gate_up_swiglu_mxfp8_shapes, prepare_mxfp8_gate_up_swiglu_mxfp8)
from .gemm import (mxfp8_gemm, mxfp8_gemm_blocker, mxfp8_linear, mxfp8_linear_blocker, preshuffle_mxfp8_weight,
                   preshuffle_mxfp8_weight_, tuned_shapes, warmup_mxfp8_gemm_weight)
from .grouped import (mxfp8_grouped_gemm, mxfp8_grouped_gemm_blocker, mxfp8_grouped_gemm_mxfp8,
                      mxfp8_grouped_gemm_mxfp8_blocker, mxfp8_grouped_gemm_shapes, prepare_mxfp8_grouped_gemm,
                      warmup_mxfp8_grouped_gemm)
from .postmix import (mxfp8_postmix_blocker, mxfp8_postmix_gates_gemm, mxfp8_postmix_gates_gemm_blocker,
                      mxfp8_postmix_gemm, mxfp8_postmix_gemm_blocker, mxfp8_postmix_pack_gate_weight,
                      mxfp8_postmix_preshuffle_weight, mxfp8_postmix_shapes, prepare_mxfp8_postmix_gemm,
                      warmup_mxfp8_postmix_gemm)
from .prepared import prepare_mxfp8_linear


def mxfp8_linear_shapes() -> frozenset:
    """(N, K) of the weights with tuned bf16 rows."""
    return tuned_shapes()


__all__ = [
    "mxfp8_gate_up_swiglu_mxfp8",
    "mxfp8_gate_up_swiglu_mxfp8_blocker",
    "mxfp8_gate_up_swiglu_mxfp8_prequant",
    "mxfp8_gate_up_swiglu_mxfp8_prequant_blocker",
    "mxfp8_gate_up_swiglu_mxfp8_shapes",
    "mxfp8_gemm",
    "mxfp8_gemm_blocker",
    "mxfp8_grouped_gemm",
    "mxfp8_grouped_gemm_blocker",
    "mxfp8_grouped_gemm_mxfp8",
    "mxfp8_grouped_gemm_mxfp8_blocker",
    "mxfp8_grouped_gemm_shapes",
    "mxfp8_linear",
    "mxfp8_linear_blocker",
    "mxfp8_linear_shapes",
    "mxfp8_postmix_gates_gemm",
    "mxfp8_postmix_gates_gemm_blocker",
    "mxfp8_postmix_gemm",
    "mxfp8_postmix_gemm_blocker",
    "mxfp8_postmix_pack_gate_weight",
    "mxfp8_postmix_preshuffle_weight",
    "mxfp8_postmix_shapes",
    "preshuffle_mxfp8_weight",
    "preshuffle_mxfp8_weight_",
    "prepare_mxfp8_gate_up_swiglu_mxfp8",
    "prepare_mxfp8_grouped_gemm",
    "prepare_mxfp8_linear",
    "prepare_mxfp8_postmix_gemm",
    "tuned_shapes",
    "warmup_mxfp8_gemm_weight",
    "warmup_mxfp8_grouped_gemm",
    "warmup_mxfp8_postmix_gemm",
]
