# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Attention op entries: norm / RoPE / KV insert, MLA and sparse MLA decode, KDA, SWA decode, NVFP4 sparse attention,
compressor insert. Callers import ops from here only (``amd_kernels/README.md``, "Op entries"); each family's
``__all__`` is re-exported."""

from amd_kernels.routes import available

from . import (  # noqa: F401
    compressor_insert_common,
    gemma_norm_rope_insert_common,
    kda_common,
    mla_decode_common,
    norm_rope_insert_common,
    norm_rope_nvfp4_insert_common,
    nvfp4_sparse_attention_common,
    rope_fp8_kv_insert_common,
    sparse_mla_decode_common,
    swa_decode_common,
)
from .compressor_insert_common import *  # noqa: F403
from .gemma_norm_rope_insert_common import *  # noqa: F403
from .kda_common import *  # noqa: F403
from .mla_decode_common import *  # noqa: F403
from .norm_rope_insert_common import *  # noqa: F403
from .norm_rope_nvfp4_insert_common import *  # noqa: F403
from .nvfp4_sparse_attention_common import *  # noqa: F403
from .rope_fp8_kv_insert_common import *  # noqa: F403
from .sparse_mla_decode_common import *  # noqa: F403
from .swa_decode_common import *  # noqa: F403

__all__ = sorted({
    "available",
    *compressor_insert_common.__all__,
    *gemma_norm_rope_insert_common.__all__,
    *kda_common.__all__,
    *mla_decode_common.__all__,
    *norm_rope_insert_common.__all__,
    *norm_rope_nvfp4_insert_common.__all__,
    *nvfp4_sparse_attention_common.__all__,
    *rope_fp8_kv_insert_common.__all__,
    *sparse_mla_decode_common.__all__,
    *swa_decode_common.__all__,
})
