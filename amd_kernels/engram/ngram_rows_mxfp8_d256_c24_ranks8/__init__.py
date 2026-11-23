# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""N-gram hash and Engram row pull of two layers over eight ranks, as the MXFP8 input of the ``wkv`` GEMM."""
from .host import NgramRows, available
from .reference import ngram_rows_reference

__all__ = ["NgramRows", "available", "ngram_rows_reference"]
