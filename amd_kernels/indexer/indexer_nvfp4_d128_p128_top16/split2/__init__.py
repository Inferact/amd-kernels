# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode indexer split over two GPUs (TP2, gfx950) on the replicated index cache: rank r scores all four heads
over its blocks b % 2 == r, the ranks exchange each head's partial top 16 and merge (``host.IndexSplit2``)."""
