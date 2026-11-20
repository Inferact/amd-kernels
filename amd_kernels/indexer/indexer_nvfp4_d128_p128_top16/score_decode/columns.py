# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MFMA column layout of the two-head decode scorers: column c holds query row c // 2 of index head c % 2 (Q contiguous
``[rows, 2, 128]``: column c at c * 128 B from the request's first row), scores ``[2, reqs * QL, S]`` contiguous over
(head, row)."""


def columns(ql: int):
    """(columns per request, query row of a column, score index of (column, request, requests, row stride, block))."""
    return (
        2 * ql,
        lambda c: c >> 1,
        lambda c, req, reqs, stride, b: (c & 1) * (reqs * ql * stride) + (req * ql + (c >> 1)) * stride + b,
    )
