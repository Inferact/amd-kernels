# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gated Engram update of the four residual streams: gate from the ``wkv`` keys, add the shared value."""
from .host import gated_post
from .reference import gated_post_reference

__all__ = ["gated_post", "gated_post_reference"]
