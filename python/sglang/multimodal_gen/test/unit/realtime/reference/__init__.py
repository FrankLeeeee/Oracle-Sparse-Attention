# SPDX-License-Identifier: Apache-2.0
"""Upstream reference implementations, copied verbatim, for parity tests.

Each module here holds functions lifted unchanged from the paper authors' own
repositories, with the commit they came from recorded in the module docstring.
Tests compare our implementations against these rather than against a
paraphrase, so "we match upstream" is a claim the test suite actually checks.

Upstream cannot simply be imported: the real modules pull in ``sageattention``,
``spas_sage_attn``, ``block_sparse_attn`` and ``cuvs``, which are CUDA
extensions not installed here. Only the top-level import blocks were dropped.

| module | upstream | commit |
|---|---|---|
| ``radial_reference`` | mit-han-lab/radial-attention | 72788d4 |
| ``xattention_reference`` | mit-han-lab/x-attention | e379887 |
| ``svg_reference`` | svg-project/Sparse-VideoGen | f89aeda |
"""
