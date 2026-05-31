# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron Core Utils

This module provides utility functions and classes for Megatron Core.
The utils package contains submodules that can be imported directly.
"""

# Re-export from utils.py for backward compatibility
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    get_model_config,
    get_torch_version,
    is_te_min_version,
    is_torch_min_version,
    log_single_rank,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)

# NVTX profiling utilities
from megatron.core.utils.nvtx_profiler import NVTXModuleProfiler

__all__ = [
    'WrappedTensor',
    'deprecate_inference_params',
    'get_model_config',
    'get_torch_version',
    'is_te_min_version',
    'is_torch_min_version',
    'log_single_rank',
    'make_viewless_tensor',
    'nvtx_range_pop',
    'nvtx_range_push',
    'NVTXModuleProfiler',
]
