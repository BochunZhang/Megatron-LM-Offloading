# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NVTX profiling utilities for module forward and backward passes."""

from typing import Dict, List, Optional, Callable
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class NVTXModuleProfiler:
    """Profiler that adds NVTX markers to module forward and backward passes.

    This profiler registers hooks on PyTorch modules to push/pop NVTX ranges
    at the beginning and end of forward and backward passes.

    Hook execution order:
    - Pre-forward hooks: Registered with prepend=True, execute first
    - Forward: Module's forward pass
    - Post-forward hooks: Execute last

    - Pre-backward hooks: Registered with prepend=True (via move_to_end), execute first
    - Backward: Autograd backward pass
    - Post-backward hooks: Execute last

    Args:
        enabled: Whether to enable profiling. Default: True
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._handles: List[RemovableHandle] = []
        self._module_tags: Dict[int, str] = {}  # module_id -> nvtx_tag

    def register_module(
        self,
        module: nn.Module,
        nvtx_tag: Optional[str] = None,
        enable_forward: bool = True,
        enable_backward: bool = True,
    ) -> None:
        """Register NVTX markers for a module.

        Args:
            module: The module to profile
            nvtx_tag: The NVTX tag to use. If None, uses module's class name
            enable_forward: Whether to profile forward pass
            enable_backward: Whether to profile backward pass
        """
        if not self.enabled:
            return

        tag = nvtx_tag or module.__class__.__name__
        self._module_tags[id(module)] = tag

        if enable_forward:
            self._register_forward_nvtx(module, tag)

        if enable_backward:
            self._register_backward_nvtx(module, tag)

    def _register_forward_nvtx(self, module: nn.Module, tag: str) -> None:
        """Register forward hooks with proper ordering."""

        def pre_forward_hook(mod: nn.Module, input):
            """Pre-forward: push NVTX range (executes first)."""
            torch.cuda.nvtx.range_push(f"{tag}.forward")
            return None

        def post_forward_hook(mod: nn.Module, input, output):
            """Post-forward: pop NVTX range (executes last)."""
            torch.cuda.nvtx.range_pop()
            return None

        # Register pre-hook with prepend=True to execute first
        handle_pre = module.register_forward_pre_hook(pre_forward_hook, prepend=True)
        self._handles.append(handle_pre)

        # Register post-hook (executes last by default)
        handle_post = module.register_forward_hook(post_forward_hook)
        self._handles.append(handle_post)

    def _register_backward_nvtx(self, module: nn.Module, tag: str) -> None:
        """Register backward hooks with proper ordering."""

        def pre_backward_hook(mod: nn.Module, grad_output):
            """Pre-backward: push NVTX range (executes first)."""
            torch.cuda.nvtx.range_push(f"{tag}.backward")
            return None

        def post_backward_hook(mod: nn.Module, grad_input, grad_output):
            """Post-backward: pop NVTX range (executes last)."""
            torch.cuda.nvtx.range_pop()
            return None

        # Register pre-hook with prepend=True to execute first
        handle_pre = module.register_full_backward_pre_hook(pre_backward_hook, prepend=True)
        self._handles.append(handle_pre)

        # Register post-hook (executes last by default)
        handle_post = module.register_full_backward_hook(post_backward_hook)
        self._handles.append(handle_post)

    def register_model(
        self,
        model: nn.Module,
        tag_prefix: str = "",
        recurse: bool = True,
    ) -> None:
        """Register NVTX markers for an entire model.

        Args:
            model: The model to profile
            tag_prefix: Prefix for all tags (e.g., "layer_0")
            recurse: Whether to recursively register submodules
        """
        if not self.enabled:
            return

        if recurse:
            for name, module in model.named_modules():
                # Skip container modules
                if not any(
                    isinstance(module, container)
                    for container in [nn.Sequential, nn.ModuleList, nn.ModuleDict]
                ):
                    full_tag = f"{tag_prefix}.{name}" if tag_prefix else name
                    self.register_module(module, nvtx_tag=full_tag)
        else:
            tag = tag_prefix or model.__class__.__name__
            self.register_module(model, nvtx_tag=tag)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._module_tags.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()
