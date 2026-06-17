# Fine-Grained Offload NVTX 标记增强方案

## 1. 当前已存在的 NVTX 标记

### 1.1 Offload/Reload 标记

```python
# fine_grained_activation_offload.py:876
def bulk_offload_group(self):
    group_to_offload = self._groups_to_offload[-1]
    torch.cuda.nvtx.range_push("activation offloading " + group_to_offload._name)
    with torch.cuda.stream(self.d2h_stream):
        # ... offload logic ...
    torch.cuda.nvtx.range_pop()

# fine_grained_activation_offload.py:903
def bulk_reload_group(self):
    group_to_reload = self._groups_to_reload[-1]
    torch.cuda.nvtx.range_push("activation reloading " + group_to_reload._name)
    with torch.cuda.stream(self.h2d_stream):
        # ... reload logic ...
    torch.cuda.nvtx.range_pop()
```

### 1.2 Transformer Layer 标记

```python
# transformer_layer.py
nvtx_range_push(suffix="input_layernorm")
nvtx_range_pop(suffix="input_layernorm")

nvtx_range_push(suffix="self_attention")
nvtx_range_pop(suffix="self_attention")

nvtx_range_push(suffix="self_attn_bda")
nvtx_range_pop(suffix="self_attn_bda")

nvtx_range_push(suffix="mlp")
nvtx_range_pop(suffix="mlp")

nvtx_range_push(suffix="mlp_bda")
nvtx_range_pop(suffix="mlp_bda")
```

## 2. 推荐的 NVTX 增强方案

### 2.1 Tensor Push/Pop 级别（Debug 用）

```python
def tensor_push(self, tensor):
    torch.cuda.nvtx.range_push(
        f"offload_push[g:{self._offloaded_group_index}, "
        f"shape:{list(tensor.shape)}, dtype:{tensor.dtype}]"
    )
    # ... original logic ...
    torch.cuda.nvtx.range_pop()

def tensor_pop(self, tensor_tag):
    group_id, idx = tensor_tag
    torch.cuda.nvtx.range_push(f"offload_pop[g:{group_id}, idx:{idx}]")
    # ... original logic ...
    torch.cuda.nvtx.range_pop()
```

### 2.2 Group 级别增强

```python
def bulk_offload_group(self):
    group_to_offload = self._groups_to_offload[-1]
    tensor_count = len(group_to_offload._tensors)
    total_size_mb = sum(
        t.numel() * t.element_size() 
        for t in group_to_offload._tensors.values()
    ) / (1024**2)
    
    torch.cuda.nvtx.range_push(
        f"activation offload[{group_to_offload._name}], "
        f"tensors:{tensor_count}, size:{total_size_mb:.2f}MB"
    )
    # ... original logic ...
    torch.cuda.nvtx.range_pop()

def bulk_reload_group(self):
    group_to_reload = self._groups_to_reload[-1]
    torch.cuda.nvtx.range_push(
        f"activation reload[{group_to_reload._name}], "
        f"stream:h2d, is_warmup:{self.is_warmup}"
    )
    # ... original logic ...
    torch.cuda.nvtx.range_pop()
```

### 2.3 Group Start/Commit 边界标记

```python
def on_group_start_forward(self, name):
    torch.cuda.nvtx.range_push(
        f"offload_group_start[{name}], idx:{self._offloaded_group_index}"
    )
    # ... original logic ...
    torch.cuda.nvtx.range_pop()

def on_group_commit_forward(self, forced_released_tensors):
    torch.cuda.nvtx.range_push(
        f"offload_group_commit[{self._groups_to_offload[-1]._name}], "
        f"forced_release:{len(forced_released_tensors)}"
    )
    # ... original logic ...
    torch.cuda.nvtx.range_pop()

def on_group_start_backward(self):
    torch.cuda.nvtx.range_push(
        f"offload_group_start_backward, "
        f"groups_to_reload:{len(self._groups_to_reload)}"
    )
    # ... original logic ...
    torch.cuda.nvtx.range_pop()

def on_group_commit_backward(self, name):
    torch.cuda.nvtx.range_push(
        f"offload_group_commit_backward[{name}]"
    )
    # ... original logic ...
    torch.cuda.nvtx.range_pop()
```

### 2.4 Stream 同步标记

```python
def on_group_start_backward(self):
    torch.cuda.nvtx.range_push("offload_wait_compute_stream")
    self.h2d_stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push("offload_bulk_reload")
    self.bulk_reload()
    torch.cuda.nvtx.range_pop()

def bulk_reload_group(self):
    with torch.cuda.stream(self.h2d_stream):
        torch.cuda.nvtx.range_push("offload_wait_offload_event")
        group_to_reload.wait_offload_event(self.h2d_stream)
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push("offload_copy_h2d")
        for tensor_tag, state in group_to_reload._tensors.items():
            if isinstance(state, tuple):
                recovered_tensor = self.reload(state)
        torch.cuda.nvtx.range_pop()
```

## 3. 完整的 NVTX 层次结构

使用增强后的 NVTX 标记，在 Nsight Systems 中可以看到清晰的层次结构：

```
├── input_layernorm
│   ├── offload_group_start[attn_norm]
│   │   ├── offload_push[g:1, shape:[1024,4,4096], dtype:torch.float32]
│   │   └── offload_push[g:1, shape:[1024,4,4096], dtype:torch.float32]
│   ├── offload_group_commit[attn_norm]
│   │   └── offload_forced_release[1]
│   └── activation offload[attn_norm]
│       ├── offload_wait_compute_stream
│       ├── offload_copy_d2h (async on d2h_stream)
│       └── offload_record_event
│
├── self_attention
│   ├── offload_group_start[qkv_linear]
│   │   └── ...
│   ├── offload_group_commit[qkv_linear]
│   └── activation offload[qkv_linear]
│       ├── offload_wait_compute_stream
│       └── offload_copy_d2h
│
... (继续其他 groups)

Backward:
├── mlp_bda_backward
│   └── ...
│
├── mlp_fc1_backward
│   ├── offload_group_start_backward[mlp_norm]
│   │   ├── offload_wait_compute_stream
│   │   └── offload_bulk_reload
│   │       ├── activation reload[mlp_norm]
│   │       │   ├── offload_wait_offload_event
│   │       │   ├── offload_copy_h2d
│   │       │   └── offload_record_reload_event
│   │       └── activation reload[attn_proj] (pre-fetch)
│   │           └── ...
│   └── offload_group_commit_backward[mlp_norm]
│       └── offload_wait_reload_event
│
├── pre_mlp_layernorm_backward
│   ├── offload_pop[g:5, idx:0]
│   └── ...
```

## 4. 性能分析关键指标

通过增强的 NVTX 标记，可以分析以下关键指标：

### 4.1 Offload 效率

| 指标 | 计算方法 | 目标值 |
|------|---------|-------|
| offload 时间占比 | offload_time / forward_time | < 10% |
| 异步效率 | (async_copy_time - sync_wait_time) / async_copy_time | > 90% |
| 内存节省 | offloaded_size / total_activation_size | > 50% |

### 4.2 Reload 效率

| 指标 | 计算方法 | 目标值 |
|------|---------|-------|
| reload 隐藏率 | reload_time_overlapped / reload_time_total | > 80% |
| 预加载命中率 | tensors_preloaded / tensors_total | > 90% |
| 同步等待时间 | wait_reload_event_time / backward_time | < 5% |

### 4.3 Group 级别分析

```python
# 在 Nsight Systems 中可以观察到:
for group in ["attn_norm", "qkv_linear", "core_attn", "attn_proj", "mlp_norm"]:
    forward_time = measure(f"activation offload[{group}]")
    backward_time = measure(f"activation reload[{group}]")
    tensor_count = get_nvtx_attr(f"offload[{group}]", "tensors")
    tensor_size = get_nvtx_attr(f"offload[{group}]", "size")
    
    efficiency = tensor_size / forward_time  # MB/ms
    print(f"{group}: {tensor_count} tensors, {tensor_size}MB, {efficiency:.2f}MB/ms")
```

## 5. 实现建议

### 5.1 条件编译

```python
# 只在 DEBUG 或 PROFILE 模式下启用详细标记
DEBUG_NVTX = os.environ.get("MEGATRON_DEBUG_NVTX", "0") == "1"

def tensor_push(self, tensor):
    if DEBUG_NVTX:
        torch.cuda.nvtx.range_push(f"offload_push[...]")
    # ... original logic ...
    if DEBUG_NVTX:
        torch.cuda.nvtx.range_pop()
```

### 5.2 层次化标记

```python
# Level 0: 仅 Group 级别（生产环境）
# Level 1: Group + Tensor 数量/大小
# Level 2: Group + 每个 Tensor 的 shape/dtype（Debug 环境）
NVTX_LEVEL = int(os.environ.get("MEGATRON_NVTX_LEVEL", "1"))

def bulk_offload_group(self):
    if NVTX_LEVEL >= 1:
        torch.cuda.nvtx.range_push(
            f"activation offload[{group._name}], "
            f"tensors:{len(group._tensors)}, "
            f"size:{total_size_mb:.2f}MB"
        )
    # ...
```

## 6. 总结

推荐的 NVTX 增强方案：

| 层级 | 标记位置 | 信息内容 | 用途 |
|------|---------|---------|------|
| **Layer** | Transformer 各子模块 | 已有 | 性能分析 |
| **Group** | `bulk_offload/reload_group` | group 名称、tensor 数量、数据量 | offload/reload 效率分析 |
| **Boundary** | `on_group_start/commit_forward/backward` | group 索引、warmup 状态 | 生命周期分析 |
| **Sync** | `wait_stream`, `wait_event` | 等待类型 | 同步开销分析 |
| **Tensor** (Debug) | `tensor_push/pop` | tag、shape、dtype | 细粒度调试 |

通过这些标记，可以在 Nsight Systems 中清晰地看到：
1. 每个 group 的 offload/reload 时间
2. 计算和通信的重叠情况
3. 内存使用模式
4. 潜在的瓶颈点
