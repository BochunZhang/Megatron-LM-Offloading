# Fine-Grained Activation Offloading 实现原理分析

## 1. 核心架构

Fine-grained offload 采用三层管理结构：

| 组件 | 作用 | 关键文件 |
|------|------|---------|
| **PipelineOffloadManager** | 单例模式，管理整个 pipeline 的 offload 流程，协调多个 chunk | `fine_grained_activation_offload.py` |
| **ChunkOffloadHandler** | 管理每个 microbatch (chunk) 的 offload，包含多个 group | `fine_grained_activation_offload.py:727` |
| **OffloadTensorGroup** | 单个 offload group，存储 tensors 和 CUDA events，不负责具体逻辑 | `fine_grained_activation_offload.py:331` |

### 1.1 PipelineOffloadManager 职责
- 注册 autograd hooks (`on_save_for_backward`, `on_get_saved_tensor`)
- 管理多个 ChunkOffloadHandler (每个 microbatch 一个)
- 协调 chunk 切换和虚拟 pipeline 并行
- 维护 d2h_stream 和 h2d_stream 用于异步传输

### 1.2 ChunkOffloadHandler 职责
- 管理单个 microbatch 内所有 tensor groups
- 执行实际的 offload/reload 操作
- 控制 CUDA stream 同步和内存池
- 每个 microbatch 独立生命周期

### 1.3 OffloadTensorGroup 结构
```python
class OffloadTensorGroup:
    def __init__(self, name):
        self._name = name              # group 名称
        self._tensors = {}             # tensor_tag -> tensor 映射
        self._offload_event = torch.cuda.Event()
        self._reload_event = torch.cuda.Event()
        self.offload = True            # 是否允许 offload
        self.use_cpu_pool = True       # 是否使用 CPU tensor pool
```

## 2. 支持的 Offload Groups

| Group Name | 位置 | Offload 内容 | 配置参数 |
|------------|------|-------------|---------|
| `attn_norm` | TransformerLayer | Attention 前 layer norm 的输入 | `offload_modules` |
| `qkv_linear` | Attention | QKV linear 层的输入 | `offload_modules` |
| `core_attn` | Attention | Core attention 的输入 | `offload_modules` |
| `attn_proj` | Attention | Attention 投影层的输入 | `offload_modules` |
| `mlp_norm` | TransformerLayer | MLP 前 layer norm 的输入 | `offload_modules` |
| `expert_fc1` | GroupedMLP (MoE) | Expert fc1 的输入 | `offload_modules` |
| `moe_act` | GroupedMLP (MoE) | MoE 激活函数的输入 | `offload_modules` |

## 3. 核心 Autograd Function

### 3.1 FineGrainedOffloadingGroupStartFunction

```python
class FineGrainedOffloadingGroupStartFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler, name):
        ctx.cpu_offload_handler = cpu_offload_handler
        cpu_offload_handler.on_group_start_forward(name)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_start_backward()  # 触发预加载
        return grad_output
```

**作用**:
- Forward: 标记 group 开始，创建 OffloadTensorGroup
- Backward: 触发 `bulk_reload()` 预加载**前一个** group

### 3.2 FineGrainedOffloadingGroupCommitFunction

```python
class FineGrainedOffloadingGroupCommitFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, cur_forward_chunk, name, forced_released_tensors, delay_offload):
        ctx.cpu_offload_handler = cur_forward_chunk
        ctx.name = name
        cur_forward_chunk.on_group_commit_forward(forced_released_tensors)
        return tensor

    @staticmethod
    def backward(ctx, *grad_output):
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward(ctx.name)  # 等待 reload
        return grad_output
```

**作用**:
- Forward: 触发 `bulk_offload()` 卸载 tensors
- Backward: 等待当前 group 的 reload 完成（同步点）

## 4. Forward 执行流程

执行顺序: `start[i] → compute[i] → commit[i] → start[i+1] → compute[i+1]`

```python
# 示例: attn_norm
with off_interface(self.offload_attn_norm, hidden_states, "attn_norm") as hidden_states:
    # 1. group_start("attn_norm") - 准备接收 tensors
    input_layernorm_output = self.input_layernorm(hidden_states)
    # 2. 计算过程
    # 3. PyTorch autograd 自动调用 on_save_for_backward，保存 tensor tag

# ... attention 计算 ...

# 4. group_commit("attn_norm") - 触发 offload
hidden_states = off_interface.group_commit(
    hidden_states, name="attn_norm", forced_released_tensors=[residual]
)
```

**关键点**:
- `commit[i]` 在 tensor[i] 最后一次使用后触发
- 延迟释放确保 tensor 在后续计算中仍可用（如 residual 在 BDA 中需要）
- offload 是异步的，不阻塞 compute stream

## 5. Backward 执行流程

执行顺序（与 Forward 相反）:
```
... → commit[1].backward → compute[1] → start[1].backward → commit[0].backward → compute[0] → start[0].backward
```

### 5.1 详细流程

```
Layer N backward:
  │
  ├─► commit[N].backward()
  │       └─► on_group_commit_backward("N")
  │               └─► 等待 reload[N] 完成（同步点）
  │
  ├─► compute[N] - 计算 Layer N 的梯度
  │       └─► 需要 tensor[N] 时调用 on_get_saved_tensor()
  │               └─► tensor_pop() 返回已 reload 的 tensor
  │
  └─► start[N].backward()
          └─► on_group_start_backward()
                  ├─► h2d_stream.wait_stream(compute_stream)  # 等待计算完成
                  └─► bulk_reload() - 预加载 Layer N-1

Layer N-1 backward:
  │
  ├─► commit[N-1].backward()
  │       └─► 等待预加载的 reload[N-1] 完成
  │
  ├─► compute[N-1] - 计算 Layer N-1 的梯度
  │       └─► （与 reload[N-2] 并行执行！）
  │
  └─► start[N-1].backward()
          └─► bulk_reload() - 预加载 Layer N-2
```

### 5.2 关键设计: 计算与通信重叠

```python
def on_group_start_backward(self):
    # 1. 确保 compute_stream 完成当前 layer 的计算
    #    （这样前一个 layer 的激活值就不再需要了）
    self.h2d_stream.wait_stream(torch.cuda.current_stream())
    
    # 2. 启动前一个 layer 的 reload（在 h2d_stream 上异步执行）
    self.bulk_reload()
```

**为什么需要 wait_stream**:
- 防止内存分配冲突: 确保 Layer N 的计算完成后再分配内存给 Layer N-1
- 防御性同步: 建立正确的 happens-before 关系
- **不阻塞计算**: wait_stream 只阻塞 h2d_stream，compute_stream 继续执行后续 layer 的计算

**重叠效果**:
- `compute[N-1]` 在 compute_stream 上执行
- `reload[N-2]` 在 h2d_stream 上执行
- 两个 stream **并行执行**，隐藏数据传输延迟

## 6. Hook 机制

### 6.1 注册

```python
def __enter__(self):
    torch._C._autograd._push_saved_tensors_default_hooks(
        self.on_save_for_backward,    # Forward hook
        self.on_get_saved_tensor        # Backward hook
    )
```

### 6.2 on_save_for_backward (Forward)

```python
def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
    # 返回 tag，PyTorch 保存这个 tag 而不是 tensor
    return self.cur_forward_chunk().tensor_push(tensor)

def tensor_push(self, tensor):
    tensor_tag = (self._offloaded_group_index, self._tensor_count_current_group)
    self.offload_groups[self._offloaded_group_index - 1].push_tensor(tensor_tag, tensor)
    return tensor_tag
```

### 6.3 on_get_saved_tensor (Backward)

```python
def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
    return self.cur_backward_chunk().tensor_pop(saved_state)

def tensor_pop(self, tensor_tag):
    group_id, idx = tensor_tag
    tensor = self.offload_groups[group_id - 1].pop_tensor(tensor_tag)
    if isinstance(tensor, tuple):  # 已 offload，需要 reload
        tensor = self.reload(tensor)
    return tensor
```

## 7. CUDA Stream 同步机制

### 7.1 三个 Stream

| Stream | 用途 |
|--------|------|
| `compute_stream` | 默认计算流 (autograd) |
| `d2h_stream` | GPU → CPU offload |
| `h2d_stream` | CPU → GPU reload |

### 7.2 同步点

**Forward - bulk_offload_group**:
```python
with torch.cuda.stream(self.d2h_stream):
    for tensor_tag, tensor in group._tensors.items():
        state = self.offload(tensor)  # GPU → CPU
        group.push_tensor(tensor_tag, state)
    group.record_offload_event(self.d2h_stream)
```

**Backward - bulk_reload_group**:
```python
with torch.cuda.stream(self.h2d_stream):
    # h2d_stream 等待 d2h_stream 的 offload 完成
    group_to_reload.wait_offload_event(self.h2d_stream)
    
    for tensor_tag, state in group_to_reload._tensors.items():
        recovered_tensor = self.reload(state)  # CPU → GPU
    group_to_reload.record_reload_event(self.h2d_stream)
```

**跨流同步 - on_group_start_backward**:
```python
def on_group_start_backward(self):
    # h2d_stream 等待 compute_stream
    self.h2d_stream.wait_stream(torch.cuda.current_stream())
    self.bulk_reload()
```

## 8. ChunkOffloadHandler 的创建

### 8.1 每个 Microbatch 一个 Handler

```python
def preprocess_for_fine_grained_offloading(self):
    off_interface.init_chunk_handler(
        vp_size=self.config.virtual_pipeline_model_parallel_size,
        vp_stage=self.vp_stage,
        min_offloaded_tensor_size=self.config.min_offloaded_tensor_size,
    )

# GPTModel.forward() 每次调用都会触发
```

**PP=4, VPP=2 时的结构**:
```
PipelineOffloadManager
    ├── _stages[0] (vp_stage=0)
    │       ├── ChunkOffloadHandler (microbatch 0)
    │       ├── ChunkOffloadHandler (microbatch 2)
    │       └── ...
    ├── _stages[1] (vp_stage=1)
    │       ├── ChunkOffloadHandler (microbatch 1)
    │       ├── ChunkOffloadHandler (microbatch 3)
    │       └── ...
    └── _cached_chunks_forward (所有 chunks，按创建顺序)
```

### 8.2 为什么每个 microbatch 需要独立 handler

- 在 1F1B 调度中，多个 microbatch 同时处于不同阶段
- 每个 microbatch 有自己的激活值生命周期
- 需要独立管理 offload/reload 状态

## 9. 已存在的 NVTX 标记

当前代码已包含以下 NVTX 标记:

```python
# fine_grained_activation_offload.py:876
torch.cuda.nvtx.range_push("activation offloading " + group_to_offload._name)
...
torch.cuda.nvtx.range_pop()

# fine_grained_activation_offload.py:903
torch.cuda.nvtx.range_push("activation reloading " + group_to_reload._name)
...
torch.cuda.nvtx.range_pop()

# transformer_layer.py
nvtx_range_push(suffix="input_layernorm")
nvtx_range_pop(suffix="input_layernorm")
```

## 10. 关键结论

1. **Fine-grained offload 是 module 级别的**，比 layer-level 的 cpu_offloading 更精细
2. **异步传输**: 使用独立 CUDA streams 实现计算和传输重叠
3. **预加载机制**: start.backward 预加载前一个 group，隐藏 latency
4. **Hook 机制**: 通过 PyTorch autograd hooks 自动管理 tensor 生命周期
5. **每个 microbatch 独立**: ChunkOffloadHandler 按 microbatch 隔离管理
6. **延迟释放**: commit 在 tensor 最后一次使用后触发，确保安全
