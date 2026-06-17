# Fine-Grained Offload 具体示例: mlp_norm

## 示例场景

以 Transformer Layer 中的 `mlp_norm` group 为例，说明完整的 offload/reload 流程。

## Forward 阶段

### 1. 代码位置

```python
# transformer_layer.py:676
with off_interface(self.offload_mlp_norm, hidden_states, "mlp_norm") as hidden_states:
    pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
```

### 2. 详细流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: group_start("mlp_norm")                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1.1 off_interface.__enter__()                                  │
│       └─► fine_grained_offloading_group_start()                 │
│             └─► ChunkOffloadHandler.on_group_start_forward()    │
│                   ├─► 创建 OffloadTensorGroup(name="mlp_norm") │
│                   ├─► _offloaded_group_index += 1                │
│                   └─► _groups_to_offload.append(group)           │
│                                                                  │
│  1.2 返回 hidden_states（identity 操作）                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 2: compute["mlp_norm"]                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2.1 pre_mlp_layernorm(hidden_states)                           │
│       ├─► PyTorch 自动调用 on_save_for_backward(hidden_states) │
│       │       └─► tensor_push()                                  │
│       │             ├─► 生成 tag: (group_id, tensor_idx)         │
│       │             └─► group._tensors[tag] = hidden_states     │
│       │                                                          │
│       └─► 计算 LayerNorm，返回 norm_output                       │
│                                                                  │
│  2.2 LayerNorm 的输入 hidden_states 被保存到 group               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 3: MLP 计算                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  pre_mlp_layernorm_output ──► MLP (fc1 → act → fc2)            │
│       │                                                          │
│       └─► mlp_output                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 4: BDA (Bias-Dropout-Add)                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  hidden_states = mlp_bda(mlp_output, residual)                  │
│       │                                                          │
│       └─► residual 在这里被使用（来自 mlp_norm 的输入）          │
│                                                                  │
│  注意: residual 就是原始的 hidden_states，现在才最后一次使用     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 5: group_commit("mlp_norm")                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  5.1 off_interface.group_commit()                               │
│       └─► FineGrainedOffloadingGroupCommitFunction.apply()      │
│             └─► on_group_commit_forward()                       │
│                   ├─► d2h_stream.wait_stream(compute_stream)    │
│                   ├─► bulk_offload()                            │
│                   │     └─► bulk_offload_group()                │
│                   │           ├─► 在 d2h_stream 上异步拷贝     │
│                   │           │   hidden_states → CPU          │
│                   │           └─► record_offload_event()       │
│                   └─► forced_released_tensors=[residual]       │
│                         └─► residual.untyped_storage().resize_(0)
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Offload 后的状态

```python
# OffloadTensorGroup._tensors 变成:
{
    (group_id, 0): (                      # 原本是 hidden_states tensor
        torch.device('cuda:0'),           # device
        cpu_backup,                       # CPU pinned memory tensor
        use_cpu_pool                      # True/False
    )
}
```

## Backward 阶段

### 1. 场景假设

假设 backward 传播到当前 layer，正在计算 MLP 部分的梯度：

```
传播顺序: ... → mlp_bda → mlp_fc2 → mlp_act → mlp_fc1 → mlp_norm → attn_proj → ...
                                    ▲
                                    │
                               当前在这里
```

### 2. 详细流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: mlp_fc1 backward 完成                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  mlp_fc1.backward() 计算出 grad_input                            │
│       │                                                          │
│       └─► 这个梯度需要传给 pre_mlp_layernorm                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 2: start["mlp_norm"].backward() 被调用                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2.1 FineGrainedOffloadingGroupStartFunction.backward()          │
│       └─► on_group_start_backward()                               │
│             ├─► h2d_stream.wait_stream(compute_stream)           │
│             │     # 等待 mlp_fc1 计算完成                        │
│             │                                                          │
│             └─► bulk_reload()                                     │
│                   └─► bulk_reload_group()                         │
│                         ├─► 在 h2d_stream 上异步执行            │
│                         │   ├─► wait_offload_event(d2h_stream)   │
│                         │   │     # 确保 offload 已完成          │
│                         │   │                                          │
│                         │   ├─► reload(state)                       │
│                         │   │     ├─► 分配 GPU tensor            │
│                         │   │     ├─► copy CPU → GPU              │
│                         │   │     └─► free(cpu_backup)            │
│                         │   │                                          │
│                         │   └─► group.push_tensor(tag, gpu_tensor) │
│                         │                                          │
│                         └─► _reloading_group.append(group)        │
│                                                                  │
│  2.2 同时（与上面并行）                                          │
│       └─► 预加载 "attn_proj" group（如果配置）                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 3: commit["mlp_norm"].backward() 被调用                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  3.1 FineGrainedOffloadingGroupCommitFunction.backward()         │
│       └─► on_group_commit_backward("mlp_norm")                    │
│             └─► for group in _reloading_group:                   │
│                   if group._name == "mlp_norm":                  │
│                       group.wait_reload_event(compute_stream)   │
│                       _reloading_group.remove(group)              │
│                                                                  │
│  3.2 等待 mlp_norm 的 reload 完成（同步点）                      │
│       └─► 确保 hidden_states 已在 GPU 上                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 4: compute["mlp_norm"] - LayerNorm backward               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  4.1 pre_mlp_layernorm.backward(grad_input)                     │
│       │                                                          │
│       ├─► PyTorch 需要 LayerNorm 的输入                         │
│       │     └─► on_get_saved_tensor(tag)                        │
│       │           └─► tensor_pop(tag)                           │
│       │                 ├─► 从 group._tensors[tag] 取出        │
│       │                 └─► 返回已 reload 的 hidden_states       │
│       │                                                          │
│       └─► 使用 hidden_states 计算 LayerNorm 的梯度              │
│             ├─► dL/dx = dL/dy * gamma / sqrt(var + eps)        │
│             └─► 需要 input 的 mean/variance                     │
│                                                                  │
│  4.2 hidden_states 的梯度计算完成                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. 关键问题解答

**Q1: mlp_norm offload 的是什么？**
- **输入**: `pre_mlp_layernorm` 的输入 `hidden_states`
- **为什么保存**: LayerNorm backward 需要输入来计算 mean/variance 的梯度

**Q2: reload 时加载的是什么？**
- 同样的 `hidden_states` tensor
- 从 CPU pinned memory 异步拷贝回 GPU

**Q3: reload 前完成的是什么任务？**
- MLP 部分所有 layer 的 backward (fc2, activation, fc1)
- `mlp_fc1.backward()` 刚刚完成

**Q4: loaded 的参数什么时候需要？**
- `commit["mlp_norm"].backward()` 时立即需要（等待完成）
- `compute["mlp_norm"]` 时立即需要（用于 LayerNorm gradient 计算）

**Q5: mlp_norm 的输入（residual）为什么在 BDA 后才释放？**
- 因为 `mlp_bda(mlp_output, residual)` 需要 residual
- residual 就是原始的 `hidden_states`
- 延迟到 BDA 后才 `group_commit`，此时才触发 offload 和释放

## 数据流向图

```
Forward:

  hidden_states (residual)
       │
       ├──────────────────────────────────────────────────┐
       │                                                  │
       │  ┌────────────────────────────────────────────┐ │
       │  │ mlp_norm group                              │ │
       │  │                                             │ │
       │  │ 1. start: 创建 group                       │ │
       │  │                                             │ │
       │  │ 2. compute: pre_mlp_layernorm               │ │
       │  │    ├─► on_save_for_backward(hidden_states)  │ │
       │  │    └─► group._tensors[tag] = hidden_states │ │
       │  │                                             │ │
       │  │ 3. commit: 在 mlp_bda 后                   │ │
       │  │    ├─► bulk_offload(hidden_states)        │ │
       │  │    │     GPU ──► CPU                       │ │
       │  │    │                                          │ │
       │  │    └─► group._tensors[tag] = state         │ │
       │  │          (device, cpu_backup, use_pool)   │ │
       │  │                                             │ │
       │  └────────────────────────────────────────────┘ │
       │                                                  │
       ▼                                                  │
  pre_mlp_layernorm_output ──► MLP (fc1 → act → fc2)    │
       │                                                  │
       │                                                  │
       ▼                                                  │
  mlp_output ──► mlp_bda(residual) ◄────────────────────┘
       │
       ▼
  next_layer


Backward:

  grad_from_next_layer
       │
       ▼
  mlp_bda.backward()
       │
       ▼
  mlp_fc2.backward()
       │
       ▼
  mlp_act.backward()
       │
       ▼
  mlp_fc1.backward() ◄── 刚刚完成！
       │
       ▼
  start["mlp_norm"].backward()
       │
       └─► bulk_reload()
             ├─► 在 h2d_stream 上异步执行
             │   ├─► wait_offload_event(d2h_stream)
             │   │
             │   ├─► reload(state)
             │   │     ├─► gpu_tensor = torch.empty(...)
             │   │     ├─► gpu_tensor.copy_(cpu_backup)
             │   │     └─► free(cpu_backup)
             │   │
             │   └─► group._tensors[tag] = gpu_tensor
             │
             └─► _reloading_group.append(group)

       │
       ▼
  commit["mlp_norm"].backward()
       │
       └─► wait_reload_event(compute_stream)
             # 确保 reload 完成

       │
       ▼
  compute["mlp_norm"]
       │
       ├─► on_get_saved_tensor(tag)
       │     └─► tensor_pop(tag)
       │           └─► 返回 reload 后的 hidden_states
       │
       └─► pre_mlp_layernorm.backward(grad)
             ├─► 使用 hidden_states 计算 mean/variance
             └─► 计算 dL/dx

       │
       ▼
  grad_to_prev_layer
```

## 关键代码片段

### Forward - 保存和 Offload

```python
# transformer_layer.py:681
with off_interface(self.offload_mlp_norm, hidden_states, "mlp_norm") as hidden_states:
    # 内部：group_start("mlp_norm")
    pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
    # PyTorch 内部：on_save_for_backward(hidden_states)
    # 保存到 group._tensors

# ... MLP 计算 ...

# transformer_layer.py:825
if self.offload_mlp_norm:
    hidden_states = off_interface.group_commit(
        hidden_states, name="mlp_norm", forced_released_tensors=[residual]
    )
```

### Backward - Reload 和使用

```python
# 在 FineGrainedOffloadingGroupStartFunction.backward
@staticmethod
def backward(ctx, grad_output):
    ctx.cpu_offload_handler.on_group_start_backward()  # 触发 bulk_reload
    return grad_output

# 在 FineGrainedOffloadingGroupCommitFunction.backward
@staticmethod
def backward(ctx, *grad_output):
    ctx.cpu_offload_handler.on_group_commit_backward(ctx.name)  # 等待 reload
    return grad_output

# 在 on_get_saved_tensor
def on_get_saved_tensor(self, saved_state):
    return self.cur_backward_chunk().tensor_pop(saved_state)  # 返回 reload 后的 tensor
```
