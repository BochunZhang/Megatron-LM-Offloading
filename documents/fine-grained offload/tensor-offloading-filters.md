# Fine-Grained Activation Offloading - Tensor 过滤机制

## 概述

Fine-grained activation offloading 在 Megatron-LM 中用于将激活值从 GPU 卸载到 CPU 以节省显存。然而，并非所有 tensor 都会被卸载，系统通过多种机制进行过滤。

本文档说明哪些 tensor 会被滤除（不卸载）及其原因。

---

## 过滤条件

### 1. Tensor 大小过滤

**代码位置**: `megatron/core/pipeline_parallel/fine_grained_activation_offload.py:865-866`

```python
def tensor_need_offloading_checker(self, tensor):
    if tensor.numel() < self.min_offloaded_tensor_size:
        return False  # 太小的 tensor 不被卸载
```

**配置项**: `min_offloaded_tensor_size` (默认: `1024 * 1024`，即 1M 元素)

- 只有元素数量 `>= 1M` 的 tensor 才会被考虑卸载
- 注意：这是**元素数量**，不是字节数

**示例**:

对于 attention norm 相关的 tensor：

| Tensor | Shape | 元素数 | dtype | 是否卸载 |
|--------|-------|--------|-------|----------|
| 激活值 | `[8192, 7168]` | 58,720,768 | bf16 | ✅ 是 (>= 1M) |
| 缩放参数 | `[8192]` | 8,192 | fp32 | ❌ 否 (< 1M) |

在上面的例子中，缩放参数 `[8192]` fp32 (约 32KB) 因为元素数不足 1M 而被滤除。

**配置修改**:

在 `TransformerConfig` 中设置：
```python
transformer_config = TransformerConfig(
    ...
    min_offloaded_tensor_size=1024 * 1024,  # 默认值 1M 元素
)
```

---

### 2. Parameter 卸载控制

**代码位置**: `megatron/core/models/gpt/gpt_model.py:501-509`

```python
def preprocess_for_fine_grained_offloading(self):
    off_interface.init_chunk_handler(...)
    
    if self.disable_param_offloading:  # 默认为 True (第 128 行)
        for param in self.decoder.parameters():
            off_interface.mark_not_offloadable(param)
        if self.mtp_process:
            for param in self.mtp.parameters():
                off_interface.mark_not_offloadable(param)
        if self.post_process:
            for param in self.output_layer.parameters():
                off_interface.mark_not_offloadable(param)
        self.disable_param_offloading = False
```

**机制**: 通过 `mark_not_offloadable()` 将模型参数标记为不可卸载

```python
def mark_not_offloadable(self, tensor: torch.Tensor):
    if tensor is not None:
        tensor.offloading_activation = False
```

在卸载检查中：
```python
# Respect tensor's offload preference if specified
if hasattr(tensor, "offloading_activation") and not tensor.offloading_activation:
    return False
```

**关于 Parameters 是否会被 save_for_backward 捕获**:

在 PyTorch 的 autograd 机制中：

| Tensor 类型 | 是否被 `save_for_backward` 捕获 |
|------------|----------------------------------|
| **中间激活值 (Activations)** | ✅ 会 |
| **模型参数 (Parameters)** | ❌ 不会 (设计如此) |

Parameters 是模块的持久属性，不会被自动保存用于 backward。因此 `mark_not_offloadable()` 更多是一种**防御性编程**——虽然理论上 parameters 不会到达卸载逻辑，但显式标记可以避免潜在的边界情况风险。

**是否可以通过设置 `disable_param_offloading = False` 来卸载参数**:

| 情况 | 结果 |
|------|------|
| `disable_param_offloading = False` | 参数不会被标记 `offloading_activation = False`，但**仍然不会被卸载** |
| 原因 | PyTorch autograd 不会将 parameters 传入 `save_for_backward` |

因此，将 `disable_param_offloading` 设置为 `False` **不会**导致参数被卸载。

---

### 3. Group 级别卸载控制

**代码位置**: `megatron/core/pipeline_parallel/fine_grained_activation_offload.py:936-949`

```python
def should_bulk_offload(self):
    # Don't offload if the group is marked as not offloadable
    if not group.offload:
        return False

    # Check if next backward chunk is this chunk (for last pipeline stage)
    next_backward_chunk = PipelineOffloadManager.get_instance().front_backward_chunk(
        group._name
    )
    if next_backward_chunk is not None and next_backward_chunk is self:
        # Don't offload the last group with the same name if it's about to be used immediately
        if self.find_next_group(group._name) is None:
            return False
    return True
```

**Group 被标记为不卸载的场景**:

1. **Warmup 后的优化** (`post_warmup_callback`)
   - 最后几个相同名称的 group 被标记为 `offload = False`
   - 目的：避免 reloading 阻塞计算流

2. **Pipeline 边界优化**
   - 如果当前 chunk 就是下一个需要 backward 的 chunk
   - 且该 group 没有后续实例，则不卸载

---

## 总结：Tensor 被滤除的原因

| 原因 | 代码位置 | 说明 |
|------|----------|------|
| **Tensor 太小** | 第 865 行 | `numel() < min_offloaded_tensor_size` (默认 1M) |
| **显式禁用 offloading** | 第 868 行 | `tensor.offloading_activation = False` |
| **Group 被标记为不卸载** | 第 937 行 | Warmup 后 `_offload_margin` 个最后 group |
| **Pipeline 边界优化** | 第 946 行 | 即将被 backward 使用的 group |

---

## 调试建议

如需查看哪些 tensor 被卸载/没被卸载：

1. **开启 DEBUG 模式**:
   ```python
   # fine_grained_activation_offload.py 第 10 行
   DEBUG = True
   DEBUG_RANK = 0
   ```

2. **查看统计信息**:
   - `post_warmup_callback` 会打印每个 group 的卸载统计表

3. **关键日志位置**:
   - `tensor_need_offloading_checker`: 显示 tensor 是否通过检查
   - `bulk_offload_group`: 显示实际执行的卸载操作
