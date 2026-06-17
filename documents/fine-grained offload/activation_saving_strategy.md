# Transformer 中的激活值保存策略

## 1. 需要保存的激活值

### 1.1 PyTorch 自动保存

| Layer 类型 | 自动保存的内容 | 用途 |
|-----------|---------------|------|
| Linear | input, weight | 计算 grad_input 和 grad_weight |
| LayerNorm | input | 计算 mean/variance 的梯度 |
| 激活函数 (GeLU/SwiGLU) | input | 计算激活函数的导数 |
| Dropout | mask | 梯度传播时归零对应位置 |

### 1.2 Custom Autograd Function 显式保存

```python
# fused_softmax.py:36
ctx.save_for_backward(softmax_results, scale_t)

# fused_bias_swiglu.py:121
ctx.save_for_backward(input_for_backward, bias)

# fused_cross_entropy.py:119
ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
```

### 1.3 Residual Connection

```python
# transformer_layer.py:573
residual = hidden_states  # 必须保存

# ... attention ...

# transformer_layer.py:630
hidden_states = self.self_attn_bda(attention_output_with_bias, residual, self.hidden_dropout)
# backward 时需要: attention_output, residual, dropout_mask
```

## 2. 可以被释放/重新计算的激活值

### 2.1 Selective Recompute

```python
# transformer_config.py:383
recompute_granularity: Optional[Literal['full', 'selective']] = None
recompute_modules: Optional[List[str]] = None  # ["core_attn", "layernorm", "mlp", ...]
```

| Module | 释放的内容 | 节省内存 |
|--------|-----------|---------|
| `core_attn` | Attention 计算中的中间 Q/K/V | 最大头状态内存 |
| `layernorm` | LayerNorm 输出 | 隐藏层维度张量 |
| `mlp` | MLP 前向传播的中间结果 | FFN 维度张量 |
| `moe_act` | MoE 激活函数输出 | 专家数量 × 隐藏层 |
| `shared_experts` | 共享专家的输出 | 共享专家内存 |
| `mla_up_proj` | MLA 上投影输出 | MLA 特定张量 |

### 2.2 CheckpointWithoutOutput 机制

```python
# tensor_parallel/random.py:621
class CheckpointWithoutOutput:
    """
    Checkpoint a model or part of the model and release the output.

    Forward: 保存 input，计算 output，但立即丢弃/释放 output
    Backward: 重新使用 input 计算 output，用于梯度计算
    """
```

**实现原理**:
```python
def checkpoint(self, run_function, *args):
    # Forward: 保存 input，计算 output
    outputs = CheckpointWithoutOutputFunction.apply(run_function, self, *args)
    self.outputs = outputs
    return outputs

def discard_output_and_register_recompute(self, hook_tensor):
    # 丢弃 output，释放内存
    # 在 hook_tensor 的 backward hook 中重新计算
    hook_tensor.register_hook(self._recompute)

def _recompute(self, _):
    # Backward: 重新计算 output
    with torch.enable_grad():
        outputs = self.run_function(*inputs)
    # 恢复内存
    for output, recomputation in zip(self.outputs, outputs):
        output.untyped_storage().copy_(recomputation.untyped_storage())
```

### 2.3 显式释放

```python
# transformer_layer.py:639
if self.offload_attn_norm:
    hidden_states = off_interface.group_commit(
        hidden_states, name="attn_norm", forced_released_tensors=[residual]
    )
    # residual.untyped_storage().resize_(0) 显式释放
```

## 3. 内存优化策略对比

| 优化技术 | 释放的激活 | 保留的激活 |
|---------|-----------|-----------|
| **无优化** | 无 | 所有中间结果 |
| **Selective Recompute** | LayerNorm/MLP/Attention 输出 | Input, Residual, Weight |
| **Checkpoint (full)** | 整个 layer 的中间结果 | 输入张量 |
| **Activation Offloading** | 被 offload 到 CPU 的激活 | 在 GPU 等待使用的 |
| **FP8 Delayed Scaling** | 部分量化中间结果 | Scaling factor |

## 4. Transformer Layer 内存占用分析

### 4.1 Dense Layer（无优化）

必须保存：
```
├── Residuals: 2 × [seq_len, batch, hidden]  # 两个残差连接
├── LayerNorm 输入: 2 × [seq_len, batch, hidden]  # 如果不 recompute
├── Q/K/V: 3 × [seq_len, batch, hidden]  # Attention 输入
├── Softmax 输出: [seq_len, batch, heads, head_dim]  # Attention 权重
└── MLP 中间: [seq_len, batch, 4×hidden]  # SwiGLU 可能 8×
```

### 4.2 启用 Selective Recompute 后

```python
# TransformerLayer.__init__
if self.config.recompute_granularity == 'selective':
    if "layernorm" in self.config.recompute_modules:
        self.recompute_input_layernorm = True
        self.recompute_pre_mlp_layernorm = True
    if "mlp" in self.config.recompute_modules:
        self.recompute_mlp = True
```

节省的内存：
```
可通过 Recompute 节省:
├── LayerNorm 输出: 2 × [seq_len, batch, hidden]  # 重新计算
├── MLP 中间: [seq_len, batch, 4×hidden]  # 重新计算
└── Attention 输出投影前: [seq_len, batch, hidden]  # 重新计算

必须保留:
├── Residuals: 2 × [seq_len, batch, hidden]
├── LayerNorm 输入: 2 × [seq_len, batch, hidden]  # 用于 recompute
└── Q/K/V: 3 × [seq_len, batch, hidden]
```

### 4.3 启用 Fine-grained Offload

```python
# 配置
--fine-grained-activation-offloading \
--offload-modules mlp_norm attn_norm
```

内存分布：
```
GPU Memory:
├── 正在使用的激活
├── 待 offload 的激活（临时）
└── 已 reload 的激活（临时）

CPU Memory (Pinned):
├── offload_groups[0]: attn_norm tensors
├── offload_groups[1]: qkv_linear tensors
├── offload_groups[2]: core_attn tensors
└── ...
```

## 5. 关键代码位置

### 5.1 Recompute 配置

```python
# transformer_config.py:1193
if self.recompute_modules is None:
    self.recompute_modules = ["core_attn"]

if self.recompute_granularity == "selective":
    allowed_modules = {
        "core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe", "shared_experts"
    }
```

### 5.2 CheckpointWithoutOutput 使用

```python
# transformer_layer.py:577
if self.recompute_input_layernorm:
    self.input_layernorm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
    with off_interface(self.offload_attn_norm, hidden_states, "attn_norm"):
        input_layernorm_output = self.input_layernorm_checkpoint.checkpoint(
            self.input_layernorm, hidden_states
        )

# transformer_layer.py:613
if self.recompute_input_layernorm:
    # discard the output of the input layernorm
    self.input_layernorm_checkpoint.discard_output_and_register_recompute(
        attention_output_with_bias[0]
    )
```

### 5.3 Offload 配置

```python
# transformer_config.py:889
fine_grained_activation_offloading: bool = False

offload_modules: Optional[list[str]] = field(default_factory=list)
"""choices: attn_norm, qkv_linear, core_attn, attn_proj, mlp_norm, expert_fc1, moe_act"""

min_offloaded_tensor_size: int = 1024 * 1024
```

## 6. 决策流程

```
Forward 中的激活值:
│
├─► 是否需要在 backward 中使用？
│       ├── 否 ──► 直接释放
│       └── 是 ──► 继续判断
│
├─► 是否配置为 recompute？
│       ├── 是 ──► CheckpointWithoutOutput
│       │             ├─► Forward: 保存 input，计算 output，丢弃 output
│       │             └─► Backward: 重新计算 output
│       └── 否 ──► 继续判断
│
├─► 是否配置为 offload？
│       ├── 是 ──► Fine-grained Offload
│       │             ├─► Forward: offload to CPU
│       │             └─► Backward: reload to GPU
│       └── 否 ──► PyTorch 默认保存
│
└─► PyTorch autograd 自动管理
      └─► 保存到 ctx.saved_tensors
```
