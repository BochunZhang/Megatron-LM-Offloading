# Megatron、TransformerEngine 与 PyTorch Autograd 关系解析

> **文档目的**：帮助理解 Megatron、TransformerEngine (TE) 和 PyTorch Autograd 三者之间的关系，特别是延迟 weight gradient 计算的设计。

---

## 1. 概述：三者关系总览

```
┌─────────────────────────────────────────────────────────────────┐
│                         Megatron                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              TransformerLayer (nn.Module)                  │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  SelfAttention (nn.Module)                           │  │ │
│  │  │  ├─ TELayerNormColumnParallelLinear (TE Module)      │  │ │
│  │  │  └─ TEDotProductAttention (TE Module)                │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  MLP (nn.Module) 叶子模块                             │  │ │
│  │  │  ├─ TEColumnParallelLinear (TE Module)               │  │ │
│  │  │  └─ TERowParallelLinear (TE Module)                  │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                    │                                            │
│                    ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           TransformerEngine (TE)                           │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  Linear (nn.Module)                                  │  │ │
│  │  │  ├─ weight (nn.Parameter)                            │  │ │
│  │  │  ├─ bias (nn.Parameter)                              │  │ │
│  │  │  └─ forward() calls _Linear.apply()                  │  │ │
│  │  │                                                      │  │ │
│  │  │  _Linear (torch.autograd.Function)                   │  │ │
│  │  │  ├─ forward() - 实际 GEMM 计算                        │  │ │
│  │  │  └─ backward() - 计算 dgrad 和 wgrad                  │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                    │                                            │
│                    ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           PyTorch Autograd                                 │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  Engine (C++) - 反向传播引擎                           │  │ │
│  │  │  ├─ 构建计算图（Node + Edge）                          │  │ │
│  │  │  ├─ 拓扑排序执行 backward                              │  │ │
│  │  │  └─ 管理梯度累加和通信同步                               │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 核心设计原则

1. **PyTorch nn.Module**：负责**参数管理**和**模块组合**
2. **TE autograd.Function**：负责**高性能计算**（CUDA kernel）
3. **Megatron 模块**：负责**分布式策略**和**执行流程控制**

---

## 2. PyTorch Autograd 基础

### 2.1 计算图结构

```
输入 x ──┐
         ├──► [Linear Op: y = x @ W^T] ──► 输出 y
权重 W ──┘              │
                        ▼
                   grad_fn (Node)
                   ┌──────────────┐
                   │   _Linear    │
                   │  (Function)  │
                   └──────────────┘
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
    next_edge[0]                 next_edge[1]
    (指向 x 的 grad_fn)          (指向 W 的 grad_fn)
```

### 2.2 反向传播执行流程

```python
# Python 层调用
loss.backward()
    ↓
# C++ Engine 执行
Engine::execute(roots, inputs, ...)
    ├─ 1. 创建 GraphTask（图任务）
    ├─ 2. 计算依赖关系（compute_dependencies）
    ├─ 3. 从根节点入队（execute_with_graph_task）
    ├─ 4. 线程主循环执行（thread_main）
    │   ├─ 从 ReadyQueue 弹出任务
    │   ├─ evaluate_function() 执行 Node
    │   │   ├─ call_pre_hooks()
    │   │   ├─ call_function() → Node::apply()
    │   │   └─ call_post_hooks()
    │   └─ 将梯度传递给前驱节点
    └─ 5. 等待完成，返回结果
```

### 2.3 关键概念

| 概念 | 说明 | 对应代码 |
|------|------|----------|
| **Node** | 计算图中的节点，对应一个操作 | `torch/csrc/autograd/function.h` |
| **Edge** | 连接节点的边，包含 `function` 和 `input_nr` | `torch/csrc/autograd/edge.h` |
| **GraphTask** | 一次 backward 执行的任务上下文 | `torch/csrc/autograd/graph_task.h` |
| **InputBuffer** | 累加多个下游传来的梯度 | `torch/csrc/autograd/input_buffer.h` |
| **ReadyQueue** | 待执行节点的优先级队列 | `torch/csrc/autograd/engine.h` |

---

## 3. TransformerEngine 设计

### 3.1 双层架构

TE 采用**双层设计**：外层 `nn.Module` 管理参数，内层 `autograd.Function` 执行计算。

```python
# 外层：nn.Module（参数管理）
class Linear(TransformerEngineBaseModule):
    def __init__(self, in_features, out_features, ...):
        # 注册参数
        self.register_parameter("weight", nn.Parameter(weight_tensor))
        self.register_parameter("bias", nn.Parameter(bias_tensor))
        # 延迟 weight gradient 存储
        self.wgrad_store = WeightGradStore(delay_wgrad_compute, ...)
    
    def forward(self, inp):
        # 调用内层 Function
        return _Linear.apply(weight, inp, bias, ..., self.wgrad_store, ...)

# 内层：autograd.Function（计算执行）
class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, inp, bias, ..., wgrad_store, ...):
        # 保存反向所需张量
        ctx.save_for_backward(inp, weight)
        ctx.wgrad_store = wgrad_store
        # 执行 GEMM
        return gemm_forward(inp, weight, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        
        # 1. 计算 data gradient (dgrad)
        dgrad = gemm(grad_output, weight)
        
        # 2. 计算 weight gradient (wgrad)
        if ctx.wgrad_store.delay_wgrad_compute():
            # 延迟计算：存储参数供后续调用
            ctx.wgrad_store.put([inp, grad_output], wgrad_gemm_func)
            wgrad = None  # 暂不返回
        else:
            # 立即计算
            wgrad = gemm(grad_output.t(), inp)
        
        return wgrad, dgrad, ...
```

### 3.2 为什么要分离？

| 层级 | 职责 | 为什么这样设计 |
|------|------|--------------|
| **nn.Module** | 参数生命周期管理 | PyTorch 标准模式，支持 optimizer、checkpoint 等 |
| **autograd.Function** | 高性能计算 | 可以调用 CUDA kernel，支持 FP8、通信重叠等优化 |

### 3.3 WeightGradStore 机制

```python
class WeightGradStore:
    """延迟 weight gradient 计算的存储队列"""
    
    def __init__(self, delay_wgrad_compute=False):
        self.enabled = delay_wgrad_compute
        self.context = queue.Queue()  # 存储待计算的 wgrad
    
    def put(self, tensor_list, func):
        """存储计算 wgrad 所需的参数和函数"""
        assert self.enabled
        self.context.put([tensor_list, func])
    
    def pop(self):
        """执行存储的 wgrad 计算"""
        assert self.enabled
        tensor_list, func = self.context.get()
        return func(*tensor_list)  # 执行实际的 GEMM
```

---

## 4. Megatron 模块层次

### 4.1 模块层级结构

```
TransformerLayer (非叶子模块)
├── input_layernorm (叶子 - TE LayerNorm)
├── self_attention (非叶子模块)
│   ├── linear_qkv (叶子 - TE Linear)
│   ├── core_attention (计算)
│   └── linear_proj (叶子 - TE Linear)
├── pre_mlp_layernorm (叶子 - TE LayerNorm)
└── mlp (叶子模块) ← 关键！包含 backward_dw
    ├── linear_fc1 (叶子 - TE ColumnParallelLinear)
    └── linear_fc2 (叶子 - TE RowParallelLinear)
```

### 4.2 叶子模块 vs 非叶子模块

| 类型 | 特征 | 是否包含参数 | backward_dw |
|------|------|------------|-------------|
| **非叶子模块** | 组合子模块，转发调用 | 否（参数在子模块） | 无 |
| **叶子模块** | 实际计算，直接持参 | 是（weight/bias） | 有 |

### 4.3 为什么 MLP 是叶子模块？

MLP 虽然自己不直接注册 `nn.Parameter`，但它是**包含参数的模块的最小容器单元**：

```python
class MLP(MegatronModule):
    def __init__(self, config, submodules):
        # 包含两个 Linear 子模块（每个都有 weight/bias）
        self.linear_fc1 = build_module(submodules.linear_fc1, ...)
        self.linear_fc2 = build_module(submodules.linear_fc2, ...)
    
    def forward(self, x):
        x = self.linear_fc1(x)  # 第一个 GEMM
        x = self.activation_func(x)
        x = self.linear_fc2(x)  # 第二个 GEMM
        return x
    
    def backward_dw(self):
        """触发子模块的 weight gradient 计算"""
        # 按 forward 相反顺序调用
        self.linear_fc2.backward_dw()  # 先计算 fc2 的 wgrad
        self.linear_fc1.backward_dw()  # 再计算 fc1 的 wgrad
```

注意：**TransformerLayer 本身没有 backward_dw**，因为它不直接包含参数，参数都在子模块中。

---

## 5. 延迟 Weight Gradient 计算

### 5.1 完整执行流程

```python
# ============ 前向传播 ============
hidden_states = transformer_layer(hidden_states)
# 内部执行：
#   input_layernorm → linear_qkv → attention → linear_proj → 
#   pre_mlp_layernorm → linear_fc1 → activation → linear_fc2

# ============ 反向传播（第一阶段）============
loss.backward()
# PyTorch Autograd 引擎执行：
#   1. 从 loss 开始遍历计算图
#   2. 执行每个 Node 的 backward（_Linear.backward）
#   3. _Linear.backward 中：
#      - 立即计算 dgrad（传递回上游）
#      - 将 wgrad 计算参数存入 WeightGradStore（延迟）
#      - 返回 (None, dgrad, ...) 给上游

# 此时：所有 activation gradient 已计算完成
#       所有 wgrad 计算参数已存储在队列中

# ============ 反向传播（第二阶段）============
for layer in model.layers:
    layer.mlp.backward_dw()  # 手动触发
# 内部执行：
#   mlp.backward_dw() → linear_fc2.backward_dw() → fc2.wgrad_store.pop() → 执行 GEMM
#                   → linear_fc1.backward_dw() → fc1.wgrad_store.pop() → 执行 GEMM
```

### 5.2 延迟计算的优势

| 优势 | 说明 |
|------|------|
| **显存优化** | 延迟存储中间激活值，减少峰值显存占用 |
| **通信重叠** | wgrad 计算可以与 all-reduce 通信重叠 |
| **流水并行** | 更好地支持 pipeline parallelism 的调度 |

### 5.3 配置启用

```python
# Megatron TransformerConfig
config = TransformerConfig(
    delay_wgrad_compute=True,  # 启用延迟 weight gradient
    ...
)

# TE Linear 会自动创建 WeightGradStore
class TELinear:
    def __init__(self, ..., delay_wgrad_compute=False):
        self.wgrad_store = WeightGradStore(delay_wgrad_compute, ...)
```

---

## 6. 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| 叶子模块 | Leaf Module | 直接包含 `nn.Parameter` 的模块，实际执行计算 |
| 非叶子模块 | Non-leaf Module | 组合子模块的容器，参数在子模块中 |
| Data Gradient | dgrad | 输入数据的梯度（传递给上游） |
| Weight Gradient | wgrad | 模型权重的梯度（用于 optimizer 更新） |
| 延迟计算 | Delayed Computation | 将计算推迟到特定时机执行，通常为了优化显存或重叠通信 |
| WeightGradStore | - | TE 中用于存储延迟 wgrad 计算参数和函数的队列 |
| GEMM | General Matrix Multiplication | 通用矩阵乘法（深度学习核心计算） |
| 列并行 | Column Parallel | 沿输出维度切分权重（Megatron TP 策略） |
| 行并行 | Row Parallel | 沿输入维度切分权重（Megatron TP 策略） |
| FP8 | FP8 | 8-bit 浮点精度（NVIDIA Hopper 架构支持） |
| Autograd Engine | - | PyTorch C++ 实现的反向传播引擎 |
| GraphTask | - | 一次 backward 执行的上下文任务对象 |
| ReadyQueue | - | 待执行 Node 的优先级队列（按拓扑序+序列号排序） |
| InputBuffer | - | 累加多个下游传来梯度的缓冲区 |

---

## 7. 代码索引

### PyTorch Autograd（本地源码）
- `torch/csrc/autograd/engine.h` - Engine 定义
- `torch/csrc/autograd/engine.cpp` - 反向传播实现
- `torch/csrc/autograd/function.h` - Node 基类
- `torch/csrc/autograd/graph_task.h` - GraphTask
- `torch/csrc/autograd/input_buffer.h` - InputBuffer

### TransformerEngine（本地源码）
- `transformer_engine/pytorch/module/linear.py` - Linear 模块
- `transformer_engine/pytorch/module/_common.py` - WeightGradStore

### Megatron-LM
- `megatron/core/transformer/transformer_layer.py` - TransformerLayer
- `megatron/core/transformer/mlp.py` - MLP（包含 backward_dw）
- `megatron/core/transformer/attention.py` - SelfAttention（包含 backward_dw）
- `megatron/core/extensions/transformer_engine.py` - TE 包装器

---

*文档生成时间：2025-06-01*
