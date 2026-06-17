# Measurement

## 1. NVTX 相关改动

### 1.1 enable nvtx

Megatron 默认禁用 nvtx 测试, nvtx_range_push 和 nvtx_range_pop 会检查 `global _nvtx_enabled` 的值 (默认为 False).

修改 megatron/training/training.py, 在 iteration == args.profile_step_start 时调用 `configure_nvtx_profiling(True)` 启用 nvtx, 在 iteration == args.profile_step_end 时调用 `configure_nvtx_profiling(False)` 关闭 nvtx.

### 1.2 增加 nvtx 标记

为 TransformerLayer / MultiLatentAttention / MLASelfAttention / MLP / MoELayer / TokenDispatcher 添加了 nvtx 标记, 主要关注涉及 GEMM 的计算和 EP 相关的通信. 

## 附录

DeepSeek-V3 前 3 层使用 TransformerLayer, 后 58 层使用 MoETransformerLayer, 其中 self_attention 子模块使用 MLASelfAttention.


### TransformerLayer

```
TransformerLayer
├── [Module 1] input_layernorm (RMSNorm / LayerNorm)
│
├── [Module 2] self_attention (自注意力模块)
│
├── [Module 3] self_attn_bda (Bias-Dropout-Add 残差连接)
│
├── [Module 4] pre_cross_attn_layernorm (可选，用于 encoder-decoder 模型)
│
├── [Module 5] cross_attention (可选的交叉注意力)
│
├── [Module 6] cross_attn_bda (可选的 Bias-Dropout-Add)
│
├── [Module 7] pre_mlp_layernorm (Layer Norm)
│
├── [Module 8] mlp (前馈网络)
│
└── [Module 9] mlp_bda (Bias-Dropout-Add 残差连接)
```

TransformerLayer 包含的模块如上, DeepSeek-V3 使用 Module 1/2/3/4/7/8/9, 核心子模块为 self_attention 和 mlp.


```
TransformerLayer.forward()
│
├── self._forward_attention() ← 注意力部分
│       │
│       ├── input_layernorm [Module 1] ← layernorm [支持 recompute, activaiton offload]
│       │
│       ├── self_attention [Module 2] ← MLASelfAttention / DotProductAttention
│       │
│       ├── self_attn_bda [Module 3] ← Bias-Dropout-Add
│       │
│       ├── pre_cross_attn_layernorm [Module 4] ← 可选
│       │
│       ├── cross_attention [Module 5] ← 可选
│       │
│       ├── cross_attn_bda [Module 6] ← 可选
│       │
│       └── hidden_states, context
│
└── self._forward_mlp() ← MLP 部分
        │
        ├── self.pre_mlp_layernorm()
        │       |
        │       └── pre_mlp_layernorm [Module 7] ← layernorm [支持 recompute, activaiton offload]
        │
        ├── mlp [Module 8] ← MLP / MoELayer [支持整个 mlp 做 recompute]
        │
        └── self._forward_post_mlp()
                |
                └── mlp_bda [Module 9] ← Bias-Dropout-Add
```

MoETransformerLayer 是 TransformerLayer 的派生类.
由于 MoE 的动态特性, 无法捕获整个 layer 到单个 CUDA graph. 
MoETransformerLayer 将 forward 过程分解为 router, expert-compute, post-process 等 stage 来支持 "partial" CUDA graph.

### MLASelfAttention

```
MLASelfAttention [继承自 MultiLatentAttention]
|
├── [Module 1] rotary_pos_emb (处理 rope embedding) [继承自 MultiLatentAttention]
|
├── [Module 2] core_attention [继承自 MultiLatentAttention]
|
├── [Module 3] linear_proj (处理输出) [继承自 MultiLatentAttention]
|
├── [Module 4] linear_q_proj (可选, 和 linear_q_down_proj & linear_q_up_proj 二选一)
|
├── [Module 5] linear_q_down_proj (可选, LoRA 压缩: hidden_size → q_lora_rank)
|
├── [Module 6] linear_q_up_proj (可选, LoRA 解压, 与 linear_q_down_proj 共同使用)
|
├── [Module 7] linear_kv_down_proj
|
├── [Module 8] linear_kv_up_proj
|
├── [Module 9] q_layernorm (可选)
│
└── [Module 10] kv_layernorm
```

```
MLASelfAttention.forward() ← 在基类 MultiLatentAttention 中定义, 通过 abstract func 被基类的 forward 函数调用
│
├── self.get_query_key_value_tensors() ← RoPE 准备和应用
│       │
│       ├── rotary_pos_emb [Module 1] ← 准备 RoPE 参数
│       │
│       ├── linear_q_down_proj [Module 5] ← Q 下投影 (可选)
│       │
│       ├── linear_kv_down_proj [Module 7] ← KV 下投影 
│       │
│       ├── q_layernorm [Module 9] ← (可选, 取决于是否调用 linear_q_down_proj)
│       │
│       ├── kv_layernorm [Module 10]
|       |
│       └── qkv_up_proj_and_rope_apply() ← Q/KV 上投影 + RoPE 应用 [支持 recompute]
|               |
|               ├── linear_q_up_proj [Module 6] ← 若调用 linear_q_down_proj 则从 latent space 上采样
|               |
|               ├── linear_q_proj [Module 4] ← 若未调用 linear_q_down_proj 则直接上采样
|               |
|               ├── linear_kv_up_proj [Module 8] linear_q_down_proj
|               |
|               ├── apply_rotary_pos_emb ← q 位置编码
|               |
|               └── apply_rotary_pos_emb ← k 位置编码
│
├── _adjust_key_value_for_inference() ← 推理时调用, 填充 kv cache
│
├── core_attention [Module 2] ← [支持 recompute, activaiton offload]
│
└── linear_proj [Module 3] ← [支持 activaiton offload]
```

MLA 的核心是将 KV 降维至 latent space, 推理时可以从 latent space 还原得到 KV, 节省开销.
MLA 可选对 Q 做低秩分解, 一方面两个小矩阵比一个大矩阵的参数量更小, 一方面可以应用 recompute_up_proj 重计算, 减少激活值.

```
MLASelfAttention.backward_dw() ← 逆序计算权重梯度, 区别于 dx
│
├── _backward_kv_proj()
|       |
│       ├── linear_kv_up_proj.backward_dw()
|       |
│       └── linear_kv_down_proj.backward_dw()
│
├── _backward_q_proj()
|       |
|       ├── linear_q_up_proj.backward_dw() ← (如果对 q 做了低秩压缩)
|       |
|       ├── linear_q_down_proj.backward_dw() ← (如果未对 q 做了低秩压缩)
|       |
|       └── linear_q_proj.backward_dw() ← (如果未对 q 做了低秩压缩)
│
└── _backward_output_proj()
        |
        └── linear_proj.backward_dw()
```

拆分了 bgrad 和 wgrad 逻辑, 前者需要被传递, 后者用于更新权重.


### Transformer MLP

#### Dense Layer MLP

```
MLP
├── [Module 1] linear_fc1 (上投影: hidden_size → ffn_hidden_size)
|
├── [Module 2] activation_func (激活函数: SwiGLU / GELU / GeGLU)
|
└── [Module 3] linear_fc2 (下投影: ffn_hidden_size → hidden_size)
```

```
MLP.forward()
│
├── linear_fc1 [Module 1] ← 上投影
│
├── activation_func [Module 2] ← 激活函数 (支持 SwiGLU/GELU 融合)
|   |
│   ├── 支持选项: use_te_activation_func, bias_activation_fusion, gated_linear_unit
|   |
│   ├── per_token_scale (动态缩放)
|   |
│   └── 融合实现: weighted_bias_swiglu_impl, weighted_bias_quick_gelul_impl, bias_geglu_impl, bias_swiglu_impl
│
└── linear_fc2 [Module 3] ← 下投影
```

```
MLP.backward()
│
└── backward_dw()  ← 逆序计算权重梯度
        ├── linear_fc2.backward_dw()
        |
        └── linear_fc1.backward_dw()
```

#### Sparse Layer MoE (Mixture of Experts)

```
MoELayer
├── [Module 1] router
│
├── [Module 2] fc1_latent_proj (可选, 用于压缩 hidden_size → moe_latent_size)
|
├── [Module 3] fc2_latent_proj (可选, 用于解压 moe_latent_size → hidden_size)
|
├── [Module 4] token_dispatcher (不是 nn.Module)
│
├── [Module 5] experts
│
└── [Module 6] shared_experts  (共享专家)

```

图中列举了 MoE layer 相关的 module, 其中 token_dispatcher 没有可学习的参数, 但需要追踪其 all-to-all 时间.

```
MoELayer.forward()
└── custom_forward() ← 封装 forward 功能, 以支持 recompute
        |
        ├── shared_experts_compute() ← 共享专家计算
        |       |
        |       └── shared_experts [Module 6] ← 支持 recompute
        |
        ├── route() ← 计算路由概率和映射, maybe_skip_or_early_return_by_cudagraph
        |       |
        |       └── router [Module 1]
        │
        ├── preprocess()  ← 预处理
        │       │
        │       ├── fc1_latent_proj [Module 2] ← (如果使用 moe_latent_size 则调用)
        │       │
        │       └── token_dispatcher.dispatch_preprocess()
        │
        ├── dispatch() ← 通信: tokens → expert devices
        │       │
        │       └── token_dispatcher.token_dispatch()
        │
        ├── routed_experts_compute()  ← 路由专家计算
        │       │
        │       ├── token_dispatcher.dispatch_postprocess()
        │       │
        │       ├── experts() [Module 5]
        │       │
        │       └── token_dispatcher.combine_preprocess()
        │
        ├── combine()
        │       │
        │       └── token_dispatcher.token_combine()
        │
        └── postprocess()
                │
                ├── token_dispatcher.combine_postprocess()
                │
                ├── fc2_latent_proj [Module 3] ← (如果使用 moe_latent_size 则调用)
                │
                └── output + shared_expert_output
```

主要功能都实现在 dispathcer 里面.

```
MoELayer.backward_dw() ← 逆序计算权重梯度
        ├── experts.backward_dw()  ← 路由专家
        │
        └── shared_experts.backward_dw()  ← 共享专家
```


### Dispatcher

#### DeepEP

