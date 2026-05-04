# 测试 MBS 对 GEMM 的影响

## 1. MLA backend 配置

```megatron/core/models/gpt/gpt_layer_specs.py
if multi_latent_attention:
    assert qk_l2_norm is False, "qk_l2_norm is not supported with MLA."
    linear_q_up_proj = (
        backend.column_parallel_layer_norm_linear()
        if qk_layernorm
        else backend.column_parallel_linear()
    )
    linear_kv_up_proj = (
        backend.column_parallel_layer_norm_linear()
        if qk_layernorm
        else backend.column_parallel_linear()
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=backend.layer_norm(),
            self_attention=ModuleSpec(
                module=MLASelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=MLASelfAttentionSubmodules(
                    linear_q_proj=backend.column_parallel_linear(),
                    linear_q_down_proj=backend.linear(),
                    linear_q_up_proj=linear_q_up_proj,
                    linear_kv_down_proj=backend.linear(),
                    linear_kv_up_proj=linear_kv_up_proj,
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=IdentityOp,
                    kv_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )
```

| Module | Megatron | TE | input | output | bias | DeepSeek V3 配置 |
|----------|--------------|-----------|---------|---------|--------|----------------|
| linear_q_down_proj | TELinear | te.pytorch.Linear | [S, B, 7168] | [S, B, 1536] | F | hidden_size → q_lora_rank |
| linear_kv_down_proj | TELinear | te.pytorch.Linear | [S, B, 7168] | [S, B, 576] | F | hidden_size → kv_lora_rank + qk_pos_emb_head_dim |
| linear_q_up_proj | TELayerNormColumnParallelLinear | te.pytorch.LayerNormLinear | [S, B, 1536] | [S, B, 24576] | F | q_lora_rank → num_heads × qk_head_dim |
| linear_kv_up_proj | TELayerNormColumnParallelLinear | te.pytorch.LayerNormLinear | [S, B, 512] | [S, B, 32768] | F |kv_lora_rank → num_heads × (qk_head_dim + v_head_dim)|
| core_attention | TEDotProductAttention | te.pytorch.DotProductAttention | Q: [S, B, 128, 192]<br>K: [S, B, 128, 192]<br>V: [S, B, 128, 128] | N/A | - | Flash Attention |
| linear_proj | TERowParallelLinear | te.pytorch.Linear | [S, B, 16384] | [S, B, 7168] | F | num_heads × v_head_dim → hidden_size |


