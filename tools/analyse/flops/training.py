# fork from megatron/training/training.py

from megatron.core.models.gpt.experimental_attention_variant_module_specs import is_linear_attention_variant


def num_floating_point_operations(args, batch_size, return_breakdown=False):
    def calculate_layer_counts():
        """Calculate the number of attention, Mamba, and MLP layers."""
        if args.hybrid_override_pattern:
            counts = {'M': 0, '*': 0, '-': 0, 'E':0}
            for layer_type in args.hybrid_override_pattern:
                if layer_type in counts:
                    counts[layer_type] += 1
            return counts['*'], counts['M'], counts['-'], counts['E']
        else:
            num_attn_layers = round(args.num_layers * args.hybrid_attention_ratio)
            num_mlp_layers = round(args.num_layers * args.hybrid_mlp_ratio)
            num_mamba_layers = args.num_layers - num_attn_layers - num_mlp_layers
            num_moe_layers = 0
            return num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers

    def mlp_layer_flops(batch_size, seq_len, hidden_size, expansion=4.0, swiglu=False):
        """Calculate FLOPs for an MLP layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        return 4 * expansion * scale_factor * batch_size * seq_len * hidden_size**2

    def moe_layer_flops(batch_size, seq_len, hidden_size, moe_ffn_hidden_size,
                        shared_expert_ffn_hidden_size, num_experts_routed_to,
                        moe_latent_size=None, swiglu=False):
        """Calculate FLOPs for an MoE layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        if moe_latent_size is None:
            routed_flops = (4 * batch_size * seq_len * hidden_size *
                            moe_ffn_hidden_size * num_experts_routed_to * scale_factor)
        else:
            # Routed experts run on moe_latent_size.
            routed_flops = (4 * batch_size * seq_len * moe_latent_size *
                            moe_ffn_hidden_size * num_experts_routed_to * scale_factor)
            # Up proj and down proj.
            routed_flops += (4 * batch_size * seq_len * hidden_size * moe_latent_size)
        shared_flops = 4 * batch_size * seq_len * hidden_size * shared_expert_ffn_hidden_size * scale_factor
        return routed_flops + shared_flops

    def attn_layer_flops(
        batch_size, seq_len, hidden_size, num_heads, gqa=True, gqa_groups=8, kv_channels=None
    ):
        """Calculate FLOPs for an attention layer."""
        p = (kv_channels * num_heads / hidden_size) if kv_channels else 1
        g = gqa_groups if gqa else num_heads
        return (
            4
            * batch_size
            * seq_len
            * hidden_size
            * p
            * (hidden_size + (hidden_size * (g / num_heads)) + (seq_len / 2))
        )

    def mamba_layer_flops(batch_size, seq_len, hidden_size, state_dim=16,
                          head_dim=64, num_groups=1, num_heads=128):
        """Calculate FLOPs for a Mamba layer."""
        # Note (rwaleffe): flops estimate for scan should be updated based on new SSD kernels,
        # but small percent of overall layer flops
        d_in = 2 * hidden_size
        if num_heads:
            nheads = num_heads
        else:
            nheads = d_in // head_dim
        return (
            (
                2
                * batch_size
                * seq_len
                * hidden_size
                * (2 * d_in + 2 * num_groups * state_dim + nheads)
            )  # in_proj
            + (7 * batch_size * seq_len * d_in * state_dim)  # scan
            + (2 * batch_size * seq_len * d_in * hidden_size)  # out_proj
        )

    def hybrid_flops(batch_size, seq_len, hidden_size,
                     num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers,
                     mamba_state_dim=128, mamba_head_dim=64,
                     mamba_num_groups=8, mamba_num_heads=128,
                     num_attn_heads=32, gqa=True,
                     gqa_groups=8, kv_channels=None,
                     mlp_expansion=4.0, swiglu=False,
                     moe_latent_size=None,
                     moe_ffn_hidden_size=2048, shared_expert_ffn_hidden_size=2048, num_experts_routed_to=1,
                     vocab_size=256000):
        """Calculate total FLOPs for the hybrid model."""
        attn_flops = num_attn_layers * attn_layer_flops(batch_size, seq_len, hidden_size,
                                                         num_attn_heads, gqa, gqa_groups, kv_channels)
        mlp_flops = num_mlp_layers * mlp_layer_flops(batch_size, seq_len, hidden_size,
                                                     mlp_expansion, swiglu)
        mamba_flops = num_mamba_layers * mamba_layer_flops(batch_size, seq_len, hidden_size,
                                                           mamba_state_dim, mamba_head_dim,
                                                           mamba_num_groups, mamba_num_heads)
        moe_flops = num_moe_layers * moe_layer_flops(batch_size, seq_len, hidden_size, moe_ffn_hidden_size,
                                                       shared_expert_ffn_hidden_size, num_experts_routed_to,
                                                       moe_latent_size, swiglu)
        logits_flops = 2 * batch_size * seq_len * hidden_size * vocab_size
        flops_fwd = attn_flops + mlp_flops + mamba_flops + moe_flops + logits_flops

        # Dense transformer (attention + mlp)
        dense_transformer_flops = attn_flops + mlp_flops
        # Sparse transformer (moe + mamba)
        sparse_transformer_flops = moe_flops + mamba_flops

        return {
            'attention': attn_flops,
            'mlp': mlp_flops,
            'moe': moe_flops,
            'mamba': mamba_flops,
            'logits': logits_flops,
            'dense_transformer': dense_transformer_flops,
            'sparse_transformer': sparse_transformer_flops,
            'total': flops_fwd * 3
        }

    def transformer_flops(return_breakdown=False):
        """Calculate FLOPs for a standard Transformer model."""
        # TODO(helenn/dnarayanan): Refactor this to reuse the helper methods.
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # MoE.
        if args.num_experts is None:
            # Every Transformer MLP is dense.
            num_dense_layers = args.num_layers
            num_moe_layers = 0
            num_experts_routed_to = 0
            last_layer_is_moe = 0
        else:
            # Calculate number of dense and MoE Transformer MLPs.
            if isinstance(args.moe_layer_freq, int):
                moe_layer_pattern = [
                    1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
                ]
            elif isinstance(args.moe_layer_freq, list):
                moe_layer_pattern = args.moe_layer_freq
            else:
                raise RuntimeError("Illegal --moe-layer-freq argument provided!")
            assert len(moe_layer_pattern) == args.num_layers, (
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {args.num_layers}, "
                f"current moe layer pattern: {args.moe_layer_freq}"
            )
            num_moe_layers = sum(moe_layer_pattern)  # Number of 1s in `moe_layer_pattern`.
            num_dense_layers = args.num_layers - num_moe_layers
            num_experts_routed_to = args.moe_router_topk
            last_layer_is_moe = moe_layer_pattern[-1]

        if args.mtp_num_layers is not None:
            mtp_num_layers = args.mtp_num_layers
            num_moe_layers += last_layer_is_moe * mtp_num_layers
            num_dense_layers += (1 - last_layer_is_moe) * mtp_num_layers
            num_layers = args.num_layers + mtp_num_layers
        else:
            mtp_num_layers = 0
            num_layers = args.num_layers

        moe_ffn_hidden_size = (
            args.moe_ffn_hidden_size
            if args.moe_ffn_hidden_size is not None
            else args.ffn_hidden_size
        )
        shared_expert_ffn_hidden_size = (
            0
            if args.moe_shared_expert_intermediate_size is None
            else args.moe_shared_expert_intermediate_size
        )

        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        forward_backward_expansion_factor = 3
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        fma_expansion_factor = 2
        # - 3x (SwiGLU enabled): h->2*ffn_h GEMM and ffn_h->h GEMM are stacked.
        # - 2x (SwiGLU disabled): h->ffn_h GEMM and ffn_h->h GEMM are stacked.
        ffn_expansion_factor = 3 if args.swiglu else 2

        if args.multi_latent_attention:
            assert not args.group_query_attention
            '''
            Basic arithmetic
            let B is batch size, s is seq_len, h is embedding dim,
            for one self_attnetion block (prenorm is not included)
            qkv projection:  6Bsh^2
            attn:            2Bs^2h
            attn over value: 2Bs^2h
            oproj:           2Bsh^2

            references
            https://arxiv.org/abs/2305.10403
            https://arxiv.org/abs/2205.05198
            '''
            ## MLA
            if args.q_lora_rank is None:
                q_term = (
                    args.hidden_size
                    * args.num_attention_heads
                    * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                )
            else:
                q_term = args.q_lora_rank * (
                    args.hidden_size
                    + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                    + 1
                )
            standard_self_attn_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * (
                    ## q lora + rope + q norm
                    q_term
                    ## kv lora + rope + kv norm
                    + args.kv_lora_rank
                    * (
                        args.hidden_size
                        + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
                        + 1
                    )
                    + args.hidden_size * args.qk_pos_emb_head_dim
                    ## o proj
                    + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
                    ## core attn
                    + args.seq_length
                    * (args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim))
                    / 2  # causal mask (only half of the mask is non-zero)
                    + args.seq_length * args.num_attention_heads * args.v_head_dim / 2
                )
            )

        else:
            ## MHA or GQA
            query_projection_size = args.kv_channels * args.num_attention_heads
            key_projection_size = args.kv_channels * args.num_query_groups
            value_projection_size = args.kv_channels * args.num_query_groups
            gate_projection_size = query_projection_size if args.attention_output_gate else 0
            standard_self_attn_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * (
                    ## qkv proj
                    args.hidden_size
                    * (
                        query_projection_size
                        + key_projection_size
                        + value_projection_size
                        + gate_projection_size
                    )
                    ## core attention
                    + query_projection_size
                    * args.seq_length
                    / 2  # causal mask (only half of the mask is non-zero)
                    * 2  # QK^T and (QK^T)V
                    ## out proj
                    + query_projection_size
                    * args.hidden_size
                )
            )

        if is_linear_attention_variant(args.experimental_attention_variant):
            # Calculate number of dense and MoE Transformer MLPs.
            if isinstance(args.linear_attention_freq, int):
                linear_attention_pattern = [
                    # [1,1,...,1,0,1,1,...,1,0,...]
                    0 if ((i + 1) % args.linear_attention_freq == 0)
                    else 1 for i in range(num_layers)
                ]
            elif isinstance(args.linear_attention_freq, list):
                linear_attention_pattern = args.linear_attention_freq
                assert len(linear_attention_pattern) == num_layers, (
                    f"Invalid length of linear_attention_pattern: {len(linear_attention_pattern)}, "
                    f"expected {num_layers}, "
                    f"current linear attention pattern: {args.linear_attention_freq}"
                )
            elif args.linear_attention_freq is None:
                # This should be caught by config validation, but raise here as a safety check
                raise ValueError(
                    f"Linear attention type {args.experimental_attention_variant} is specified "
                    "but linear_attention_freq is None. "
                    "Please set linear_attention_freq to specify the LA/SDPA layer pattern."
                )
            else:
                raise ValueError(
                    f"Invalid linear_attention_freq: {type(args.linear_attention_freq)},"
                    f" {args.linear_attention_freq}"
                )
            num_linear_attention_layers = sum(linear_attention_pattern)
            num_standard_attention_layers = num_layers - num_linear_attention_layers

            if args.experimental_attention_variant == "gated_delta_net":
                # Calculate the FLOPs for the gated delta net attention.
                qk_head_dim = args.linear_key_head_dim
                v_head_dim = args.linear_value_head_dim
                num_qk_heads = args.linear_num_key_heads
                num_v_heads = args.linear_num_value_heads
                qk_dim = qk_head_dim * num_qk_heads
                v_dim = v_head_dim * num_v_heads
                linear_self_attn_term = (
                    forward_backward_expansion_factor
                    * fma_expansion_factor
                    * (
                        ## in proj
                        args.hidden_size
                        * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
                        ## conv1d
                        + args.linear_conv_kernel_dim
                        * (2 * qk_dim + v_dim)
                        ## gated delta rule
                        + num_v_heads
                        * (v_head_dim ** 2)
                        * 4  # KK^T, VK^T, S(a(I-bKK^T)), and SQ
                        ## out proj
                        + args.hidden_size
                        * v_dim
                    )
                )
            else:
                raise ValueError(
                    "Invalid experimental_attention_variant: "
                    f"{args.experimental_attention_variant}"
                )
        else:
            num_linear_attention_layers = 0
            linear_self_attn_term = 0
            num_standard_attention_layers = num_layers

        self_attn_term = (
            linear_self_attn_term * num_linear_attention_layers
            + standard_self_attn_term * num_standard_attention_layers
        )

        # Calculate breakdown components
        bs_sl = batch_size * args.seq_length
        fb_fae = forward_backward_expansion_factor * fma_expansion_factor

        # MLP component
        dense_mlp_flops = bs_sl * fb_fae * args.hidden_size * (
            args.ffn_hidden_size * ffn_expansion_factor
        ) * num_dense_layers

        # MoE routed experts
        moe_routed_flops = bs_sl * fb_fae * args.hidden_size * (
            moe_ffn_hidden_size * num_experts_routed_to * ffn_expansion_factor
        ) * num_moe_layers

        # MoE shared experts
        moe_shared_flops = bs_sl * fb_fae * args.hidden_size * (
            shared_expert_ffn_hidden_size * ffn_expansion_factor
        ) * num_moe_layers

        # Self Attention
        attn_flops = bs_sl * self_attn_term

        # MTP norms and proj
        mtp_flops = bs_sl * fb_fae * mtp_num_layers * (
            3 * args.hidden_size  # MTP eh norm + final norm
            + 2 * args.hidden_size * args.hidden_size  # MTH eh proj
        )

        # Logit
        logits_flops = bs_sl * fb_fae * args.hidden_size * args.padded_vocab_size * (mtp_num_layers + 1)

        total_floating_point_operations = dense_mlp_flops + moe_routed_flops + moe_shared_flops + attn_flops + mtp_flops + logits_flops

        if return_breakdown:
            return {
                'attention': attn_flops,
                'dense_mlp': dense_mlp_flops,
                'moe_routed': moe_routed_flops,
                'moe_shared': moe_shared_flops,
                'moe_total': moe_routed_flops + moe_shared_flops,
                'mtp': mtp_flops,
                'logits': logits_flops,
                'dense_transformer': dense_mlp_flops + attn_flops,
                'sparse_transformer': moe_routed_flops + moe_shared_flops,
                'total': total_floating_point_operations
            }
        return total_floating_point_operations

    # Main entrypoint for FLOPs calculation.
    if args.is_hybrid_model:
        # Calculate the number of each type of layer.
        num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers = calculate_layer_counts()

        # Compute hybrid model FLOPs.
        result = hybrid_flops(
            batch_size=batch_size,
            seq_len=args.seq_length,
            hidden_size=args.hidden_size,
            num_attn_layers=num_attn_layers,
            num_mamba_layers=num_mamba_layers,
            num_mlp_layers=num_mlp_layers,
            num_moe_layers=num_moe_layers,
            mamba_state_dim=args.mamba_state_dim,
            mamba_head_dim=args.mamba_head_dim,
            mamba_num_groups=args.mamba_num_groups,
            mamba_num_heads=args.mamba_num_heads,
            num_attn_heads=args.num_attention_heads,
            gqa=args.group_query_attention,
            gqa_groups=args.num_query_groups,
            kv_channels=args.kv_channels,
            mlp_expansion=args.ffn_hidden_size / args.hidden_size,
            swiglu=args.swiglu,
            moe_latent_size=args.moe_latent_size,
            moe_ffn_hidden_size=(args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None
                                 else args.ffn_hidden_size),
            shared_expert_ffn_hidden_size=(0 if args.moe_shared_expert_intermediate_size is None
                                           else args.moe_shared_expert_intermediate_size),
            num_experts_routed_to=args.moe_router_topk,
            vocab_size=args.padded_vocab_size,
        )
        if return_breakdown:
            return result
        return result['total']
    else:
        # Compute standard Transformer model FLOPs.
        result = transformer_flops(return_breakdown=return_breakdown)
        if return_breakdown:
            return result
        return result