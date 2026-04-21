# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical GEMM flops for model training without instantiating
a model and running training iterations on GPU(s)."""

from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from megatron.training.training import num_floating_point_operations

if __name__ == "__main__":
    initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)
    args = get_args()

    # Calculate flops per training step (per micro batch)
    # Note: This is flops per micro batch, not per global batch
    flops_per_step = num_floating_point_operations(args, args.micro_batch_size)

    print("=" * 80)
    print("THEORETICAL GEMM FLOATING POINT OPERATIONS")
    print("=" * 80)
    print(f"Micro batch size: {args.micro_batch_size}")
    print(f"Global batch size: {args.global_batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"FFN hidden size: {args.ffn_hidden_size}")
    print(f"Number of attention heads: {args.num_attention_heads}")
    print(f"Pipeline parallel size: {args.pipeline_model_parallel_size}")
    print(f"Tensor parallel size: {args.tensor_model_parallel_size}")
    print(f"Expert parallel size: {args.expert_model_parallel_size}")
    if hasattr(args, 'num_experts'):
        print(f"Number of MoE experts: {args.num_experts}")
    if hasattr(args, 'moe_layer_freq'):
        print(f"MoE layer frequency: {args.moe_layer_freq}")
    print("=" * 80)
    print(f"FLOPs per micro batch: {flops_per_step:.6e}")
    print(f"FLOPs per global batch: {flops_per_step * args.global_batch_size / args.micro_batch_size:.6e}")
    print("=" * 80)
