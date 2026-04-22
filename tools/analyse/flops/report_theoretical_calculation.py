# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical GEMM flops for model training without instantiating
a model and running training iterations on GPU(s)."""

import json
import os
import sys
import pandas as pd
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from .training import num_floating_point_operations


def flops_to_tflops(flops):
    """Convert FLOPs to TFLOPs."""
    return flops / 1e12


def flops_breakdown_to_tflops(flops_breakdown):
    """Convert FLOPs breakdown to TFLOPs."""
    return {key: flops_to_tflops(value) for key, value in flops_breakdown.items()}


def save_to_excel(result, output_path):
    """Save result to Excel file."""
    # Create DataFrame for config
    config_df = pd.DataFrame(list(result['config'].items()), columns=['Parameter', 'Value'])

    # Create DataFrame for TFLOPs breakdown
    tflops_breakdown = flops_breakdown_to_tflops(result['flops_breakdown'])
    tflops_df = pd.DataFrame([
        [key, value] for key, value in tflops_breakdown.items()
    ], columns=['Component', 'TFLOPs'])

    # Create DataFrame for summary
    tflops_per_micro_batch = flops_to_tflops(result['flops_per_micro_batch'])
    tflops_per_global_batch = flops_to_tflops(result['flops_per_global_batch'])
    summary_df = pd.DataFrame([
        ['TFLOPs per micro batch', tflops_per_micro_batch],
        ['TFLOPs per global batch', tflops_per_global_batch]
    ], columns=['Metric', 'TFLOPs'])

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output_path) as writer:
        config_df.to_excel(writer, sheet_name='Config', index=False)
        tflops_df.to_excel(writer, sheet_name='TFLOPs Breakdown', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

if __name__ == "__main__":
    # Check if output path is provided as positional argument
    assert len(sys.argv) > 2
    output_dir = sys.argv[1]
    model_name = sys.argv[2]
    sys.argv = [sys.argv[0]] + sys.argv[3:]
    
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(output_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)
    args = get_args()

    # Calculate flops per training step (per micro batch)
    # Note: This is flops per micro batch, not per global batch
    flops_breakdown = num_floating_point_operations(args, args.micro_batch_size, return_breakdown=True)
    tflops_breakdown = flops_breakdown_to_tflops(flops_breakdown)

    print("=" * 80)
    print("THEORETICAL GEMM FLOATING POINT OPERATIONS (TFLOPs)")
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

    flops_per_step = flops_breakdown['total']
    tflops_per_step = tflops_breakdown['total']
    tflops_per_global_batch = tflops_per_step * args.global_batch_size / args.micro_batch_size
    print(f"TFLOPs per micro batch: {tflops_per_step:.6e}")
    print(f"TFLOPs per global batch: {tflops_per_global_batch:.6e}")
    print("=" * 80)
    print("\nTFLOPs Breakdown:")
    for key, value in tflops_breakdown.items():
        print(f"  {key}: {value:.6e}")
    print("=" * 80)

    # Prepare result dictionary for JSON output
    result = {
        'config': {
            'micro_batch_size': args.micro_batch_size,
            'global_batch_size': args.global_batch_size,
            'seq_length': args.seq_length,
            'num_layers': args.num_layers,
            'hidden_size': args.hidden_size,
            'ffn_hidden_size': args.ffn_hidden_size,
            'num_attention_heads': args.num_attention_heads,
            'pipeline_model_parallel_size': args.pipeline_model_parallel_size,
            'tensor_model_parallel_size': args.tensor_model_parallel_size,
            'expert_model_parallel_size': args.expert_model_parallel_size,
        },
        'flops_breakdown': flops_breakdown,
        'tflops_breakdown': tflops_breakdown,
        'flops_per_micro_batch': flops_per_step,
        'flops_per_global_batch': flops_per_step * args.global_batch_size / args.micro_batch_size,
        'tflops_per_micro_batch': tflops_per_step,
        'tflops_per_global_batch': tflops_per_global_batch,
    }

    # Add optional fields
    if hasattr(args, 'num_experts'):
        result['config']['num_experts'] = args.num_experts
    if hasattr(args, 'moe_layer_freq'):
        result['config']['moe_layer_freq'] = args.moe_layer_freq

    # Save to JSON file
    json_path = os.path.join(output_dir, f"{model_name}_flops.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nFLOPs breakdown saved to: {json_path}")

    # Save to Excel file
    excel_path = os.path.join(output_dir, f"{model_name}_flops.xlsx")
    save_to_excel(result, excel_path)
    print(f"TFLOPs breakdown saved to: {excel_path}")
