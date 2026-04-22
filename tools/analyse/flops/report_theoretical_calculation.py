# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical GEMM flops for model training without instantiating
a model and running training iterations on GPU(s)."""

import json
import os
import sys
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from tools.analyse.flops.flops_estimator import num_floating_point_operations_details


def logs(args, configs, results, total):
    print("=" * 80)
    print("THEORETICAL GEMM FLOATING POINT OPERATIONS (TFLOPs)")
    print("=" * 80)
    for key, value in configs.items():
        print(f"{key}: {value}")
    print("=" * 80)
    print(f"TFLOPs per micro batch: {total / 1e12:.3f}")
    print(f"TFLOPs per global batch: {total / 1e12 * args.global_batch_size / args.micro_batch_size:.3f}")
    print("=" * 80)
    print("\nDetailed TFLOPs Breakdown:")
    for term in results:
        print(f"{term}:")
        print(f"    layers: {results[term]['layers']}")
        print(f"    forward: {results[term]['forward'] / 1e12:.3f}")
        print(f"    backward: {results[term]['backward'] / 1e12:.3f}")
        print(f"    total: {results[term]['total'] / 1e12:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    assert len(sys.argv) > 2
    output_dir = sys.argv[1]
    model_name = sys.argv[2]
    sys.argv = [sys.argv[0]] + sys.argv[3:]

    initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)

    args = get_args()
    results, total = num_floating_point_operations_details(args, args.micro_batch_size)

    # Prepare result dictionary for JSON output
    configs = {
        'micro_batch_size': args.micro_batch_size,
        'global_batch_size': args.global_batch_size,
        'seq_length': args.seq_length,
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'ffn_hidden_size': args.ffn_hidden_size,
        'num_attention_heads': args.num_attention_heads,
    }
    # Add optional fields
    if hasattr(args, 'num_experts'):
        configs['num_experts'] = args.num_experts
    if hasattr(args, 'moe_layer_freq'):
        configs['moe_layer_freq'] = args.moe_layer_freq
    logs(args, configs, results, total)


    output_dir = os.path.join(output_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON file
    json_path = os.path.join(output_dir, f"{model_name}.config.json")
    with open(json_path, 'w') as f:
        json.dump(configs, f, indent=2)
    print(f"\nconfiguration saved to: {json_path}")

    json_path = os.path.join(output_dir, f"{model_name}.flops.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFLOPs saved to: {json_path}")
