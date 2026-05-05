#!/usr/bin/env python3
"""
MXFP8 Linear Layer GEMM Test for Blackwell

This script demonstrates how to use TransformerEngine's Linear layer with MXFP8
quantization for training. MXFP8 uses blockwise scaling (32 elements per block)
with E8M0 format (power-of-2 scaling factors).

Requirements:
- Blackwell GPU (compute capability >= 10.0)
- TransformerEngine installed
- PyTorch 2.1+

Usage:
    python mxfp8_linear_test.py --batch_size 1 --seq_len 4096 --hidden_size 7168 --out_features 1532 --iterations 10"
    python mxfp8_linear_test.py --batch_size 1 --seq_len 4096 --hidden_size 7168 --out_features 1532 --iterations 10 --graph"
    python mxfp8_linear_test.py --batch_size 1 --seq_len 4096 --hidden_size 7168 --out_features 1532 --iterations 10 --backward"
    python mxfp8_linear_test.py --batch_size 1 --seq_len 4096 --hidden_size 7168 --out_features 1532 --iterations 10 --graph --backward"
"""

import argparse
from typing import Optional
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling, Format
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast
import pandas as pd


def check_mxfp8_support():
    """Check if MXFP8 is supported on the current device."""
    mxfp8_available, reason = FP8GlobalStateManager.is_mxfp8_available()
    if not mxfp8_available:
        print(f"MXFP8 not supported: {reason}")
        return False
    print("MXFP8 is supported on this device!")
    return True


def test_mxfp8_linear(
    name: str = "gemm",
    norm: Optional[str] = None,
    batch_size: int = 1,
    seq_len: int = 4096,
    hidden_size: int = 7168,
    out_features: int = 1536,
    num_iterations: int = 10,
    graph: bool = False,
    backward: bool = False,
    learning_rate: float = 1e-3,
):
    """
    Test MXFP8 Linear layer forward pass.

    Parameters
    ----------
    batch_size : int
        Batch size
    seq_len : int
        Sequence length
    hidden_size : int
        Input dimension (must be divisible by 32 for MXFP8)
    out_features : int
        Output dimension (must be divisible by 32 for MXFP8, defaults to hidden_size)
    num_iterations : int
        Number of warmup iterations
    """

    # Ensure dimensions are divisible by 32 for MXFP8
    assert hidden_size % 32 == 0, "hidden_size must be divisible by 32 for MXFP8"
    assert out_features % 32 == 0, "out_features must be divisible by 32 for MXFP8"

    # Create Linear layer with MXFP8
    if backward == False:
        recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
    else:
        recipe = MXFP8BlockScaling(fp8_format=Format.HYBRID)

    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        if norm == None:
            linear = te.Linear(
                in_features=hidden_size,
                out_features=out_features,
                bias=False,
                name=name,
                params_dtype=torch.bfloat16,  # Match input dtype
            )
        else:
            linear = te.LayerNormLinear(
                in_features=hidden_size,
                out_features=out_features,
                normalization=norm,
                bias=False,
                name=name,
                params_dtype=torch.bfloat16,  # Match input dtype
            )

    # Move to GPU
    linear = linear.cuda()

    # Create random input
    # Shape: [seq_len, batch_size, hidden_size] (s, b, h)
    x = torch.randn(
        seq_len, 
        batch_size, 
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda"
    )

    print(f"\n{'='*60}")
    print(f"MXFP8 Linear Layer Forward Pass Test")
    print(f"{'='*60}")
    print(f"Input shape: {x.shape} [{x.dtype}]")
    print(f"Weight shape: {linear.weight.shape} [{linear.weight.dtype}")
    print(f"Recipe: {recipe}")
    
    if backward == True:
        # Simple MSE loss for demonstration
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(linear.parameters(), lr=learning_rate)

        t = torch.randn(
            seq_len, 
            batch_size, 
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda"
        )

    if graph == True:
        from transformer_engine.pytorch.graph import make_graphed_callables
        linear = make_graphed_callables(
            linear,
            (x,),
        )

    # Warmup iterations
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        for _ in range(num_iterations):
            y = linear(x)
            if backward == True:
                loss = criterion(y, t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    torch.cuda.synchronize()

    print("End!")

# def test_mxfp8_atten(
#     name: str = "atten",
#     attn: bool = False,
#     norm: bool = False,
#     batch_size: int = 1,
#     seq_len: int = 4096,
#     hidden_size: int = 7168,
#     out_features: int = 1536,
#     atten_head: int = 128,
#     kv_channels: int = 192,
#     num_iterations: int = 10,
#     graph: bool = False,
#     backward: bool = False,
#     learning_rate: float = 1e-3,
# ):
#     """
#     Test MXFP8 Linear layer forward pass.

#     Parameters
#     ----------
#     batch_size : int
#         Batch size
#     seq_len : int
#         Sequence length
#     hidden_size : int
#         Input dimension (must be divisible by 32 for MXFP8)
#     out_features : int
#         Output dimension (must be divisible by 32 for MXFP8, defaults to hidden_size)
#     num_iterations : int
#         Number of warmup iterations
#     """

#     # Ensure dimensions are divisible by 32 for MXFP8
#     assert hidden_size % 32 == 0, "hidden_size must be divisible by 32 for MXFP8"
#     assert out_features % 32 == 0, "out_features must be divisible by 32 for MXFP8"

#     # Create Linear layer with MXFP8
#     if backward == False:
#         recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
#     else:
#         recipe = MXFP8BlockScaling(fp8_format=Format.HYBRID)

#     with fp8_autocast(enabled=True, fp8_recipe=recipe):
#         if attn == True:
#             gemm = te.DotProductAttention(
#                 num_attention_heads=atten_head,
#                 kv_channels=kv_channels,
#                 attention_dropout=0.0,
#                 attn_mask_type="causal",
#                 qkv_format = "sbhd",
#             )

#         elif attn == False:
#             gemm = te.Linear(
#                 in_features=hidden_size,
#                 out_features=out_features,
#                 bias=False,
#                 name=name,
#                 params_dtype=torch.bfloat16,  # Match input dtype
#             )
#         else:
#             gemm = te.LayerNormLinear(
#                 in_features=hidden_size,
#                 out_features=out_features,
#                 bias=False,
#                 name=name,
#                 params_dtype=torch.bfloat16,  # Match input dtype
#             )

#     # Move to GPU
#     gemm = gemm.cuda()

#     # Create random input
#     # Shape: [seq_len, batch_size, hidden_size] (s, b, h)
#     x = torch.randn(
#         seq_len, 
#         batch_size, 
#         hidden_size,
#         dtype=torch.bfloat16,
#         device="cuda"
#     )

#     print(f"\n{'='*60}")
#     print(f"MXFP8 Linear Layer Forward Pass Test")
#     print(f"{'='*60}")
#     print(f"Input shape: {x.shape} [{x.dtype}]")
#     print(f"Weight shape: {linear.weight.shape} [{linear.weight.dtype}")
#     print(f"Recipe: {recipe}")
    
#     if backward == True:
#         # Simple MSE loss for demonstration
#         criterion = torch.nn.MSELoss()
#         optimizer = torch.optim.AdamW(linear.parameters(), lr=learning_rate)

#         t = torch.randn(
#             seq_len, 
#             batch_size, 
#             hidden_size,
#             dtype=torch.bfloat16,
#             device="cuda"
#         )

#     if graph == True:
#         from transformer_engine.pytorch.graph import make_graphed_callables
#         linear = make_graphed_callables(
#             linear,
#             (x,),
#         )

#     # Warmup iterations
#     with fp8_autocast(enabled=True, fp8_recipe=recipe):
#         for _ in range(num_iterations):
#             y = linear(x)
#             if backward == True:
#                 loss = criterion(y, t)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#     torch.cuda.synchronize()

#     print("End!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MXFP8 Linear Layer GEMM Test for Blackwell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mxfp8_linear_test.py --batch_size 1 --seq_len 4096 --hidden_size 7168 --out_features 1532 --iterations 10"
  python mxfp8_linear_test.py --batch_size 1 --seq_len 4096 --hidden_size 7168 --out_features 1532 --iterations 10 --graph"
  python mxfp8_linear_test.py --batch_size 1 --seq_len 4096 --hidden_size 7168 --out_features 1532 --iterations 10 --backward"
  python mxfp8_linear_test.py --batch_size 1 --seq_len 4096 --hidden_size 7168 --out_features 1532 --iterations 10 --graph --backward"
""",
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default='gemm',
        help="GEMM name, e.g. linear_q_down_proj",
    )
    parser.add_argument(
        "--operator",
        type=str,
        default='linear',
        help='chose from ["linear", "norm_linear", "attention"]'
    )
    parser.add_argument(
        "--norm",
        type=str,
        default='linear',
        help='chose from ["linear", "norm_linear", "attention"]'
    )
    parser.add_argument(
        "--backward",
        action='store_true',
        help="enable backward test",
    )
    parser.add_argument(
        "--graph",
        action='store_true',
        help="enable backward test)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="Sequence length (default: 4096)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=7168,
        help="Hidden size (must be divisible by 32 for MXFP8, default: 7168)",
    )
    parser.add_argument(
        "--out_features",
        type=int,
        default=1536,
        help="Output dimension (must be divisible by 32 for MXFP8, defaults to hidden_size)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of warmup iterations for forward test (default: 10)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for training (default: 1e-3)",
    )


    args = parser.parse_args()

    # Check GPU and MXFP8 support
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This test requires a GPU.")

    device_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    print(f"Device: {device_name}")
    print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")

    if not check_mxfp8_support():
        print("\nMXFP8 requires Blackwell GPU (compute capability >= 10.0)")
        print("Falling back to regular FP8 if available...")
        exit(1)

    # Validate hidden_size
    if args.hidden_size % 32 != 0:
        raise ValueError(f"hidden_size must be divisible by 32 for MXFP8, got {args.hidden_size}")

    # Validate out_features if specified
    if args.out_features is not None and args.out_features % 32 != 0:
        raise ValueError(f"out_features must be divisible by 32 for MXFP8, got {args.out_features}")

    # Run test
    if args.operator == 'linear':
        test_mxfp8_linear(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            out_features=args.out_features,
            name=args.name,
            num_iterations=args.iterations,
            graph=args.graph,
            backward=args.backward,
            learning_rate=args.learning_rate,
        )
    elif args.operator == 'norm_linear':
        test_mxfp8_linear(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            out_features=args.out_features,
            name=args.name,
            norm='RMSNorm',
            num_iterations=args.iterations,
            graph=args.graph,
            backward=args.backward,
            learning_rate=args.learning_rate,
        )

    print(f"\n{'='*60}")
    print(f"MXFP8 test(s) completed successfully!")
    print(f"{'='*60}")