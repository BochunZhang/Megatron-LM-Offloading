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
    python mxfp8_linear_test.py --case forward --batch_size 16 --seq_len 128 --hidden_size 768
    python mxfp8_linear_test.py --case training --batch_size 16 --seq_len 128 --hidden_size 768 --num_epochs 5
    python mxfp8_linear_test.py --case graph --batch_size 16 --seq_len 128 --hidden_size 768
"""

import argparse
from typing import Optional
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling, Format
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast
import pandas as pd

def compute_tflops(batch_size, seq_len, hidden_size, elapsed_time_ms, is_training=False):
    """
    Compute TFLOPS for GEMM operations.

    For Linear layer y = x @ W^T:
    - FLOPs = 2 * batch_size * seq_len * hidden_size * hidden_size
    - Factor of 2 accounts for multiply and add operations
    - For training, multiply by 3 (forward + dgrad + wgrad)
    """
    flops = 2 * batch_size * seq_len * hidden_size * hidden_size
    if is_training:
        flops *= 3  # Forward + backward (dgrad + wgrad)

    tflops = flops / (elapsed_time_ms * 1e-3) / 1e12
    return tflops

def check_mxfp8_support():
    """Check if MXFP8 is supported on the current device."""
    mxfp8_available, reason = FP8GlobalStateManager.is_mxfp8_available()
    if not mxfp8_available:
        print(f"MXFP8 not supported: {reason}")
        return False
    print("MXFP8 is supported on this device!")
    return True

def create_linear_layer_with_mxfp8(
    in_features: int,
    out_features: int,
    fp8_format: Format = Format.E4M3,
):
    """
    Create a Linear layer configured for MXFP8 quantization.

    Parameters
    ----------
    in_features : int
        Input dimension (must be divisible by 32 for MXFP8)
    out_features : int
        Output dimension (must be divisible by 32 for MXFP8)
    fp8_format : Format
        FP8 format (E4M3 for forward, or HYBRID for E4M3 forward + E5M2 backward)
    """
    # MXFP8 recipe with blockwise scaling
    recipe = MXFP8BlockScaling(fp8_format=fp8_format)

    # Create Linear layer within FP8 autocast context
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        linear = te.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            params_dtype=torch.bfloat16,  # Match input dtype
        )

    return linear, recipe

def test_mxfp8_linear_forward(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    out_features: Optional[int] = None,
    num_iterations: int = 10,
    results=None,
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
    out_features : Optional[int]
        Output dimension (must be divisible by 32 for MXFP8, defaults to hidden_size)
    num_iterations : int
        Number of warmup iterations
    """
    # Default to hidden_size if not specified
    if out_features is None:
        out_features = hidden_size

    # Ensure dimensions are divisible by 32 for MXFP8
    assert hidden_size % 32 == 0, "hidden_size must be divisible by 32 for MXFP8"
    assert out_features % 32 == 0, "out_features must be divisible by 32 for MXFP8"

    # Create Linear layer with MXFP8
    recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)

    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        linear = te.Linear(
            in_features=hidden_size,
            out_features=out_features,
            bias=True,
            params_dtype=torch.bfloat16,  # Match input dtype
        )

    # Move to GPU
    linear = linear.cuda()

    # Create random input
    # Shape: [batch_size * seq_len, hidden_size]
    x = torch.randn(batch_size * seq_len, hidden_size,
                  dtype=torch.bfloat16,
                  device="cuda",
                  requires_grad=True)

    print(f"\n{'='*60}")
    print(f"MXFP8 Linear Layer Forward Pass Test")
    print(f"{'='*60}")
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {linear.weight.shape}")
    print(f"Recipe: {recipe}")

    # Warmup iterations
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        for _ in range(num_iterations):
            y = linear(x)

    torch.cuda.synchronize()

    # Timed forward pass
    import time
    start_time = time.time()

    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        y = linear(x)

    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000

    # Compute TFLOPS
    tflops = compute_tflops(batch_size, seq_len, hidden_size, elapsed_ms, is_training=False)

    print(f"Forward pass time: {elapsed_ms:.4f} ms")
    print(f"TFLOPS: {tflops:.4f}")
    print(f"Output shape: {y.shape}")
    print(f"Output dtype: {y.dtype}")

    # Store results
    if results is not None:
        results["test"] = "forward"
        results["batch_size"] = batch_size
        results["seq_len"] = seq_len
        results["hidden_size"] = hidden_size
        results["out_features"] = out_features
        results["time_ms"] = elapsed_ms
        results["tflops"] = tflops

    return y

def test_mxfp8_linear_training(
    batch_size: int = 16,
    seq_len: int = 128,
    hidden_size: int = 768,
    out_features: Optional[int] = None,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    results=None,
):
    """
    Test MXFP8 Linear layer with full training loop (forward + backward).

    This demonstrates MXFP8 for training scenarios with gradient computation.
    """
    # Default to hidden_size if not specified
    if out_features is None:
        out_features = hidden_size

    # Ensure dimensions are divisible by 32 for MXFP8
    assert hidden_size % 32 == 0, "hidden_size must be divisible by 32 for MXFP8"
    assert out_features % 32 == 0, "out_features must be divisible by 32 for MXFP8"

    print(f"\n{'='*60}")
    print(f"MXFP8 Linear Layer Training Test")
    print(f"{'='*60}")

    # Create Linear layer with MXFP8
    # Use HYBRID format: E4M3 for forward, E5M2 for backward
    recipe = MXFP8BlockScaling(fp8_format=Format.HYBRID)

    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        linear = te.Linear(
            in_features=hidden_size,
            out_features=out_features,
            bias=True,
            params_dtype=torch.bfloat16,  # Match input dtype
        )

    # Move to GPU
    linear = linear.cuda()

    # Simple MSE loss for demonstration
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(linear.parameters(), lr=learning_rate)

    # Training data
    # Shape: [batch_size, seq_len, hidden_size]
    x_train = torch.randn(batch_size, seq_len, hidden_size,
                        dtype=torch.bfloat16,
                        device="cuda")
    y_target = torch.randn(batch_size, seq_len, out_features,
                        dtype=torch.bfloat16,
                        device="cuda")

    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Input size (hidden_size): {hidden_size}")
    print(f"Output size (out_features): {out_features}")
    print(f"Learning rate: {learning_rate}")
    print(f"Recipe: {recipe}")
    print(f"\nTraining...")

    import time
    total_time = 0.0

    # Training loop
    for epoch in range(num_epochs):
        # Flatten input: [batch_size * seq_len, hidden_size]
        x_flat = x_train.view(-1, hidden_size)
        y_flat = y_target.view(-1, out_features)

        # Timing
        torch.cuda.synchronize()
        start_time = time.time()

        # Forward pass with MXFP8
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            y_pred = linear(x_flat)
            loss = criterion(y_pred, y_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed_ms = (time.time() - start_time) * 1000
        total_time += elapsed_ms

        # Compute TFLOPS for this iteration
        tflops = compute_tflops(batch_size, seq_len, hidden_size, elapsed_ms, is_training=True)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}, Time: {elapsed_ms:.4f} ms, TFLOPS: {tflops:.4f}")

    # Average statistics
    avg_time_ms = total_time / num_epochs
    avg_tflops = compute_tflops(batch_size, seq_len, hidden_size, avg_time_ms, is_training=True)

    print(f"\nTraining completed!")
    print(f"Average time per epoch: {avg_time_ms:.4f} ms")
    print(f"Average TFLOPS: {avg_tflops:.4f}")
    print(f"Final loss: {loss.item():.6f}")
    print(f"Weight norm: {linear.weight.norm().item():.6f}")

    # Store results
    if results is not None:
        results["test"] = "training"
        results["batch_size"] = batch_size
        results["seq_len"] = seq_len
        results["hidden_size"] = hidden_size
        results["out_features"] = out_features
        results["num_epochs"] = num_epochs
        results["avg_time_ms"] = avg_time_ms
        results["avg_tflops"] = avg_tflops
        results["final_loss"] = loss.item()

def test_mxfp8_with_captured_graphs(
    batch_size: int = 16,
    seq_len: int = 128,
    hidden_size: int = 768,
    results=None,
):
    """
    Test MXFP8 with CUDA Graph capture for inference.
    CUDA Graphs can significantly improve latency by reducing CPU overhead.
    """
    from transformer_engine.pytorch.graph import make_graphed_callables

    # Ensure dimensions are divisible by 32 for MXFP8
    assert hidden_size % 32 == 0, "hidden_size must be divisible by 32 for MXFP8"

    print(f"\n{'='*60}")
    print(f"MXFP8 Linear Layer with CUDA Graphs")
    print(f"{'='*60}")

    # Create Linear layer
    recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)

    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        linear = te.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,  # Match input dtype
        )

    linear = linear.cuda()

    # Create static input for graph capture
    x_static = torch.randn(batch_size * seq_len, hidden_size,
                        dtype=torch.bfloat16,
                        device="cuda")

    # Capture CUDA Graph
    graphed_linear = make_graphed_callables(
        lambda x: linear(x),
        [x_static],
    )

    # Run with captured graph
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        y = graphed_linear(x_static)

    # Benchmark
    import time
    num_iterations = 100

    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            y = graphed_linear(x_static)

    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000
    avg_time_ms = elapsed_ms / num_iterations

    # Compute TFLOPS
    tflops = compute_tflops(batch_size, seq_len, hidden_size, avg_time_ms, is_training=False)

    print(f"Input shape: {x_static.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Average time per iteration: {avg_time_ms:.4f} ms")
    print(f"TFLOPS: {tflops:.4f}")
    print(f"CUDA Graph capture successful!")

    # Store results
    if results is not None:
        results["test"] = "graph"
        results["batch_size"] = batch_size
        results["seq_len"] = seq_len
        results["hidden_size"] = hidden_size
        results["avg_time_ms"] = avg_time_ms
        results["tflops"] = tflops
        results["num_iterations"] = num_iterations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MXFP8 Linear Layer GEMM Test for Blackwell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mxfp8_linear_test.py --case forward --batch_size 16 --seq_len 128 --hidden_size 768
  python mxfp8_linear_test.py --case training --batch_size 16 --seq_len 128 --hidden_size 768 --num_epochs 5
  python mxfp8_linear_test.py --case graph --batch_size 16 --seq_len 128 --hidden_size 768
        """,
    )

    parser.add_argument(
        "--case",
        type=str,
        choices=["forward", "training", "graph", "all"],
        default="all",
        help="Test case to run: 'forward', 'training', 'graph', or 'all' (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length (default: 128)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Hidden size (must be divisible by 32 for MXFP8, default: 768)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of warmup iterations for forward test (default: 10)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for training (default: 1e-3)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for Excel results file",
    )
    parser.add_argument(
        "--out-features",
        type=int,
        default=None,
        help="Output dimension (must be divisible by 32 for MXFP8, defaults to hidden_size)",
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

    # Collect results for Excel output
    all_results = []

    # Run selected test(s)
    if args.case == "forward" or args.case == "all":
        result = {}
        test_mxfp8_linear_forward(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            out_features=args.out_features,
            num_iterations=args.iterations,
            results=result,
        )
        all_results.append(result)

    if args.case == "training" or args.case == "all":
        result = {}
        test_mxfp8_linear_training(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            out_features=args.out_features,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            results=result,
        )
        all_results.append(result)

    if args.case == "graph" or args.case == "all":
        result = {}
        test_mxfp8_with_captured_graphs(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            results=result,
        )
        all_results.append(result)

    # Save results to Excel if output path is specified
    if args.output_path and all_results:
        import os
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        df = pd.DataFrame(all_results)
        df.to_excel(args.output_path, index=False, engine='openpyxl')
        print(f"\nResults saved to: {args.output_path}")

    print(f"\n{'='*60}")
    print(f"MXFP8 test(s) completed successfully!")
    print(f"{'='*60}")