"""Example script to compare offloaded tensors from two runs.

Usage:
    python compare_offload_tensors.py --dir1 /path/to/run1/offload-tensor --dir2 /path/to/run2/offload-tensor
    python compare_offload_tensors.py --file1 /path/to/tensor1.pt --file2 /path/to/tensor2.pt
"""

import os
import sys
import argparse

import torch

def compare_tensors(tensor_a: torch.Tensor, tensor_b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8):
    """
    Compare two tensors for equality.

    Comparison order:
    1. numel (total number of elements)
    2. shape (dimensions)
    3. values (element-wise comparison)

    If shapes differ but numel matches, reshapes before comparing values.

    Args:
        tensor_a: First tensor
        tensor_b: Second tensor
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison

    Returns:
        dict with keys:
            - numel_match: bool
            - shape_match: bool
            - value_match: bool
            - message: str description of result
    """

    message = ""
    message += f"tensor_a: dtype={tensor_a.dtype}, shape={list(tensor_a.shape)}, numel={tensor_a.numel()};\n"
    message += f"tensor_b: dtype={tensor_b.dtype}, shape={list(tensor_b.shape)}, numel={tensor_b.numel()};\n"

    # 1. Compare numel
    numel_a = tensor_a.numel()
    numel_b = tensor_b.numel()
    if numel_a != numel_b:
        message += f"numel mismatch!\n"
        return message
    
    # 2. Compare dtype
    if tensor_a.dtype != tensor_b.dtype:
        message += f"dtype mismatch!\n"
        return message

    # 3. Compare shape
    shape_a = list(tensor_a.shape)
    shape_b = list(tensor_b.shape)
    shapes_same = shape_a == shape_b
    if shapes_same:
        message += f"shapes match.\n"
    else:
        message += f"shape mismatch, but numel matches, flattened to compare.\n"

    # 4. Compare values (reshape if needed)
    if shapes_same:
        a_view = tensor_a
        b_view = tensor_b
    else:
        a_view = tensor_a.view(-1)
        b_view = tensor_b.view(-1)

    value_match = torch.equal(a_view, b_view)
    if value_match:
        message += f"values equal!\n"
        return message

    value_match = torch.allclose(a_view, b_view, rtol=rtol, atol=atol)
    if value_match:
        message += f"values allclose(rtol={rtol}, atol={atol})!\n"
    else:
        message += f"values differ!\n"
    
    return message


def compare_tensor_files(file_path_a: str, file_path_b: str, **kwargs):
    """
    Load and compare two tensor files.

    Args:
        file_path_a: Path to first tensor file (.pt)
        file_path_b: Path to second tensor file (.pt)
        **kwargs: Passed to compare_tensors

    Returns:
        dict with comparison results
    """
    tensor_a = torch.load(file_path_a, map_location='cuda:0', weights_only=True)
    tensor_b = torch.load(file_path_b, map_location='cuda:0', weights_only=True)
    return compare_tensors(tensor_a, tensor_b, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Compare offloaded tensors")
    parser.add_argument("--file1", type=str, help="First tensor file")
    parser.add_argument("--file2", type=str, help="Second tensor file")

    args = parser.parse_args()

    if args.file1 and args.file2:
        result = compare_tensor_files(args.file1, args.file2)
        print(result)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
