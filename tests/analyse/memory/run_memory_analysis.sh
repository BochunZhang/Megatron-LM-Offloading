#!/bin/bash

# DeepSeek-V3 Memory Analysis Script
#
# This script runs theoretical memory analysis for DeepSeek-V3 model
# with different parallel configurations.
#
# Usage: bash run_memory_analysis.sh

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"

# Path to the memory analysis script
MEMORY_SCRIPT="$PROJECT_ROOT/tools/run_theoretical_memory.py"

# Check if the script exists
if [ ! -f "$MEMORY_SCRIPT" ]; then
    echo "Error: Memory analysis script not found at $MEMORY_SCRIPT"
    exit 1
fi

echo "===================================================================================================="
echo "DeepSeek-V3 Theoretical Memory Analysis"
echo "===================================================================================================="

# Function to run memory analysis for a given configuration
run_analysis() {
    local TP=$1
    local PP=$2
    local VPP=$3
    local DP=$4
    local EP=$5
    local CASE_NAME=$6

    echo ""
    echo "===================================================================================================="
    echo "Running $CASE_NAME"
    echo "===================================================================================================="
    echo "Configuration:"
    echo "  Tensor Parallel (TP):            $TP"
    echo "  Pipeline Parallel (PP):          $PP"
    echo "  Virtual Pipeline Parallel (VPP):  ${VPP:-N/A}"
    echo "  Data Parallel (DP):              $DP"
    echo "  Expert Parallel (EP):            $EP"
    echo ""

    # Build the command
    CMD="python3 \"$MEMORY_SCRIPT\""
    CMD="$CMD --tensor-model-parallel-size $TP"
    CMD="$CMD --pipeline-model-parallel-size $PP"
    CMD="$CMD --data-parallel-size $DP"
    CMD="$CMD --expert-model-parallel-size $EP"

    # Add VPP if provided
    if [ -n "$VPP" ]; then
        CMD="$CMD --virtual-pipeline-model-parallel-size $VPP"
    fi

    # Add model-specific parameters
    CMD="$CMD --num-layers 61"
    CMD="$CMD --hidden-size 7168"
    CMD="$CMD --ffn-hidden-size 18432"
    CMD="$CMD --num-attention-heads 128"
    CMD="$CMD --kv-channels 128"
    CMD="$CMD --seq-length 4096"
    CMD="$CMD --padded-vocab-size 320128"
    CMD="$CMD --num-experts 256"
    CMD="$CMD --moe-ffn-hidden-size 2048"
    CMD="$CMD --moe-shared-expert-intermediate-size 2048"
    CMD="$CMD --moe-layer-freq '([0]*3+[1]*58)'"
    CMD="$CMD --multi-latent-attention"
    CMD="$CMD --q-lora-rank 1536"
    CMD="$CMD --kv-lora-rank 512"
    CMD="$CMD --qk-head-dim 128"
    CMD="$CMD --qk-pos-emb-head-dim 64"
    CMD="$CMD --v-head-dim 128"
    CMD="$CMD --micro-batch-size 1"
    CMD="$CMD --global-batch-size 64"
    CMD="$CMD --verbose"
    CMD="$CMD --detailed"

    # Execute the command
    eval $CMD

    echo ""
    echo "Finished $CASE_NAME"
    echo "===================================================================================================="
}

# ========================================
# Test Case 1: 256 GPUs configuration
# ========================================
# From: https://github.com/NVIDIA/Megatron-LM/blob/eb0783b6d35607ef1953eaca60b37b886b1a25d0/docs/discussions/deepseek-v3-gb200-optimization/deepseek-v3-gb200-reproduce-guide.md
# Configuration: 256 GPUs, PP=8, VPP=4, DP=EP=32
run_analysis \
    1 \
    8 \
    4 \
    32 \
    32 \
    "Case 1: 256 GPUs (TP=1, PP=8, VPP=4, DP=32, EP=32)"

# ========================================
# Test Case 2: 64 GPUs configuration
# ========================================
# Configuration: 64 GPUs, DP=64, EP=32
run_analysis \
    1 \
    1 \
    "" \
    64 \
    32 \
    "Case 2: 64 GPUs (TP=1, PP=1, DP=64, EP=32)"

echo ""
echo "===================================================================================================="
echo "All test cases completed!"
echo "===================================================================================================="
