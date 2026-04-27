#!/bin/bash
# GEMM Performance Testing Script
#
# Usage:
#   ./run_gemm_test.sh --recipe mxfp8 --test batch --profile
#   ./run_gemm_test.sh --recipe mxfp8 --test forward

set -e

# Default values
RECIPE=""
TEST=""
PROFILE=false
OUTPUT_DIR="tests/gemm/results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --recipe)
            RECIPE="$2"
            shift 2
            ;;
        --test)
            TEST="$2"
            shift 2
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --recipe <recipe> --test <test> [--profile]"
            echo ""
            echo "Options:"
            echo "  --recipe    Recipe type (e.g., mxfp8)"
            echo "  --test      Test type: 'batch' (batch size sweep) or single test name (forward, training, graph)"
            echo "  --profile   Enable Nsight Systems profiling"
            echo ""
            echo "Examples:"
            echo "  $0 --recipe mxfp8 --test batch --profile"
            echo "  $0 --recipe mxfp8 --test forward"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$RECIPE" ]]; then
    echo "Error: --recipe is required"
    exit 1
fi

if [[ -z "$TEST" ]]; then
    echo "Error: --test is required"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Python script path
PY_SCRIPT="tests/gemm/mxfp8_gemm.py"

# Check if python script exists
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "Error: Python script not found at $PY_SCRIPT"
    exit 1
fi

# Output base path
OUTPUT_BASE="$OUTPUT_DIR/${RECIPE}.${TEST}"

if [[ "$RECIPE" == "mxfp8" ]]; then
    if [[ "$TEST" == "batch" ]]; then
        # Batch size sweep test
        echo "Running batch size sweep for MXFP8 GEMM..."
        echo "Configuration: hidden_size=7168, out_features=1536, seq_len=4096"
        echo ""

        # Define batch sizes
        BATCH_SIZES=(1 2 4 8 16)

        # Create results file for batch test
        BATCH_RESULTS=()

        for bs in "${BATCH_SIZES[@]}"; do
            echo "Testing with batch_size=$bs"
            echo "----------------------------------------"

            # Output file for this batch size
            OUTPUT_FILE="${OUTPUT_BASE}.bs${bs}.xlsx"

            # Build command
            CMD="python $PY_SCRIPT \n                --case forward \n                --batch_size $bs \n                --seq_len 4096 \n                --hidden_size 7168 \n                --out-features 1536 \n                --output-path $OUTPUT_FILE"

            # Add profiling if enabled
            if [[ "$PROFILE" == true ]]; then
                echo "Running with Nsight Systems profiling..."
                NSYS_OUTPUT="${OUTPUT_BASE}.bs${bs}.nsys-rep"
                nsys profile \n                    --trace=cuda,nvtx \n                    --force-overwrite=true \n                    --output="$NSYS_OUTPUT" \n                    $CMD
            else
                $CMD
            fi

            echo ""
        done

        echo "Batch size sweep completed!"
        echo "Results saved to: $OUTPUT_BASE.bs*.xlsx"
        if [[ "$PROFILE" == true ]]; then
            echo "Profiles saved to: $OUTPUT_BASE.bs*.nsys-rep"
        fi

    else
        # Single test
        echo "Running single test: $TEST"
        echo ""

        OUTPUT_FILE="${OUTPUT_BASE}.xlsx"

        # Build command
        CMD="python $PY_SCRIPT \n            --case $TEST \n            --output-path $OUTPUT_FILE"

        # Add profiling if enabled
        if [[ "$PROFILE" == true ]]; then
            echo "Running with Nsight Systems profiling..."
            NSYS_OUTPUT="${OUTPUT_BASE}.nsys-rep"
            nsys profile \n                --trace=cuda,nvtx \n                --force-overwrite=true \n                --output="$NSYS_OUTPUT" \n                $CMD
        else
            $CMD
        fi

        echo ""
        echo "Results saved to: $OUTPUT_FILE"
        if [[ "$PROFILE" == true ]]; then
            echo "Profile saved to: $NSYS_OUTPUT"
        fi
    fi
else
    echo "Error: Unknown recipe '$RECIPE'. Currently only 'mxfp8' is supported."
    exit 1
fi

echo ""
echo "All tests completed successfully!"