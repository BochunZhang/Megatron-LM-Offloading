#!/bin/bash
# GEMM Performance Testing Script
#
# Usage:
#   ./run_gemm_test.sh --recipe mxfp8 --test batch --profile
#   ./run_gemm_test.sh --recipe mxfp8 --test forward

set -e

# Default values
RECIPE="mxfp8"
TEST="batch"
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


# Python script path
case "$RECIPE" in
    mxfp8)
        PY_SCRIPT="tests/gemm/mxfp8_gemm.py"
        ;;
    bf16)
        PY_SCRIPT="tests/gemm/mxfp8_gemm.py"
        ;;
    *)
        echo "Fatal: No matching script for recipe '$RECIPE'"
        exit 1
        ;;
esac


# test config
case "$TEST" in
    batch)
        BATCH=(1 2 4 8 16)
        HIDDEN=(7168)
        OUT=(1536)
        SEQ=(4096)
        ;;
    *)
        echo "Fatal: No matching testcase for '$TEST'"
        exit 1
        ;;
esac



# Check if python script exists
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "Error: Python script not found at $PY_SCRIPT"
    exit 1
fi

# Output base path
OUTPUT_BASE="$OUTPUT_DIR/${RECIPE}-${TEST}"

mkdir -p $OUTPUT_BASE


# Add profiling if enabled
if [[ "$PROFILE" == true ]]; then
    export export NVTE_NVTX_ENABLED=1
fi


for mbs in "${BATCH[@]}"; do
    for hds in "${HIDDEN[@]}"; do
        for out in "${OUT[@]}"; do
            for seq in "${SEQ[@]}"; do
                echo "Testing with batch_size=$mbs, hidden_size=$hds, output_size=$out, seq_length=$seq"
                echo "----------------------------------------"
                
                # Output file for this batch size
                OUTPUT_FILE="${OUTPUT_BASE}/mbs${mbs}.seq${seq}.hds${hds}.out${out}"

                # Add profiling if enabled
                if [[ "$PROFILE" == true ]]; then
                        NSYS_ARGS=(
                        nsys profile -s none -t nvtx,cuda,cudnn,cublas
                        --cudabacktrace=all 
                        --cuda-graph-trace=node 
                        --python-backtrace=cuda 
                        --wait all 
                        --force-overwrite true 
                        -o $OUTPUT_FILE.nsys-rep
                    )
                fi

                PYTHON_ARGS=(
                    python3 "$PY_SCRIPT"
                    --seq_len "$seq"
                    --batch_size "$mbs"
                    --hidden_size "$hds"
                    --out_features "$out"
                )

                ${NSYS_ARGS[@]} \
                ${PYTHON_ARGS[@]}
            done
        done
    done
done

echo ""
echo "All tests completed successfully!"