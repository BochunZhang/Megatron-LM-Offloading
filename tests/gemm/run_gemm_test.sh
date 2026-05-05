#!/bin/bash
# GEMM Performance Testing Script
#
# Usage:
#   ./run_gemm_test.sh --recipe mxfp8 --test batch --profile
#   ./run_gemm_test.sh --recipe mxfp8 --test forward

set -e

# Default values
RECIPE="mxfp8"
TEST="forward"
GRAPH=false
PROFILE=false
TRAIN=forward
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
        --graph)
            GRAPH=true
            shift
            ;;
        --backward)
            TRAIN=train
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --recipe <recipe> --test <test> [--profile]"
            echo ""
            echo "Options:"
            echo "  --test      Test type: 'linear' 'normal' or 'operator' (sweep operator and batch size)"
            echo "  --recipe    Recipe type (e.g., mxfp8)"
            echo "  --profile   Enable Nsight Systems profiling"
            echo ""
            echo "Examples:"
            echo "  $0 --recipe mxfp8 --test demo --profile"
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



# Check if python script exists
if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "Error: Python script not found at $PY_SCRIPT"
    exit 1
fi


# Add profiling if enabled
if [[ "$PROFILE" == true ]]; then
    export export NVTE_NVTX_ENABLED=1
fi

# Function to run a single test configuration
run_mxfp8_linear() {
    local script=$1
    local name=$2       # linear_proj
    local type=$3       # linear

    local mbs=$4
    local hds=$5
    local out=$6
    local seq=$7

    echo "Testing '$name' ($type) with batch_size=$mbs, hidden_size=$hds, output_size=$out, seq_length=$seq"
    echo "----------------------------------------"

    # Output file for this batch size
    mkdir -p "${OUTPUT_DIR}/$name.$type.$TRAIN"
    local nsysfile="${OUTPUT_DIR}/$name.$type.$TRAIN/mbs${mbs}.seq${seq}.hds${hds}.out${out}.nsys-rep"

    # Build profiling arguments if enabled
    local nsys_args=()
    if [[ "$PROFILE" == true ]]; then
        nsys_args=(
            nsys profile -s none -t nvtx,cuda,cudnn,cublas
            --cudabacktrace=all
            --cuda-graph-trace=node
            --python-backtrace=cuda
            --wait all
            --force-overwrite true
            -o "$nsysfile"
        )
    fi

    # Build python arguments
    local python_args=(
        python3 "$script"
        --name $name
        --operator $type
        --seq_len "$seq"
        --batch_size "$mbs"
        --hidden_size "$hds"
        --out_features "$out"
    )

    if [ "$TRAIN" = 'train' ]; then
        python_args+=(
            --backward
        )
    fi

    if [ "$graph" = true ]; then
        python_args+=(
            --graph
        )
    fi

    # Execute test
    "${nsys_args[@]}" "${python_args[@]}"
}


# test config
case "$TEST" in
    linear)
        run_mxfp8_linear $PY_SCRIPT 'gemm' 'linear' 1 7168 1536 4096 
        ;;

    normal)
        run_mxfp8_linear $PY_SCRIPT 'gemm' 'norm_linear' 1 7168  24576 4096
        ;;    

    operator)
        BATCH=(1 2 4 6 8 16 32 64)
        for mbs in "${BATCH[@]}"; do
            run_mxfp8_linear $PY_SCRIPT 'linear_q_down_proj'  'linear'      1 7168  1536  4096
            run_mxfp8_linear $PY_SCRIPT 'linear_kv_down_proj' 'linear'      1 7168  576   4096
            run_mxfp8_linear $PY_SCRIPT 'linear_q_up_proj'    'norm_linear' 1 1536  24576 4096
            run_mxfp8_linear $PY_SCRIPT 'linear_kv_up_proj'   'norm_linear' 1 512   32768 4096
            run_mxfp8_linear $PY_SCRIPT 'linear_proj'         'linear'      1 16384 7168  4096
        done
        ;;
    *)
        echo "Fatal: No matching testcase for '$TEST'"
        exit 1
        ;;
esac


echo ""
echo "All tests completed successfully!"