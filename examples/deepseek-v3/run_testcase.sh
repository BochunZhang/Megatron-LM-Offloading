#!/bin/bash
# Testcase runner for deepseek-v3 training
# Usage: ./run_testcase.sh --test <testcase_name>

set -e

# Function to display usage
usage() {
    echo "Usage: $0 --test <testcase_name>"
    echo ""
    echo "Available testcases:"
    echo "  measure-operators-4gpu  - Test with dp=ep=4, alltoall dispatcher, seq_len=4096,"
    echo "                            mbs=1/2/4/8, 2 dense layers + 3 MoE layers, 32 experts, no cuda graph"
    echo ""
    echo "Example:"
    echo "  $0 --test measure-operators-4gpu"
    exit 1
}

# Function to run measure-operators-4gpu testcase
run_measure_operators_4gpu() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    echo "========================================"
    echo "Running testcase: measure-operators-4gpu"
    echo "========================================"
    echo "Configuration:"
    echo "  - DP: 4, EP: 4"
    echo "  - Dispatcher: alltoall"
    echo "  - Sequence Length: 4096"
    echo "  - Layer Layout: 2 dense + 3 MoE (total 5 layers)"
    echo "  - Experts: 32"
    echo "  - CUDA Graph: disabled"
    echo "  - Micro Batch Sizes: 1, 2, 4, 8"
    echo "========================================"

    local mbs_values=(1 2 4 8)

    for mbs in "${mbs_values[@]}"; do
        echo ""
        echo "===== Running with Micro Batch Size: $mbs ====="

        PARAM=(
            "$script_dir/train_deepseek_v3_gb200.sh"
            --tp 1
            --pp 1
            --ep 4
            --micro-batch-size $mbs
            --num-expert 32
            --num-layer 5
            --moe-freq "([0]*2+[1]*3)"
            --seq-length 4096
            --dispatcher alltoall
        )
        bash "${PARAM[@]}"

        if [ $? -eq 0 ]; then
            echo "===== Successfully completed with MBS=$mbs ====="
        else
            echo "===== Failed with MBS=$mbs ====="
        fi

        echo ""
    done

    echo ""
    echo "========================================"
    echo "measure-operators-4gpu testcase completed"
    echo "========================================"
}

# Parse arguments
if [ $# -eq 0 ]; then
    usage
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            if [ -z "$2" ]; then
                echo "Error: --test requires a testcase name"
                usage
            fi
            TESTCASE_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
done

if [ -z "${TESTCASE_NAME+x}" ]; then
    echo "Error: --test is required"
    usage
fi

# Execute testcase based on name
case "$TESTCASE_NAME" in
    measure-operators-4gpu)
        run_measure_operators_4gpu
        ;;
    *)
        echo "Error: Unknown testcase '$TESTCASE_NAME'"
        echo ""
        echo "Available testcases:"
        echo "  measure-operators-4gpu"
        exit 1
        ;;
esac