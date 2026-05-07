#!/bin/bash
# Testcase runner for deepseek-v3 training
# Usage: ./run_testcase.sh --test <testcase_name> [--graph] [--offload <module>]

set -e

# Parse arguments first
TESTCASE_NAME=""
ENABLE_GRAPH=false
ENABLE_PROFILE=false
OFFLOAD_MODULES=()

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
        --graph)
            ENABLE_GRAPH=true
            shift
            ;;
        --profile)
            ENABLE_PROFILE=true
            shift
            ;;
        --offload)
            if [ -z "$2" ]; then
                echo "Error: --offload requires module list"
                usage
            fi
            IFS=',' read -ra OFFLOAD_MODULES <<< "$2"
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

# Function to display usage
usage() {
    echo "Usage: $0 --test <testcase_name> [--graph] [--profile] [--offload <module>]"
    echo ""
    echo "Options:"
    echo "  --test <testcase_name>    Specify which testcase to run (required)"
    echo "  --graph                    Enable CUDA graph (passed to training script)"
    echo "  --profile                  Enable profiling (passed to training script)"
    echo "  --offload <modules>        Enable offloading modules (comma-separated list)"
    echo "                             Available modules: act, weight, opt"
    echo "                             For example: --offload act,weight"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Available testcases:"
    echo "  5layer-ep4-alltoall        - Test with dp=ep=4, alltoall dispatcher, seq_len=4096,"
    echo "                               mbs=1/2/4/8, 2 dense layers + 3 MoE layers, 32 experts"
    echo "  5layer-ep4-hybridep        - Test with dp=ep=4, hybridep dispatcher, seq_len=4096,"
    echo "                               mbs=1/2/4/8, 2 dense layers + 3 MoE layers, 32 experts"
    echo "  5layer-vpp4-alltoall       - Test with pp=4, vpp=4, 4 GPUs, seq_len=4096,"
    echo "                               mbs=1, gbs=64, 3 dense layers + 26 MoE layers, 256 experts,"
    echo "                               one test only"
    echo ""
    echo "Examples:"
    echo "  $0 --test 5layer-ep4-alltoall"
    echo "  $0 --test 5layer-ep4-alltoall --graph"
    echo "  $0 --test 5layer-ep4-alltoall --profile"
    echo "  $0 --test 5layer-ep4-alltoall --offload act,weight"
    echo "  $0 --test 5layer-ep4-hybridep --graph --offload opt"
    echo "  $0 --test 5layer-vpp4-alltoall"
    exit 1
}

if [ -z "${TESTCASE_NAME+x}" ]; then
    echo "Error: --test is required"
    usage
fi

# Function to build all arguments to pass to train_deepseek_v3_gb200.sh
build_script_args() {
    local args=()

    # Add profile argument
    if [ "$ENABLE_PROFILE" = true ]; then
        args+=(--profile)
    fi

    # Add graph argument
    if [ "$ENABLE_GRAPH" = true ]; then
        args+=(--enable-cuda-graph)
    fi

    # Add offload arguments
    for module in "${OFFLOAD_MODULES[@]}"; do
        case "$module" in
            act)
                args+=(--cpu-offloading)
                ;;
            weight)
                # Note: Check train_deepseek_v3_gb200.sh for exact weight offload flag
                args+=(--offload-weights)
                ;;
            opt)
                args+=(--optimizer-offload)
                ;;
            *)
                echo "Warning: Unknown offload module '$module', skipping"
                ;;
        esac
    done

    echo "${args[@]}"
}

# Function to build offload arguments (deprecated, use build_script_args instead)
build_offload_args() {
    local args=()
    for module in "${OFFLOAD_MODULES[@]}"; do
        case "$module" in
            act)
                args+=(--cpu-offloading)
                ;;
            weight)
                # Note: Check train_deepseek_v3_gb200.sh for exact weight offload flag
                args+=(--offload-weights)
                ;;
            opt)
                args+=(--optimizer-offload)
                ;;
            *)
                echo "Warning: Unknown offload module '$module', skipping"
                ;;
        esac
    done
    echo "${args[@]}"
}

# Function to run 5layer-ep4-alltoall testcase
run_5layer_ep4_alltoall() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    echo "========================================"
    echo "Running testcase: 5layer-ep4-alltoall"
    echo "========================================"
    echo "Configuration:"
    echo "  - DP: 4, EP: 4"
    echo "  - Dispatcher: alltoall"
    echo "  - Sequence Length: 4096"
    echo "  - Layer Layout: 2 dense + 3 MoE (total 5 layers)"
    echo "  - Experts: 32"
    echo "  - CUDA Graph: $ENABLE_GRAPH"
    echo "  - Profile: $ENABLE_PROFILE"
    echo "  - Offload: ${OFFLOAD_MODULES[*]:-none}"
    echo "  - Micro Batch Sizes: 1, 2, 4, 8"
    echo "========================================"

    local mbs_values=(1 2 4 8)
    local script_args=($(build_script_args))

    for mbs in "${mbs_values[@]}"; do
        echo ""
        echo "===== Running with Micro Batch Size: $mbs ====="

        PARAM=(
            "$script_dir/train_deepseek_v3_gb200.sh"
            --tp 1
            --pp 1
            --ep 4
            --micro-batch-size $mbs
            --global-batch-size 128
            --num-expert 32
            --num-layer 5
            --moe-freq "([0]*2+[1]*3)"
            --seq-length 4096
            --dispatcher alltoall
        )

        if [ ${#script_args[@]} -gt 0 ]; then
            PARAM+=("${script_args[@]}")
        fi

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
    echo "5layer-ep4-alltoall testcase completed"
    echo "========================================"
}

# Function to run 5layer-ep4-hybridep testcase
run_5layer_ep4_hybridep() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    echo "========================================"
    echo "Running testcase: 5layer-ep4-hybridep"
    echo "========================================"
    echo "Configuration:"
    echo "  - DP: 4, EP: 4"
    echo "  - Dispatcher: hybridep"
    echo "  - Sequence Length: 4096"
    echo "  - Layer Layout: 2 dense + 3 MoE (total 5 layers)"
    echo "  - Experts: 32"
    echo "  - CUDA Graph: $ENABLE_GRAPH"
    echo "  - Profile: $ENABLE_PROFILE"
    echo "  - Offload: ${OFFLOAD_MODULES[*]:-none}"
    echo "  - Micro Batch Sizes: 1, 2, 4, 8"
    echo "========================================"

    local mbs_values=(1 2 4 8)
    local script_args=($(build_script_args))

    for mbs in "${mbs_values[@]}"; do
        echo ""
        echo "===== Running with Micro Batch Size: $mbs ====="

        PARAM=(
            "$script_dir/train_deepseek_v3_gb200.sh"
            --tp 1
            --pp 1
            --ep 4
            --micro-batch-size $mbs
            --global-batch-size 128
            --num-expert 32
            --num-layer 5
            --moe-freq "([0]*2+[1]*3)"
            --seq-length 4096
            --dispatcher hybridep
        )

        if [ ${#script_args[@]} -gt 0 ]; then
            PARAM+=("${script_args[@]}")
        fi

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
    echo "5layer-ep4-hybridep testcase completed"
    echo "========================================"
}

# Function to run 5layer-vpp4-alltoall testcase
run_5layer_vpp4_alltoall() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    echo "========================================"
    echo "Running testcase: 5layer-vpp4-alltoall"
    echo "========================================"
    echo "Configuration:"
    echo "  - PP: 4, VPP: 4"
    echo "  - Dispatcher: alltoall"
    echo "  - Sequence Length: 4096"
    echo "  - Layer Layout: 3 dense + 26 MoE (total 29 layers)"
    echo "  - Experts: 256"
    echo "  - Micro Batch Size: 1"
    echo "  - Global Batch Size: 64"
    echo "  - CUDA Graph: $ENABLE_GRAPH"
    echo "  - Profile: $ENABLE_PROFILE"
    echo "  - Offload: ${OFFLOAD_MODULES[*]:-none}"
    echo "========================================"

    local script_args=($(build_script_args))

    PARAM=(
        "$script_dir/train_deepseek_v3_gb200.sh"
        --tp 1
        --pp 4
        --pp-layout "Et|(tt|)*14L"
        --ep 1
        --micro-batch-size 1
        --global-batch-size 64
        --num-layer 29
        --moe-freq "([0]*29)"
        --seq-length 4096
        --dispatcher alltoall
    )

    if [ ${#script_args[@]} -gt 0 ]; then
        PARAM+=("${script_args[@]}")
    fi

    bash "${PARAM[@]}"

    if [ $? -eq 0 ]; then
        echo "===== Successfully completed 5layer-vpp4-alltoall ====="
    else
        echo "===== Failed 5layer-vpp4-alltoall ====="
    fi

    echo ""
    echo "========================================"
    echo "5layer-vpp4-alltoall testcase completed"
    echo "========================================"
}

# Execute testcase based on name
case "$TESTCASE_NAME" in
    5layer-ep4-alltoall)
        run_5layer_ep4_alltoall
        ;;
    5layer-ep4-hybridep)
        run_5layer_ep4_hybridep
        ;;
    5layer-vpp4-alltoall)
        run_5layer_vpp4_alltoall
        ;;
    *)
        echo "Error: Unknown testcase '$TESTCASE_NAME'"
        echo ""
        echo "Available testcases:"
        echo "  5layer-ep4-alltoall"
        echo "  5layer-ep4-hybridep"
        echo "  5layer-vpp4-alltoall"
        exit 1
        ;;
esac