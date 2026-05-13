#!/bin/bash
# run.sh - Test runner with matrix expansion
# Usage: ./run.sh test <testcase>  or  ./run.sh list
#
# Dependencies: yq (https://github.com/mikefarah/yq)
#   Install: brew install yq  (macOS)

set -e

MEGATRON_DIR=$(pwd)
WORKSPACE_DIR=$(pwd)
LOGS_DIR="${WORKSPACE_DIR}/logs"     # megatron/logs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTCASES_DIR="$(cd "${SCRIPT_DIR}/../testcases" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.sh"

# Check yq is installed
if ! command -v yq &> /dev/null; then
    echo "yq is not installed. installing yq..."
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        YQ_BINARY="yq_linux_arm64"
    elif [ "$ARCH" = "x86_64" ]; then
        YQ_BINARY="yq_linux_amd64"
    else
        echo "unknown architecture: $ARCH"
        echo "Others: see https://github.com/mikefarah/yq"
        exit 1
    fi

    curl -L "https://github.com/mikefarah/yq/releases/download/v4.53.2/${YQ_BINARY}" -o yq && \
    chmod +x yq && \
    sudo mv yq /usr/local/bin/ && \
    echo "yq version: $(yq --version)"
fi

# ========== Helper Functions ==========
log() {
    # echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*";
    :
}

# Convert short param name to long train.sh param name
# Values with special characters (like parentheses) are wrapped in single quotes
param_to_arg() {
    local name=$1
    local value=$2

    # Check if value contains special characters that need quoting
    # Special chars: ( ) [ ] { } * ? & | ; < > $ ` \ " space
    case "$value" in
        *[\(\)\[\]\{\}\*\?\&\|\;\<\>\$\`\\\"\ ]*)
            value="'$value'"
            ;;
    esac

    case "$name" in
        # map short name to long name
        pp) echo "--pipeline-parallel $value" ;;
        tp) echo "--tensor-parallel $value" ;;
        ep) echo "--expert-parallel $value" ;;
        mbs) echo "--micro-batch-size $value" ;;
        gbs) echo "--global-batch-size $value" ;;
        expert) echo "--num-expert $value" ;;
        layer) echo "--num-layer $value" ;;
        moe) echo "--moe-freq $value" ;;
        seq-len) echo "--seq-length $value" ;;
        pp-layout) echo "--pipeline-parallel-layout $value" ;;
        # echo long name
        *) echo "--$name $value" ;;
    esac
}

# Convert feature to train.sh args using yq
feature_to_args() {
    local file=$1
    local args=""

    # Check profile
    local profile=$(yq '.feature.profile // false' "$file")
    if [[ "$profile" == "true" ]]; then
        args+=" --profile"
    fi

    # Check graph
    local graph=$(yq '.feature.graph // false' "$file")
    if [[ "$graph" == "true" ]]; then
        args+=" --graph"
    fi

    # Check offload
    local offload_count=$(yq '.feature.offload | length' "$file")
    if [ "$offload_count" -gt 0 ]; then
        for i in $(seq 0 $((offload_count - 1))); do
            local item=$(yq ".feature.offload[$i]" "$file")
            case "$item" in
                act) args+=" --offload-act" ;;
                weight|wt) args+=" --offload-weight" ;;
                optim|opt) args+=" --offload-optim" ;;
                fine) args+=" --offload-fine" ;;
            esac
        done
    fi

    echo "$args"
}

# Get all matrix keys using yq
get_matrix_keys() {
    local file=$1
    yq '.matrix | keys | .[]' "$file" 2>/dev/null
}

# Parse matrix values using yq
parse_matrix_values() {
    local file=$1
    local key=$2
    yq ".matrix.$key | join(\" \")" "$file" 2>/dev/null
}

# Build base config from param section using yq
build_base_config() {
    local file=$1
    local config=""

    # Get all param keys and build config
    local keys=$(yq '.param | keys | .[]' "$file" 2>/dev/null)
    for key in $keys; do
        local value=$(yq ".param.$key" "$file")
        # Remove surrounding quotes if present
        value=$(echo "$value" | sed 's/^"//;s/"$//')
        config+="$(param_to_arg "$key" "$value") "
    done

    echo "$config"
}

# Get all matrix combinations using iterative approach (BFS-like)
get_matrix_combinations() {
    local file=$1
    local keys=($(get_matrix_keys "$file"))

    if [ ${#keys[@]} -eq 0 ]; then
        echo ""
        return
    fi

    # Build arrays of values for each key
    declare -a values_arrays
    for i in "${!keys[@]}"; do
        values_arrays[$i]=$(parse_matrix_values "$file" "${keys[$i]}")
    done

    # Iterative combination generation
    local combinations=("")

    for i in "${!keys[@]}"; do
        local key="${keys[$i]}"
        local values=(${values_arrays[$i]})
        local new_combinations=()

        for combo in "${combinations[@]}"; do
            for val in "${values[@]}"; do
                if [ -z "$combo" ]; then
                    new_combinations+=("${key}=${val}")
                else
                    new_combinations+=("${combo} ${key}=${val}")
                fi
            done
        done

        combinations=("${new_combinations[@]}")
    done

    printf '%s\n' "${combinations[@]}"
}

# Run single test
run_single_test() {
    local test_name=$1
    local base_config=$2
    local matrix_config=$3
    local feature_args=$4
    local dry_run=$5

    # Build run name from matrix config
    local run_suffix=""
    if [ -n "$matrix_config" ]; then
        run_suffix=$(echo "$matrix_config" | tr ' ' '-' | tr '=' '-')
    fi

    local run_name="${test_name}"
    [ -n "$run_suffix" ] && run_name="${test_name}-${run_suffix}"

    log "[RUN] $run_name"

    # Convert matrix config to args
    local matrix_args=""
    if [ -n "$matrix_config" ]; then
        for pair in $matrix_config; do
            local key="${pair%%=*}"
            local val="${pair#*=}"
            matrix_args+=" $(param_to_arg "$key" "$val")"
        done
    fi

    # Build full command
    local full_config="$base_config$matrix_args"

    log "  Config: $full_config"
    log "  Features: $feature_args"

    # Execute
    local cmd="$TRAIN_SCRIPT $full_config $feature_args"

    if [ "$dry_run" = true ]; then
        log "  [DRY-RUN] Command: $cmd"
    else
        log "  Command: $cmd"
        eval "$cmd" || true
    fi
}

# ========== Commands ==========

cmd_list() {
    echo "Available testcases:"
    for yaml in "$TESTCASES_DIR"/*.yaml; do
        if [ -f "$yaml" ]; then
            local name=$(basename "$yaml" .yaml)
            local desc=$(yq '.description // ""' "$yaml")
            printf "  %-30s %s\n" "$name" "$desc"
        fi
    done
}

cmd_test() {
    local test_name=$1
    local dry_run=$2
    local yaml_file="$TESTCASES_DIR/${test_name}.yaml"

    if [ ! -f "$yaml_file" ]; then
        log "[ERROR] Testcase not found: $yaml_file"
        exit 1
    fi

    log "[TESTCASE] $test_name"
    log "  File: $yaml_file"

    # Read base config from param section (using yq)
    local base_config=$(build_base_config "$yaml_file")

    # Get feature args
    local feature_args=$(feature_to_args "$yaml_file")

    log "  Base config: $base_config"
    log "  Features: $feature_args"

    # Get matrix combinations
    local combinations=($(get_matrix_combinations "$yaml_file"))

    if [ ${#combinations[@]} -eq 0 ] || [ -z "${combinations[0]}" ]; then
        log "  No matrix, running single test"
        run_single_test "$test_name" "$base_config" "" "$feature_args" "$dry_run"
    else
        log "  Matrix combinations: ${#combinations[@]}"
        for combo in "${combinations[@]}"; do
            run_single_test "$test_name" "$base_config" "$combo" "$feature_args" "$dry_run"
        done
    fi

    log "[DONE] $test_name"
}

# ========== Log Directory Management ==========
prepare_logs() {
    if [ -d "$LOGS_DIR" ]; then
        local timestamp=$(date +%Y%m%d-%H%M%S)
        mv "$LOGS_DIR" "${LOGS_DIR}-${timestamp}"
        log "Archived existing logs to logs-${timestamp}"
    fi

    mkdir -p "$LOGS_DIR"
}

# Copy YAML files to logs after test
copy_yaml_to_logs() {
    cp "$TESTCASES_DIR/$1.yaml" "$LOGS_DIR/"
}

# ========== Main ==========
main() {
    local command="${1:-}"
    local dry_run=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -d|--dry-run)
                dry_run=true
                shift
                ;;
            list|test)
                command="$1"
                shift
                break
                ;;
            *)
                shift
                ;;
        esac
    done

    case "$command" in
        list)
            cmd_list
            ;;
        test)
            if [ -z "${1:-}" ]; then
                echo "Usage: $0 [-d|--dry-run] test <testcase-name>"
                exit 1
            fi
            if [ "$dry_run" != true ]; then
                prepare_logs
            fi
            cmd_test "$1" "$dry_run"
            if [ "$dry_run" != true ]; then
                copy_yaml_to_logs "$1"
            fi
            ;;
        *)
            echo "Usage: $0 [-d|--dry-run] {list|test <testcase>}"
            echo ""
            echo "Commands:"
            echo "  list              List all available testcases"
            echo "  test <name>       Run specified testcase"
            echo ""
            echo "Options:"
            echo "  -d, --dry-run     Dry run mode (print commands only, do not execute)"
            echo ""
            echo "Examples:"
            echo "  $0 list"
            echo "  $0 test ep4-alltoall"
            echo "  $0 -d test ep4-alltoall"
            echo "  $0 --dry-run test ep4-alltoall"
            exit 1
            ;;
    esac
}

main "$@"
