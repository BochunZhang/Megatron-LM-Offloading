#!/bin/bash
# bin/run-test.sh - Layer 3: Test Runner
# Responsibility: Test execution management, result collection, report generation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/features.sh"

# ========== Configuration ==========
LOGS_DIR="${WORKSPACE_PATH:-$(cd "$SCRIPT_DIR/../.." && pwd)}/logs"
TESTCASES_DIR="${SCRIPT_DIR}/../testcases"
TRAIN_SCRIPT="${SCRIPT_DIR}/train.sh"

# ========== Help ==========
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Layer 3: Test Runner - Execute test cases

Options:
    --test <name>           Execute specified testcase (YAML filename without extension)
    --test-all              Execute all testcases in testcases directory
    --list                  List all available testcases

    --feature <f>           Enable feature (can be used multiple times)
    --features <f1,f2,...>  Enable multiple features (comma separated)

    --mbs <list>            Override micro batch size list (e.g., 1,2,4)
    --iters <n>             Override training iterations
    --dry-run               Print commands only, do not execute

    -h, --help              Show help

Examples:
    $0 --test ep4-alltoall
    $0 --test ep4-alltoall --features graph,profile
    $0 --test ep4-alltoall --feature graph --feature offload_act
    $0 --test ep4-alltoall --mbs 1,2 --features graph
    $0 --test-all --features graph
EOF
    echo ""
    feature_help
    exit 0
}

# ========== Log Function ==========
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ========== API: Parse testcase config ==========
# Simple YAML parser, extracts key: value format
parse_yaml() {
    local file=$1
    local key=$2

    grep -E "^${key}:" "$file" 2>/dev/null | head -1 | sed 's/^[^:]*:[[:space:]]*//' | sed 's/[[:space:]]*$//'
}

parse_yaml_list() {
    local file=$1
    local key=$2

    # Extract list format [1, 2, 4]
    local line=$(grep -E "^${key}:" "$file" 2>/dev/null | head -1)
    if [[ "$line" == *"["* ]]; then
        echo "$line" | sed 's/.*\[\(.*\)\].*/\1/' | tr ',' ' ' | tr -d ' '
    else
        echo "$line" | sed 's/^[^:]*:[[:space:]]*//' | tr ',' ' '
    fi
}


# ========== API: Collect Metrics ==========
collect_metrics() {
    local test_name=$1
    local log_file=$2

    local metrics_file="$LOGS_DIR/${test_name}.metrics"

    # Extract key metrics
    {
        echo "# Test: $test_name"
        echo "# Time: $(date)"
        grep -E "(throughput|memory|loss)" "$log_file" 2>/dev/null | tail -20 || true
    } > "$metrics_file"
}

# ========== API: Run Single Test ==========
run_single_test() {
    local test_name=$1
    local config_str=$2
    local features=$3
    local dry_run=$4

    log "[RUN] $test_name"
    log "  Config: $config_str"
    log "  Features: $features"

    # Build feature arguments
    local feature_args=$(build_feature_args "$features")

    # 构建完整命令
    local cmd="$TRAIN_SCRIPT $config_str $feature_args"

    if [[ "$dry_run" == true ]]; then
        log "[DRY-RUN] $cmd"
        return 0
    fi
}

# ========== API: Build config string from YAML ==========
build_config_from_yaml() {
    local yaml_file=$1
    local override_mbs=$2
    local override_iters=$3

    local config=""

    # Base configuration
    local pp=$(parse_yaml "$yaml_file" "pp")
    local tp=$(parse_yaml "$yaml_file" "tp")
    local ep=$(parse_yaml "$yaml_file" "ep")
    local gbs=$(parse_yaml "$yaml_file" "global-batch-size")
    local num_expert=$(parse_yaml "$yaml_file" "num-expert")
    local num_layer=$(parse_yaml "$yaml_file" "num-layer")
    local seq_len=$(parse_yaml "$yaml_file" "seq-length")
    local dispatcher=$(parse_yaml "$yaml_file" "dispatcher")
    local moe_freq=$(parse_yaml "$yaml_file" "moe-freq")
    local pp_layout=$(parse_yaml "$yaml_file" "pp-layout")

    [[ -n "$pp" ]] && config+="--pp $pp "
    [[ -n "$tp" ]] && config+="--tp $tp "
    [[ -n "$ep" ]] && config+="--ep $ep "
    [[ -n "$gbs" ]] && config+="--global-batch-size $gbs "
    [[ -n "$num_expert" ]] && config+="--num-expert $num_expert "
    [[ -n "$num_layer" ]] && config+="--num-layer $num_layer "
    [[ -n "$seq_len" ]] && config+="--seq-length $seq_len "
    [[ -n "$dispatcher" ]] && config+="--dispatcher $dispatcher "
    [[ -n "$moe_freq" ]] && config+="--moe-freq \"$moe_freq\" "
    [[ -n "$pp_layout" ]] && config+="--pp-layout \"$pp_layout\" "

    # Override training iterations
    if [[ -n "$override_iters" ]]; then
        config+="--train-iters $override_iters "
    fi

    echo "$config"
}

# ========== API: Expand Parameter Matrix ==========
expand_matrix() {
    local yaml_file=$1
    local key=$2

    parse_yaml_list "$yaml_file" "$key"
}

# ========== API: Execute Testcase ==========
run_testcase() {
    local test_name=$1
    local features=$2
    local mbs_override=$3
    local iters_override=$4
    local dry_run=$5

    local yaml_file="$TESTCASES_DIR/${test_name}.yaml"

    if [[ ! -f "$yaml_file" ]]; then
        log "[ERROR] Testcase not found: $yaml_file"
        return 1
    fi

    log "[TESTCASE] $test_name"
    log "  File: $yaml_file"

    # Get base configuration
    local base_config=$(build_config_from_yaml "$yaml_file" "" "$iters_override")

    # Determine mbs list
    local mbs_list="${mbs_override:-$(expand_matrix "$yaml_file" "mbs")}"
    [[ -z "$mbs_list" ]] && mbs_list="1"  # default mbs=1

    # Execute each mbs combination
    local idx=0
    local failed=0
    for mbs in $mbs_list; do
        idx=$((idx+1))
        local run_name="${test_name}-mbs${mbs}"
        local run_config="${base_config}--micro-batch-size $mbs"

        if ! run_single_test "$run_name" "$run_config" "$features" "$dry_run"; then
            failed=$((failed+1))
        fi
    done

    log "[SUMMARY] $test_name: $idx runs, $failed failed"
    return $((failed > 0 ? 1 : 0))
}

# ========== Main Logic ==========
main() {
    local test_name=""
    local test_all=false
    local list=false
    local features=""
    local mbs_override=""
    local iters_override=""
    local dry_run=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --test) test_name="$2"; shift 2 ;;
            --test-all) test_all=true; shift ;;
            --list) list=true; shift ;;
            --feature) features+="$2 "; shift 2 ;;
            --features) features+="${2//,/ } "; shift 2 ;;
            --mbs) mbs_override="$2"; shift 2 ;;
            --iters) iters_override="$2"; shift 2 ;;
            --dry-run) dry_run=true; shift ;;
            -h|--help) usage ;;
            *) echo "Unknown: $1"; exit 1 ;;
        esac
    done

    # List testcases
    if [[ "$list" == true ]]; then
        echo "Available testcases in $TESTCASES_DIR:"
        for f in "$TESTCASES_DIR"/*.yaml; do
            if [[ -f "$f" ]]; then
                local name=$(basename "$f" .yaml)
                local desc=$(parse_yaml "$f" "description")
                printf "  %-30s %s\n" "$name" "$desc"
            fi
        done
      exit 0
    fi

    # 验证功能
    if [[ -n "$features" ]]; then
        if ! validate_features "$features"; then
            exit 1
        fi
    fi

    # 执行单个testcase
    if [[ -n "$test_name" ]]; then
        run_testcase "$test_name" "$features" "$mbs_override" "$iters_override" "$dry_run"
        exit $?
    fi

    # 执行所有testcase
    if [[ "$test_all" == true ]]; then
        local total=0
        local failed=0

        for yaml in "$TESTCASES_DIR"/*.yaml; do
            if [[ -f "$yaml" ]]; then
                local name=$(basename "$yaml" .yaml)
                total=$((total+1))
                if ! run_testcase "$name" "$features" "$mbs_override" "$iters_override" "$dry_run"; then
                    failed=$((failed+1))
                fi
            fi
        done

        log "[FINAL] Total: $total testcases, Failed: $failed"
        exit $((failed > 0 ? 1 : 0))
    fi

    # 没有指定操作
    usage
}

main "$@"
