# bin/features.sh - Layer 2: Feature Controller
# Responsibility: Manage feature switches, provide unified parameter building and conflict checking

# ========== Feature Definition Table ==========
# Define all supported features and their parameter mappings
#
# Format: FEATURE_NAME -> Layer 1 parameter
# graph         -> --enable-cuda-graph
# profile       -> --profile
# offload_act   -> --activation-offload
# offload_wt    -> --weights-offload
# offload_opt   -> --optimizer-offload
# fine_grained  -> --fine-grained-offload

# ========== Feature Conflict Check ==========
# Define incompatible feature combinations (space separated)
# Format: "feature1 feature2"
FEATURE_CONFLICTS=(
    # "graph offload_act"  # CUDA Graph may conflict with activation offload
)

# ========== API: Validate Feature Combination ==========
validate_features() {
    local features="$1"
    local valid=0

    # Check conflicts
    for conflict in "${FEATURE_CONFLICTS[@]}"; do
        local f1=$(echo "$conflict" | awk '{print $1}')
        local f2=$(echo "$conflict" | awk '{print $2}')
        if [[ "$features" == *"$f1"* && "$features" == *"$f2"* ]]; then
            echo "[WARN] Feature conflict: $f1 + $f2" >&2
        fi
    done

    # Check unknown features
    local valid_features="graph profile offload_act offload_wt offload_opt fine_grained"
    for f in $features; do
        if [[ ! " $valid_features " == *" $f "* ]]; then
            echo "[ERROR] Unknown feature: $f" >&2
            valid=1
        fi
    done

    return $valid
}

# ========== API: Build Feature Arguments ==========
build_feature_args() {
    local features="$1"
    local args=()

    [[ "$features" == *"graph"* ]] && args+=("--enable-cuda-graph")
    [[ "$features" == *"profile"* ]] && args+=("--profile")
    [[ "$features" == *"offload_act"* ]] && args+=("--activation-offload")
    [[ "$features" == *"offload_wt"* ]] && args+=("--weights-offload")
    [[ "$features" == *"offload_opt"* ]] && args+=("--optimizer-offload")
    [[ "$features" == *"fine_grained"* ]] && args+=("--fine-grained-offload")

    echo "${args[@]}"
}

# ========== API: Parse Feature String ==========
# Support: "graph,profile,offload_act" or "graph profile offload_act"
parse_features() {
    local input="$1"
    # Replace comma with space for standardized output
    echo "${input//,/ }"
}

# ========== API: Feature Help ==========
feature_help() {
  cat <<'EOF'
Available Features:
    graph         - Enable CUDA graph
    profile       - Enable profiling (nsys)
    offload_act   - Enable activation offloading
    offload_wt    - Enable weight offloading
    offload_opt   - Enable optimizer state offloading
    fine_grained  - Enable fine-grained activation offloading

  Feature Combinations:
    Multiple features can be combined with comma:
      --features graph,profile,offload_act

    Or use multiple --feature flags:
      --feature graph --feature offload_act

  Examples:
    ./run-test.sh --test ep4-alltoall --features graph,profile
    ./run-test.sh --test ep4-alltoall --feature graph --feature offload_act
EOF
}

# ========== If run directly, show help ==========
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    feature_help
fi
