#!/bin/bash
# Script to analyze theoretical GEMM flops for DeepSeek-V3 model

# =============================================================================
# Environment Variables
# =============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0
export NVLINK_DOMAIN_SIZE=72
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export NCCL_DEBUG=VERSION
export NCCL_GRAPH_REGISTER=0

# Local rank
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=4
export TOKENIZERS_PARALLELISM=false

# Add megatron to PYTHONPATH
export CURRENT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH=$CURRENT_PATH:$PYTHONPATH

# =============================================================================
# Default Settings
# =============================================================================
GPUS_PER_NODE=4
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64
TRAIN_ITERS=20
TRAIN_SAMPLES=$[$TRAIN_ITERS * $GLOBAL_BATCH_SIZE]
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)

# DeepSeek-V3 Model Parameters
NUM_LAYER=61
NUM_EXPERT=256
MOE_FREQ="([0]*[1]*58)"
SEQ_LEN=4096

# Default dispatcher (alltoall as requested)
DISPATCHER="alltoall"

# Default parallel settings
CASE=""
PP=1
TP=1
EP=32

# =============================================================================
# Argument Parsing
# =============================================================================
params=$(getopt -o "" --long "case:,dispatcher:,micro-batch-size:,global-batch-size:" -- "$@")
eval set -- "$params"

while true; do
    case "$1" in
        --case)
            CASE="$2"
            shift 2
            ;;
        --dispatcher)
            DISPATCHER="$2"
            shift 2
            ;;
        --micro-batch-size)
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --global-batch-size)
            GLOBAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate dispatcher
# =============================================================================
case "$DISPATCHER" in
    deepep|hybridep|alltoall|allgather)
        ;;
    *)
        echo "Error: invalid dispatcher '$DISPATCHER'. Valid options: deepep, hybridep, alltoall, allgather"
        exit 1
        ;;
esac

# =============================================================================
# Configure Parallel Strategy Based on Case
# =============================================================================
if [ -z "$CASE" ]; then
    echo "Error: --case parameter is required"
    echo "Usage: $0 --case <1|2> [--dispatcher <dispatcher>] [--micro-batch-size <size>] [--global-batch-size <size>]"
    exit 1
fi

case "$CASE" in
    1)
        echo "=== Case 1: 256 GPUs, PP=8, VPP=4, DP=EP=32 ==="
        WORLD_SIZE=256
        PP=8
        VPP=4
        EP=32
        DP=32
        ;;
    2)
        echo "=== Case 2: 64 GPUs, DP=64, EP=32 ==="
        WORLD_SIZE=64
        PP=1
        TP=1
        EP=32
        DP=64
        ;;
    *)
        echo "Error: invalid case '$CASE'. Valid options: 1, 2"
        exit 1
        ;;
esac

# Calculate derived values
NNODES=$[$WORLD_SIZE / $GPUS_PER_NODE]

echo "Configuration:"
echo "  World size: $WORLD_SIZE"
echo "  Nodes: $NNODES"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  PP: $PP"
if [ -n "$VPP" ]; then
    echo "  VPP: $VPP"
fi
echo "  TP: $TP"
echo "  EP: $EP"
echo "  DP: $DP"
echo "  Dispatcher: $DISPATCHER"
echo ""

# =============================================================================
# Distributed Args
# =============================================================================
export GLOO_SOCKET_IFNAME=eth0
export MASTER_ADDR=localhost
export MASTER_PORT=6000
export NNODES=1
export RANK=0

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1
    --node_rank 0
    --master_addr localhost
    --master_port 6000
)

# =============================================================================
# Model Parallel Args
# =============================================================================
MODEL_PARALLEL_ARGS=(
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --expert-model-parallel-size $EP
    --context-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

if [ -n "$VPP" ]; then
    MODEL_PARALLEL_ARGS+=("--virtual-pipeline-model-parallel-size" "$VPP")
fi

# =============================================================================
# GPT Model Args
# =============================================================================
GPT_MODEL_ARGS=(
    --use-mcore-models
    --use-flash-attn
    --disable-bias-linear
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --transformer-impl transformer_engine
    --seq-length $SEQ_LEN
    --no-mmap-bin-files
    --num-layers $NUM_LAYER
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --kv-channels 128
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 10000
    --make-vocab-size-divisible-by 3232
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --multi-latent-attention
    --clip-grad 1.0
    --weight-decay 0.1
    --qk-layernorm
    --te-rng-tracker
    --bf16
    --adam-beta1 0.9
    --adam-beta2 0.95
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0
)

# =============================================================================
# MoE Args
# =============================================================================
MOE_ARGS=(
    --num-experts $NUM_EXPERT
    --moe-layer-freq $MOE_FREQ
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-4
    --moe-router-group-topk 4
    --moe-router-num-groups 8
    --moe-router-pre-softmax
    --moe-router-topk-scaling-factor 2.5
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    --moe-router-force-load-balancing
)

# Add dispatcher-specific arguments
case "$DISPATCHER" in
    deepep)
        MOE_ARGS+=(
            --moe-token-dispatcher-type flex
            --moe-flex-dispatcher-backend deepep
            --moe-router-fusion
            --moe-permute-fusion
            --cuda-graph-impl transformer_engine
            --cuda-graph-scope attn moe_router moe_preprocess
            --moe-router-padding-for-quantization
        )
        ;;
    hybridep)
        MOE_ARGS+=(
            --moe-token-dispatcher-type flex
            --moe-flex-dispatcher-backend hybridep
            --moe-hybridep-num-sms 32
            --cuda-graph-impl transformer_engine
            --cuda-graph-scope attn moe_router moe_preprocess
            --moe-router-fusion
            --moe-router-padding-for-quantization
        )
        ;;
    alltoall)
        MOE_ARGS+=(
            --moe-token-dispatcher-type alltoall
            --moe-router-padding-for-quantization
        )
        ;;
    allgather)
        MOE_ARGS+=(
            --moe-token-dispatcher-type allgather
        )
        ;;
esac

MOE_ARGS+=(
    --overlap-grad-reduce
    --overlap-param-gather
)

# =============================================================================
# Training Args
# =============================================================================
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples $TRAIN_SAMPLES
    --exit-duration-mins 220
    --no-save-optim
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 10
    --enable-experimental
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --lr-decay-samples 584765624
    --lr-warmup-samples 1536000
    --lr-warmup-init 3.9e-7
    --lr 3.9e-6
    --min-lr 3.9e-7
    --lr-decay-style cosine
    --eval-iters 0
    --eval-interval 200
    --init-method-std 0.02
)

# =============================================================================
# Optimizer Args
# =============================================================================
OPTIMIZER_ARGS=(
    --use-precision-aware-optimizer
    --main-grads-dtype fp32
    --main-params-dtype fp32
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
)

# =============================================================================
# FP8 Recipe Args
# =============================================================================
FP8_RECIPE_ARGS=(
    --fp8-recipe mxfp8
    --fp8-format e4m3
    --fp8-param-gather
    --reuse-grad-buf-for-mxfp8-param-ag
)

# =============================================================================
# Recompute Args
# =============================================================================
RECOMPUTE_ARGS=(
    --recompute-granularity selective
    --recompute-modules moe_act mlp
)

# =============================================================================
# Data Args (using mock data)
# =============================================================================
DATA_ARGS=(
    --data-cache-path ./data-cache
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model unsloth/DeepSeek-V3
    --mock-data
    --vocab-file ./datasets/vocab.json
    --merge-file ./datasets/merges.txt
    --split 99,1,0
    --num-workers 6
    --no-create-attention-mask-in-dataloader
)

# =============================================================================
# Logging Args
# =============================================================================
LOGGING_ARGS=(
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval 1
    --logging-level 40
)

# =============================================================================
# Load Args
# =============================================================================
LOAD_ARGS=(
    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
    --load None
    --no-save
)

# =============================================================================
# Run the theoretical flops calculation
# =============================================================================
echo "Running theoretical GEMM flops calculation..."
echo ""

python3 $CURRENT_PATH/tools/report_theoretical_calculation.py \
    ${DISTRIBUTED_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOAD_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${FP8_RECIPE_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${MOE_ARGS[@]}

echo ""
echo "Calculation completed for case $CASE"
