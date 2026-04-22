#!/bin/bash

# Script to calculate theoretical memory usage for DeepSeek-V3
# Usage: ./deepseek-v3.sh --case <1|2> [additional options]

# Environment variables
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
export TOKENIZERS_PARALLELISM=false

# Local rank settings
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=4

# Add megatron to PYTHONPATH
export CURRENT_PATH=$(pwd)
export PYTHONPATH=$CURRENT_PATH:$PYTHONPATH

# Default settings
GPUS_PER_NODE=4
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64
TRAIN_SAMPLES=1280

# Model architecture
PP=1
TP=1
EP=1
NUM_EXPERT=256
NUM_LAYER=61
MOE_FREQ="([0]*3+[1]*58)"
SEQ_LEN=4096

# Default dispatcher
DISPATCHER="alltoall"

# Parse arguments
params=$(getopt -o "" --long "case:,pp:,tp:,ep:,micro-batch-size:,global-batch-size:,num-expert:,num-layer:,moe-freq:,seq-length:,dispatcher:" -- "$@")
eval set -- "$params"

while true; do
    case "$1" in
        --case)
            CASE="$2"
            shift 2
            ;;
        --pp)
            PP="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --ep)
            EP="$2"
            shift 2
            ;;
        --micro-batch-size)
            MICRO_BATCH_SIZE="$2"
            shift 2
            ;;
        --global-batch-size)
            GLOBAL_BATCH_SIZE="$2"
            TRAIN_SAMPLES=1280
            shift 2
            ;;
        --num-expert)
            NUM_EXPERT="$2"
            shift 2
            ;;
        --num-layer)
            NUM_LAYER="$2"
            shift 2
            ;;
        --moe-freq)
            MOE_FREQ="$2"
            shift 2
            ;;
        --seq-length)
            SEQ_LEN="$2"
            shift 2
            ;;
        --dispatcher)
            DISPATCHER="$2"
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

# Handle case selection
if [ -n "${CASE+x}" ]; then
    case "$CASE" in
        1)
            # Case 1: 256 GPUs, 8 PP, 4 VPP, 32 DP, 32 EP
            if [ -z "${WORLD_SIZE+x}" ]; then
                WORLD_SIZE=256
            fi
            PP=8
            VPP=4
            DP=32
            EP=32
            ;;
        2)
            # Case 2: 64 GPUs, 64 DP, 32 EP
            if [ -z "${WORLD_SIZE+x}" ]; then
                WORLD_SIZE=64
            fi
            DP=64
            EP=32
            PP=1
            TP=1
            ;;
        *)
            echo "Error: invalid case '$CASE'. Valid options: 1, 2"
            exit 1
            ;;
    esac
else
    # Use default WORLD_SIZE if not set
    if [ -z "${WORLD_SIZE+x}" ]; then
        echo "Error: either --case or WORLD_SIZE must be specified"
        exit 1
    fi
fi

# Validate dispatcher
case "$DISPATCHER" in
    deepep|hybridep|alltoall|allgather)
        ;;
    *)
        echo "Error: invalid dispatcher '$DISPATCHER'. Valid options: deepep, hybridep, alltoall, allgather"
        exit 1
        ;;
esac

# Calculate DP if not set by case
if [ -z "${DP+x}" ]; then
    DP=$[$WORLD_SIZE / $TP / $PP]
fi

# Setup distributed environment
NNODES=$[$WORLD_SIZE / $GPUS_PER_NODE]

if [ $WORLD_SIZE -gt $LOCAL_WORLD_SIZE ]; then
    # Multi-node setup
    export MASTER_ADDR=${MASTER_ADDR:-localhost}
    export MASTER_PORT=${MASTER_PORT:-6000}
    export RANK=${RANK:-0}
else
    # Single-node setup
    export GLOO_SOCKET_IFNAME=eth0
    export MASTER_ADDR=localhost
    export MASTER_PORT=6000
    export NNODES=1
    export RANK=0
fi

# Build output directory
MODEL="deepseek-v3-dp$DP-tp$TP-pp$PP-ep$EP"
OUTPUT_DIR="$CURRENT_PATH/tests/analyse/memory/output/$MODEL"
mkdir -p $OUTPUT_DIR

# Copy script to output directory
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cp $SCRIPT_PATH $OUTPUT_DIR/

# Distributed arguments
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# Model parallel arguments
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

# GPT model arguments
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

# MoE arguments
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
    deepep)
        MOE_ARGS+=(
            --moe-token-dispatcher-type flex
            --moe-flex-dispatcher-backend deepep
            --moe-matmul-router-fusion
            --moe-router-fusion
            --moe-permute-fusion
        )
        ;;
    hybridep)
        MOE_ARGS+=(
            --moe-token-dispatcher-type flex
            --moe-flex-dispatcher-backend hybridep
            --moe-hybridep-num-sms 32
            --moe-router-fusion
            --moe-router-padding-for-quantization
        )
        ;;
esac

MOE_ARGS+=(
    --overlap-grad-reduce
    --overlap-param-gather
)

# Training arguments
TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples $TRAIN_SAMPLES
    --exit-duration-in-mins 220
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

# Optimizer arguments
OPTIMIZER_ARGS=(
    --use-precision-aware-optimizer
    --main-grads-dtype fp32
    --main-params-dtype fp32
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
)

# FP8 recipe arguments
FP8_RECIPE_ARGS=(
    --fp8-recipe mxfp8
    --fp8-format e4m3
    --fp8-param-gather
    --reuse-grad-buf-for-mxfp8-param-ag
)

# Recompute arguments
RECOMPUTE_ARGS=(
    --recompute-granularity selective
    --recompute-modules moe_act mlp
)

# Data arguments
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

# Logging arguments
LOGGING_ARGS=(
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval 1
    --logging-level 40
)

# Load arguments
LOAD_ARGS=(
    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
    --load None
    --save $OUTPUT_DIR/checkpoints
    --save-interval 500
    --dist-ckpt-strictness log_all
)

# Print configuration
echo "========================================="
echo "DeepSeek-V3 Theoretical Memory Analysis"
echo "========================================="
echo "WORLD_SIZE: $WORLD_SIZE"
echo "DP: $DP, TP: $TP, PP: $PP, EP: $EP"
echo "GPUS_PER_NODE: $GPUS_PER_NODE, NNODES: $NNODES"
echo "MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "SEQ_LEN: $SEQ_LEN"
echo "NUM_EXPERT: $NUM_EXPERT"
echo "DISPATCHER: $DISPATCHER"
echo "========================================="

# Run theoretical memory analysis
torchrun \
    ${DISTRIBUTED_ARGS[@]} \
    $CURRENT_PATH/tools/report_theoretical_memory.py \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${LOAD_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${FP8_RECIPE_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${MOE_ARGS[@]} \
    2>&1 | tee $OUTPUT_DIR/memory_analysis.log

echo "========================================="
echo "Memory analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="
