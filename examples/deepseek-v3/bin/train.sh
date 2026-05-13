#!/bin/bash
# train.sh - Training execution script
# Responsibility: Execute training with given parameters

MEGATRON_PATH=$(pwd)
WORKSPACE_PATH=$(pwd)
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========== 1. Environment Variables ==========
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
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH

# ========== 2. Default Configuration ==========
GPUS_PER_NODE=4
NNODES=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

# Base params (short names for internal use)
TP=1
PP=1
EP=4
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64
TRAIN_ITERS=20
TRAIN_SAMPLES=$((TRAIN_ITERS * GLOBAL_BATCH_SIZE))
TIMESTEMP=$(date +%Y-%m-%d-%H-%M-%S)

NUM_EXPERT=256
NUM_LAYER=61
MOE_FREQ="([0]*3+[1]*58)"
SEQ_LEN=4096

ENABLE_CUDA_GRAPH=false
DISPATCHER="hybridep"
PP_LAYOUT=""

# Advanced features
ENABLE_PROFILE=false
ENABLE_GRAPH=false

# Offload settings
OFFLOAD_ACTIVATION=false
OFFLOAD_WEIGHTS=false
OFFLOAD_OPTIMIZER=false
OFFLOAD_FINE_GRAINED=false
OPTIMIZER_OFFLOAD_FRACTION=1.0
CPU_OFFLOADING=false
CPU_OFFLOADING_DOUBLE_BUFFERING=false

# ========== 3. Parameter Parsing ==========
# Base params: --tensor-parallel, --pipeline-parallel, --expert-parallel, etc.
# Advanced features: --profile, --graph, --offload-act, --offload-weight, --offload-optim, --offload-fine

params=$(getopt -o "" --long \
  "tensor-parallel:,pipeline-parallel:,expert-parallel:,micro-batch-size:,global-batch-size:,num-expert:,num-layer:,moe-freq:,seq-length:,pipeline-parallel-layout:,dispatcher:,train-iters:,profile,graph,offload-act,offload-weight,offload-optim,offload-fine,optimizer-offload-fraction:" \
  -- "$@")
eval set -- "$params"

while true; do
    case "$1" in
        --tensor-parallel) TP="$2"; shift 2 ;;
        --pipeline-parallel) PP="$2"; shift 2 ;;
        --expert-parallel) EP="$2"; shift 2 ;;
        --micro-batch-size) MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --global-batch-size) GLOBAL_BATCH_SIZE="$2"; TRAIN_SAMPLES=$((TRAIN_ITERS * GLOBAL_BATCH_SIZE)); shift 2 ;;
        --num-expert) NUM_EXPERT="$2"; shift 2 ;;
        --num-layer) NUM_LAYER="$2"; shift 2 ;;
        --moe-freq) MOE_FREQ="$2"; shift 2 ;;
        --seq-length) SEQ_LEN="$2"; shift 2 ;;
        --pipeline-parallel-layout) PP_LAYOUT="$2"; shift 2 ;;
        --dispatcher) DISPATCHER="$2"; shift 2 ;;
        --train-iters) TRAIN_ITERS="$2"; TRAIN_SAMPLES=$((TRAIN_ITERS * GLOBAL_BATCH_SIZE)); shift 2 ;;
        --profile) ENABLE_PROFILE=true; shift ;;
        --graph) ENABLE_GRAPH=true; shift ;;
        --offload-act) OFFLOAD_ACTIVATION=true; CPU_OFFLOADING=true; CPU_OFFLOADING_DOUBLE_BUFFERING=true; shift ;;
        --offload-weight) OFFLOAD_WEIGHTS=true; CPU_OFFLOADING=true; CPU_OFFLOADING_DOUBLE_BUFFERING=true; shift ;;
        --offload-optim) OFFLOAD_OPTIMIZER=true; shift ;;
        --offload-fine) OFFLOAD_FINE_GRAINED=true; shift ;;
        --optimizer-offload-fraction) OPTIMIZER_OFFLOAD_FRACTION="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# ========== 4. DeepEP Installation Check ==========
get_compute_capability() {
    local compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv -i 0 2>/dev/null | tail -1)
    echo "$compute_cap"
}

check_and_install_deep_ep() {
    local dispatcher_type=$1
    local original_path=$(pwd)
    local third_party_path="$MEGATRON_PATH/third_party"
    local deep_ep_version="1.2.1+9af0e0d"

    mkdir -p "$third_party_path"
    local compute_cap=$(get_compute_capability)

    case "$dispatcher_type" in
        deepep)
            local current_version=$(pip3 list | grep deep_ep | awk '{print $2}')
            if [[ "$current_version" == "$deep_ep_version" ]]; then
                return 0
            else
                cd "$third_party_path"
                if [[ ! -d "DeepEP" ]]; then
                    git config --global http.sslverify false
                    git clone https://github.com/deepseek-ai/DeepEP.git DeepEP
                fi
                cd DeepEP
                git checkout v1.2.1
                TORCH_CUDA_ARCH_LIST="$compute_cap" pip3 install --no-build-isolation .
                cd "$original_path"
                return $?
            fi
            ;;
        hybridep)
            if python -c "from deep_ep import HybridEpConfigInstance" 2>/dev/null; then
                return 0
            else
                cd "$third_party_path"
                if [[ ! -d "HybridEP" ]]; then
                    git config --global http.sslverify false
                    git clone https://github.com/deepseek-ai/DeepEP.git HybridEP
                fi
                cd HybridEP
                git checkout hybrid-ep
                TORCH_CUDA_ARCH_LIST="$compute_cap" pip3 install --no-build-isolation .
                cd "$original_path"
                return $?
            fi
            ;;
        *)
            echo "Error: unknown dispatcher type '$dispatcher_type'"
            return 1
            ;;
    esac
}

case "$DISPATCHER" in
    deepep|hybridep)
        check_and_install_deep_ep "$DISPATCHER" || { echo "Failed to install deep_ep"; exit 1; }
        ;;
    alltoall|allgather)
        ;;
    *)
        echo "Error: invalid dispatcher '$DISPATCHER'. Valid: deepep, hybridep, alltoall, allgather"
        exit 1
        ;;
esac

# ========== 5. Derived Variables ==========
DP=$((WORLD_SIZE / TP / PP))

if [ -n "${WORLD_SIZE+x}" ] && [ $WORLD_SIZE -gt $LOCAL_WORLD_SIZE ]; then
    MODEL="dlc-deepseek-v3-dp${DP}-tp${TP}-pp${PP}-ep${EP}-mbs${MICRO_BATCH_SIZE}-gbs${GLOBAL_BATCH_SIZE}-expert${NUM_EXPERT}-layer${NUM_LAYER}-seq${SEQ_LEN}"
    BASE_PATH=$WORKSPACE_PATH/logs-temp/$MODEL
    LOGS_PATH=$WORKSPACE_PATH/logs-temp/$MODEL/rank$RANK
else
    export GLOO_SOCKET_IFNAME=eth0
    export NNODES=1
    export RANK=0
    MODEL="dsw-deepseek-v3-dp${DP}-tp${TP}-pp${PP}-ep${EP}-mbs${MICRO_BATCH_SIZE}-gbs${GLOBAL_BATCH_SIZE}-expert${NUM_EXPERT}-layer${NUM_LAYER}-seq${SEQ_LEN}"
    BASE_PATH=$WORKSPACE_PATH/logs-temp/$MODEL
    LOGS_PATH=$WORKSPACE_PATH/logs-temp/$MODEL
fi

TENSORBOARD_PATH=$LOGS_PATH/tensorboard
CHECKPOINTS_PATH=$LOGS_PATH/checkpoints

# ========== 6. Log Setup ==========
rm -rf $LOGS_PATH
mkdir -p $WORKSPACE_PATH/logs
mkdir -p $TENSORBOARD_PATH
mkdir -p $CHECKPOINTS_PATH
mkdir -p ./data-cache

cp "${BASH_SOURCE[0]}" $LOGS_PATH/

# ========== 7. Argument Arrays ==========
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

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

[ -n "$PP_LAYOUT" ] && MODEL_PARALLEL_ARGS+=(--pipeline-model-parallel-layout "$PP_LAYOUT")

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
    --overlap-grad-reduce
    --overlap-param-gather
)

# Dispatcher-specific
case "$DISPATCHER" in
    deepep)
        MOE_ARGS+=(
          --moe-token-dispatcher-type flex
          --moe-flex-dispatcher-backend deepep
          --moe-router-fusion
          --moe-permute-fusion
          --moe-router-padding-for-quantization
        )
        if [ "$ENABLE_GRAPH" = true ]; then
          MOE_ARGS+=(
            --cuda-graph-impl transformer_engine
            --cuda-graph-scope attn moe_router moe_preprocess
          )
        fi
        ;;
    hybridep)
        MOE_ARGS+=(
          --moe-token-dispatcher-type flex
          --moe-flex-dispatcher-backend hybridep
          --moe-hybridep-num-sms 32
          --moe-router-fusion
          --moe-router-padding-for-quantization
        )
        if [ "$ENABLE_GRAPH" = true ]; then
          MOE_ARGS+=(
            --cuda-graph-impl transformer_engine
            --cuda-graph-scope attn moe_router moe_preprocess
          )
        fi
        ;;
    alltoall)
        MOE_ARGS+=(
          --moe-token-dispatcher-type alltoall
          --moe-router-padding-for-quantization
        )
        if [ "$ENABLE_GRAPH" = true ]; then
          MOE_ARGS+=(
            --cuda-graph-impl transformer_engine
            --cuda-graph-scope attn
          )
        fi
        ;;
    allgather)
        MOE_ARGS+=(--moe-token-dispatcher-type allgather)
        if [ "$ENABLE_GRAPH" = true ]; then
          MOE_ARGS+=(
            --cuda-graph-impl transformer_engine
            --cuda-graph-scope attn
          )
        fi
        ;;
esac

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
    --enable-experimental
)

OPTIMIZER_ARGS=(
    --use-precision-aware-optimizer
    --main-grads-dtype fp32
    --main-params-dtype fp32
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
)

FP8_RECIPE_ARGS=(
    --fp8-recipe mxfp8
    --fp8-format e4m3
    --fp8-param-gather
    --reuse-grad-buf-for-mxfp8-param-ag
)

RECOMPUTE_ARGS=(
    --recompute-granularity selective
    --recompute-modules moe_act mlp
)

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

LOGGING_ARGS=(
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --log-interval 1
    --logging-level 40
    --tensorboard-dir $TENSORBOARD_PATH
    --record-memory-history
    --memory-snapshot-path $BASE_PATH
)

LOAD_ARGS=(
    --no-load-optim
    --no-load-rng
    --auto-detect-ckpt-format
    --load None
    --save $CHECKPOINTS_PATH
    --save-interval 500
    --dist-ckpt-strictness log_all
)

# Offloading
OFFLOADING_ARGS=()
if [ "$OFFLOAD_FINE_GRAINED" = true ]; then
    OFFLOADING_ARGS+=(
        --fine-grained-activation-offloading
        --offload-modules "attn_norm" "core_attn" "attn_proj" "mlp_norm" "expert_fc1" "moe_act"
    )
fi

if [ "$CPU_OFFLOADING" = true ]; then
    OFFLOADING_ARGS+=(
        --cpu-offloading
        --cpu-offloading-num-layers $NUM_LAYER
    )
    [ "$OFFLOAD_ACTIVATION" = false ] && OFFLOADING_ARGS+=(--cpu-offloading-activation)
    [ "$OFFLOAD_WEIGHTS" = true ] && OFFLOADING_ARGS+=(--cpu-offloading-weights)
    [ "$CPU_OFFLOADING_DOUBLE_BUFFERING" = true ] && OFFLOADING_ARGS+=(--cpu-offloading-double-buffering)
fi

if [ "$OFFLOAD_OPTIMIZER" = true ]; then
    OFFLOADING_ARGS+=(
        --optimizer-offload
        --optimizer-offload-fraction $OPTIMIZER_OFFLOAD_FRACTION
        --overlap-cpu-optimizer-d2h-h2d
    )
fi

# Profile
PROFILE_ARGS=()
NSYS_ARGS=()
if [ "$ENABLE_PROFILE" = true ] && [ $RANK -eq 0 ]; then
    export NVTE_NVTX_ENABLED=1
    PROFILE_ARGS=(
        --profile
        --profile-ranks 0 1 2 3
        --profile-step-start 16
        --profile-step-end 18
    )
    NSYS_ARGS=(
        nsys profile -s none -t nvtx,cuda,cudnn,cublas
        --cudabacktrace=all
        --cuda-graph-trace=node
        --python-backtrace=cuda
        --wait all
        -o $LOGS_PATH/$MODEL-$TIMESTEMP.nsys-rep
        --force-overwrite true
        --capture-range=cudaProfilerApi
        --capture-range-end=stop
    )
fi

# ========== 8. Execute Training ==========
echo "train.sh: Starting training with config:"
echo "  TP=$TP, PP=$PP, EP=$EP, DP=$DP"
echo "  MBS=$MICRO_BATCH_SIZE, GBS=$GLOBAL_BATCH_SIZE"
echo "  Layers=$NUM_LAYER, Experts=$NUM_EXPERT, Seq=$SEQ_LEN"
echo "  Dispatcher=$DISPATCHER, Graph=$ENABLE_GRAPH, Profile=$ENABLE_PROFILE"
echo "  Offload: act=$OFFLOAD_ACTIVATION, weight=$OFFLOAD_WEIGHTS, optim=$OFFLOAD_OPTIMIZER, fine=$OFFLOAD_FINE_GRAINED"

numarun \
    ${NSYS_ARGS[@]} \
    torchrun \
    ${DISTRIBUTED_ARGS[@]} \
    ./pretrain_gpt.py \
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
    ${OFFLOADING_ARGS[@]} \
    ${PROFILE_ARGS[@]} \
    > $LOGS_PATH/train.log 2>&1

# ========== 9. Archive Results ==========
if [ $RANK -eq 0 ]; then
    if [ -n "${WORLD_SIZE+x}" ] && [ $WORLD_SIZE -gt $LOCAL_WORLD_SIZE ]; then
      sleep 60
    fi
    mv $BASE_PATH $WORKSPACE_PATH/logs/$MODEL-$TIMESTEMP 2>/dev/null || true
fi

echo "train.sh: Training completed. Logs: $LOGS_PATH/train.log"
