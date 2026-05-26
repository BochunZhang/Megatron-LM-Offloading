#!/bin/bash

# DeepSeek V3 model training script for 4xGB200 GPU device
# EP=4, 128 experts (32 experts per device)
# Reduced layers to fit within memory constraints

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
DATA_PATH=$3 #<Specify path and file prefix>

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# DeepSeek V3 Model Configuration (scaled for 4 GPU GB200)
DEEPSEEK_MODEL_ARGS=(
    --use-mcore-models
    --num-layers 14
    --hidden-size 7168
    --ffn-hidden-size 18432
    --num-attention-heads 128
    --kv-channels 128
    --seq-length 4096
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-base 10000
    --make-vocab-size-divisible-by 3232
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --multi-latent-attention
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --qk-layernorm

    # MLA (Multi-Latent Attention) configuration
    --q-lora-rank 1536
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --rotary-scaling-factor 40
    --mscale 1.0
    --mscale-all-dim 1.0

    # MoE (Mixture-of-Experts) configuration
    --num-experts 128
    --moe-layer-freq [0]*3+[1]*11
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type seq_aux_loss
    --moe-router-topk 8
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend hybridep
    --moe-router-pre-softmax
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-4
    --moe-router-group-topk 4
    --moe-router-num-groups 8
    --moe-router-topk-scaling-factor 2.5
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-router-dtype fp32
    --moe-permute-fusion
    --moe-router-force-load-balancing



    # MTP (Multi-Token Prediction) configuration
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 512
    --train-samples 24414062
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --bf16
    --lr 1e-5
    --lr-decay-style cosine
    --min-lr 1e-6
    --lr-warmup-samples 1536000
    --lr-warmup-init 1e-7
    --lr-decay-samples 24413696
    --init-method-std 0.02
    --manual-gc
    --manual-gc-interval 10
    --recompute-granularity selective
    --recompute-modules mlp moe mla_up_proj layernorm
    --no-check-for-nan-in-loss-and-grad
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --expert-model-parallel-size 4
    --expert-tensor-parallel-size 1
)

OPTIMIZER_ARGS=(
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --use-flash-attn
    --disable-bias-linear
    --sequence-parallel
    --transformer-impl transformer_engine
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model deepseek-ai/DeepSeek-V3
    --split 99,1,0
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --num-workers 4
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 100
    --eval-interval 100
    --eval-iters 32
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --log-num-zeros-in-grad
    --log-params-norm
    --log-validation-ppl-to-tensorboard
    --log-throughput
    --logging-level 40
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${DEEPSEEK_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
