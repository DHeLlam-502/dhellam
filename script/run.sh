#!/bin/bash
export NVTE_FRAMEWORK=pytorch
cd ../search/
bash search.sh
cd ../script/
_GPUS_PER_NODE=8
_MASTER_ADDR=localhost
_MASTER_PORT=12344
_NNODES=1
_NCCL_SOCKET_IFNAME=eth0

_TENSOR_MODEL_PARALLEL_SIZE=8
_CONTEXT_PARALLEL_SIZE=1
_PIPELINE_MODEL_PARALLEL_SIZE=1

_CODE=../3rdparty/Megatron-LM/pretrain_gpt.py # [The Path of Your Pretraining Code]
_DATA_PATH=../data/llama-data # [The Path of Your Dataset]

_DHELLAM=true
_SCHEDULE_CONFIG=../search/config/best.json

_MTYPE=llama
_NUM_LAYERS=8
_HIDDEN_SIZE=8192
_FFN_HIDDEN_SIZE=28672
_NUM_ATTENTION_HEADS=64
_SEQ_LENGTH=8192
_VOCAB_SIZE=32000

_GLOBAL_BATCH_SIZE=16
_MICRO_BATCH_SIZE=4
_TRAIN_ITERS=2

_NODE_RANK=0
# export CUDA_VISIBLE_DEVICES=7
export NCCL_SOCKET_IFNAME=${_NCCL_SOCKET_IFNAME}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NVTE_FUSED_ATTN=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
# export NCCL_NTHREADS=256
# export NCCL_MAX_NCHANNELS=2
export GLOO_SOCKET_IFNAME=${_NCCL_SOCKET_IFNAME}
/sbin/ldconfig
GPUS_PER_NODE=$_GPUS_PER_NODE
MASTER_ADDR=$_MASTER_ADDR
MASTER_PORT=$_MASTER_PORT
NUM_NODES=$_NNODES
NODE_RANK=$_NODE_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
DATASET_PATH=$_DATA_PATH

GPT_MODEL_ARGS=(
    --num-layers $_NUM_LAYERS
    --hidden-size $_HIDDEN_SIZE
    --num-attention-heads $_NUM_ATTENTION_HEADS
    --seq-length $_SEQ_LENGTH
    --max-position-embeddings $_SEQ_LENGTH
    --transformer-impl transformer_engine
    --vocab-size $_VOCAB_SIZE
)

if [ "$_MTYPE" = "gpt" ]; then
    # GPT-datasets
    VOCAB_FILE=$DATASET_PATH/gpt2-vocab.json #<Specify path to file>/gpt2-vocab.json
    MERGE_FILE=$DATASET_PATH/gpt2-merges.txt #<Specify path to file>/gpt2-merges.txt
    DATA_PATH=$DATASET_PATH/my-gpt2_text_document #<Specify path and file prefix>_text_document
    DATA_ARGS=(
        --data-path $DATA_PATH 
        --vocab-file $VOCAB_FILE 
        --merge-file $MERGE_FILE 
        --split 949,50,1
        --tokenizer-type GPT2BPETokenizer
    )
elif [ "$_MTYPE" = "llama" ]; then
    # Llama-datasets
    DATA_PATH=$DATASET_PATH/oscar-en-10k-meg-llama_text_document #<Specify path and file prefix>_text_document
    TOKENIZER_PATH=${DATASET_PATH}/tokenizer.model
    DATA_CACHE_PATH=${DATASET_PATH}/cache
    DATA_ARGS=(
        --data-cache-path ${DATA_CACHE_PATH}
        --tokenizer-type Llama2Tokenizer
        --tokenizer-model ${TOKENIZER_PATH}
        --data-path $DATA_PATH 
        --split 949,50,1
    )
    GPT_MODEL_ARGS+=(
        --ffn-hidden-size $_FFN_HIDDEN_SIZE
        --swiglu
    )
else
    echo "unsupported model type: $_MTYPE"
    exit 1
fi


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
    --node_rank $NODE_RANK
)



TRAINING_ARGS=(
    --global-batch-size $_GLOBAL_BATCH_SIZE
    --micro-batch-size $_MICRO_BATCH_SIZE
    --train-iters $_TRAIN_ITERS
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
    --use-mcore-models
    --distributed-backend nccl
    --initial-loss-scale 65536
    --disable-bias-linear
    --fp16
    --use-distributed-optimizer
    --use-flash-attn
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $_TENSOR_MODEL_PARALLEL_SIZE
    --context-parallel-size $_CONTEXT_PARALLEL_SIZE
	--pipeline-model-parallel-size $_PIPELINE_MODEL_PARALLEL_SIZE 
    --distributed-timeout-minutes 2
    --sequence-parallel
    --attention-mask-cuda-init
)


EVAL_AND_LOGGING_ARGS=(
    --log-throughput
    --timing-log-level 2
    --log-interval 1
    --save-interval 10000
    --eval-interval 10000
    --eval-iters 0
)
if [ "$_DHELLAM" = "true" ]; then
    TRAINING_ARGS+=(
        --dhellam
        --schedule-config $_SCHEDULE_CONFIG
        )  # add dhellam
fi
torchrun ${DISTRIBUTED_ARGS[@]} $_CODE \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
