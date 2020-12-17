#!/bin/bash

 export PYTHONPATH=/workspace/models/PaddleNLP
 export DATA_DIR=/workspace/models/bert_data/
 export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
 # 设置以下环境变量为您所用训练机器的IP地址
 export TRANER_IPS="10.10.0.1,10.10.0.2,10.10.0.3,10.10.0.4"

 batch_size=${1:-32}
 use_amp=${2:-"True"}
 max_steps=${3:-500}
 logging_steps=${4:-20}

 CMD="python3.7 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 --ips $TRAINER_IPS ./run_pretrain.py"

 $CMD \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size $batch_size   \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --input_dir $DATA_DIR \
    --output_dir ./tmp2/ \
    --logging_steps $logging_steps \
    --save_steps 50000 \
    --max_steps $max_steps \
    --use_amp $use_amp\
    --enable_addto True