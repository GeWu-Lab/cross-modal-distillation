#!/bin/bash

VIDEO_PATH=datasets/activitynet/video_set_4fps
AUDIO_PATH=datasets/activitynet/audio_features
ANNOTATION_PATH=datasets/activitynet/annotation.json

RESULT_DIR=ckpt
FOLDER=train_activitynet_ckpt

export CUDA_VISIBLE_DEVICES=0,1,2

OPTIM_TYPE=adam
LOG_PATH=logs/train_activitynet.log
mkdir ${RESULT_DIR}/${FOLDER}

python multimodal_main.py \
--video_dir ${VIDEO_PATH} \
--audio_dir ${AUDIO_PATH} \
--annotation_path ${ANNOTATION_PATH} \
--result_path ${RESULT_DIR}/${FOLDER} \
--n_classes 200 \
--subset training \
--batch_size 64 --n_threads 8 --checkpoint 100 \
--sample_t_stride 1 \
--train_crop random \
--lr_scheduler multistep \
--optim_type ${OPTIM_TYPE} \
--manual_seed 3 \
--tensorboard \
--learning_rate 3e-4 --weight_decay 5e-4 \
--n_epochs=200 --use_audio \
--train_subset training \
--val_freq 5 \
--train \
--threshold 0.2 \
--use_norm \
--use_triplet \
--with_adj

