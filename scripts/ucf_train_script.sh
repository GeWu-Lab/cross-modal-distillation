#!/bin/bash

VIDEO_PATH=datasets/ucf_51/video_set_4fps
AUDIO_PATH=datasets/ucf_51/audio_features
ANNOTATION_PATH=datasets/ucf_51/annotation.json

RESULT_DIR=ckpt
FOLDER=train_ucf51_ckpt

export CUDA_VISIBLE_DEVICES=2

OPTIM_TYPE=adam
LOG_PATH=logs/ucf51.log
mkdir ${RESULT_DIR}/${FOLDER}

python multimodal_main.py \
--video_dir ${VIDEO_PATH} \
--audio_dir ${AUDIO_PATH} \
--annotation_path ${ANNOTATION_PATH} \
--result_path ${RESULT_DIR}/${FOLDER} \
--n_classes 51 \
--subset training \
--batch_size 32 --n_threads 16 --checkpoint 100 \
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
--threshold 0.07 \
--use_norm \
--use_triplet \
--with_adj 
