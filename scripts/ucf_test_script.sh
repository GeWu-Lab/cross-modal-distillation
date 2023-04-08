#!/bin/bash

VIDEO_PATH=datasets/ucf_51/video_set_4fps
AUDIO_PATH=datasets/ucf_51/audio_features
ANNOTATION_PATH=datasets/ucf_51/annotation.json

RESULT_DIR=ckpt
FOLDER=ucf51_ckpt

export CUDA_VISIBLE_DEVICES=0,1,2

OPTIM_TYPE=adam
LOG_PATH=logs/ucf51.log
mkdir ${RESULT_DIR}/${FOLDER}

python multimodal_main.py \
--video_dir ${VIDEO_PATH} \
--annotation_path ${ANNOTATION_PATH} \
--result_path ${RESULT_DIR}/${FOLDER} \
--resume_path ${RESULT_DIR}/${FOLDER}/save_model.pth \
--n_classes 51 \
--sample_t_stride 1 \
--n_threads 8  --output_topk 5 --inference_batch_size 1 \
--inference  \

python -m eval_accuracy ${ANNOTATION_PATH} ${RESULT_DIR}/${FOLDER}/validation.json --subset validation -k 1 
