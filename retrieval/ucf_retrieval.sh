#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

VIDEO_PATH=/datasets/ucf_51/video_set_4fps
AUDIO_PATH=/datasets/ucf_51/audio_features
ANNOTATION_PATH=datasets/ucf_51/annotation.json

OPTIM_TYPE=adam
RESULT_DIR=ckpt
FOLDER=ucf51_ckpt_demo

python retrieval/get_retrieval_feature.py \
--video_dir ${VIDEO_PATH} \
--audio_dir ${AUDIO_PATH} \
--annotation_path ${ANNOTATION_PATH} \
--result_path ${RESULT_DIR}/${FOLDER} \
--resume_path ${RESULT_DIR}/${FOLDER}/save_model.pth \
--n_classes 51 \
--sample_t_stride 1 \
--n_threads 4  --output_topk 5 --inference_batch_size 1 \
--inference_subset validation

python retrieval/get_retrieval_feature.py \
--video_dir ${VIDEO_PATH} \
--audio_dir ${AUDIO_PATH} \
--annotation_path ${ANNOTATION_PATH} \
--result_path ${RESULT_DIR}/${FOLDER} \
--resume_path ${RESULT_DIR}/${FOLDER}/save_model.pth \
--n_classes 51 \
--sample_t_stride 1 \
--n_threads 4  --output_topk 5 --inference_batch_size 1 \
--inference_subset training
