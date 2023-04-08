import json
import random
import os
import numpy as np
import torch

from torch.backends import cudnn
from pathlib import Path
from args import parse_args
from model_utils import make_data_parallel

from train_utils import get_train_module_utils

from models import modality_noise_filter
from models.r21d_with_featuremap import generate_video_model
from train import train
from utils import Logger, worker_init_fn
from val_utils import get_val_utils,get_inference_utils

from inference import inference


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_args()


    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.begin_epoch = 1

    return opt

def resume_model(resume_path,  model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    return model

def resume_distillation_model(resume_path, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['audio_state_dict'])
    else:
        model.load_state_dict(checkpoint['audio_state_dict'])
    return model


def main_worker(opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    cudnn.benchmark = False

    video_model = generate_video_model(opt)
    

    distillation_model = modality_noise_filter.generate_mnf(opt)

    params = {
        "video_params":video_model.parameters(),
        "audio_params":distillation_model.parameters()
    }

    if opt.resume_path is not None:
        video_model = resume_model(opt.resume_path,  video_model)
        distillation_model = resume_distillation_model(opt.resume_path, distillation_model)

    video_model = make_data_parallel(video_model, opt.device)
    distillation_model = make_data_parallel(distillation_model,opt.device)

    if opt.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None


    if opt.train:
        (train_loader, train_sampler, train_logger, train_batch_logger, optimizers,schedulers) = get_train_module_utils(opt, params)
        val_loader, val_logger = get_val_utils(opt)
        train(train_loader,val_loader,video_model,distillation_model,optimizers,train_logger,val_logger,train_batch_logger,tb_writer,schedulers,opt)
    elif opt.inference:
        inference_loader, inference_class_names = get_inference_utils(opt,opt.inference_subset)
        inference_result_path = opt.result_path / '{}.json'.format(opt.inference_subset)
        inference(inference_loader, video_model, inference_result_path,
                            inference_class_names, opt.inference_no_average,
                            opt.output_topk)

if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    main_worker(opt)