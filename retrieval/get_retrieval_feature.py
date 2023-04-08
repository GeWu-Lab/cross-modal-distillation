import sys 
sys.path.append(".") 
import json
import random
import os
import numpy as np
import torch
import torchvision
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from pathlib import Path
from args import parse_args
from model_utils import make_data_parallel

from models.r21d_with_featuremap import generate_video_model

from utils import Logger, worker_init_fn
import pickle

from av_val_utils import get_av_inference_utils



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

def get_retrieval_result(inference_loader,video_model,result_path,subset):
    video_model.eval()

    
    video_features = {}
    labels = {}

    for idx,batch in enumerate(inference_loader):
        video = batch['clip'].cuda()
        audio = batch['audio'].cuda()
        target = batch['target']
        video_name = batch['video_name'][0]
        print(idx)
        feature_map,video_feature,p_v = video_model(video)

        v_f = F.normalize(video_feature,dim = 1)

        video_feature = torch.mean(video_feature,dim = 0)
        
        labels[video_name] = target.cpu().numpy()
        video_features[video_name] = video_feature.detach().cpu().numpy()

    with open(os.path.join(result_path,"{}_video_features.pkl".format(subset)),"wb") as f:
        pickle.dump(video_features,f)



def main_worker(opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    cudnn.benchmark = False
    

    video_model = generate_video_model(opt)


    if opt.resume_path is not None:
        video_model = resume_model(opt.resume_path,  video_model)


    video_model = make_data_parallel(video_model, opt.device)

    inference_loader, inference_class_names = get_av_inference_utils(opt,opt.inference_subset)
    
    get_retrieval_result(inference_loader,video_model,opt.result_path,opt.inference_subset)



if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')

    main_worker(opt)


