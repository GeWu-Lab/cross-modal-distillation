import torch
from torch import nn

def make_data_parallel(model, device):
    if device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()
    return model