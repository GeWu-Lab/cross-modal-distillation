import torch
import torch.nn.functional as F
from torch import nn

EPISILON=1e-10

def py_softmax(x,dim = None):
    return torch.exp(x - torch.logsumexp(x,dim = dim,keepdim = True))

def mul_softmax(x,temperature,dim = None):

    l = torch.mul(torch.exp(x),temperature)
    # e = torch.mul(x,temperature)
    # l = torch.exp(e)

    s = torch.sum(l,dim = dim,keepdim = True)
    
    softmax = l / s

    return softmax


def normal_temp_softmax(x,temperature,dim = None):
  true_value = x / temperature
  l = F.softmax(true_value,dim = dim)

  return l

class CSC_Loss(torch.nn.Module):

  def __init__(self,change_temperature = False,threshold = 0.07):
    super(CSC_Loss, self).__init__()
    self.change_temperature = change_temperature
    self.threshold = threshold

  def forward(self, video_feature, selected_video_feature,targets ,cos_similarity):
    ### cuda implementation
    video_feature = F.normalize(video_feature, dim = 1)
    selected_video_feature = F.normalize(selected_video_feature, dim = 1)

    n_batch = video_feature.shape[0]
    
    # target shape is n_batch ,it's value is number of label 
    targets_mask = targets.unsqueeze(1) - targets
    # if they have the same label the mask is 1
    label_mask = (torch.zeros_like(targets_mask) == targets_mask).float()

    ##  cos distance
    v_av_cos = torch.mm(video_feature,selected_video_feature.T)

    # define the temperature
    if self.change_temperature:
      temperature_v_av =  cos_similarity + self.threshold
      # print(temperature_v_av)
    else:
      temperature_v_av = self.threshold * torch.ones(n_batch,n_batch).cuda()
    
    final_softmax = normal_temp_softmax(v_av_cos,temperature_v_av,dim = -1)

    log_final_softmax = (- torch.log(final_softmax + EPISILON ) * label_mask).sum(dim = -1)

    v_av_loss = log_final_softmax / (label_mask.sum(1)) 
    v_av_loss = v_av_loss.mean()
    
    return v_av_loss 
