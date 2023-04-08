import torch
import torch.nn as nn
import torch.nn.functional as F

class MNF(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.aud_dim = opt.aud_dim
        self.n_classes = opt.n_classes
        embed_dim = opt.aud_dim

        self.gated_feature_composer = torch.nn.Sequential(
        torch.nn.Linear(2 * embed_dim, embed_dim // 2),
        torch.nn.BatchNorm1d(embed_dim // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(embed_dim // 2, embed_dim),
        )
        # extra audio 
        self.attn = nn.MultiheadAttention(embed_dim,num_heads = 2)

        self.trans_sv = torch.nn.Sequential(
        torch.nn.Linear( embed_dim, embed_dim),
        torch.nn.BatchNorm1d(embed_dim),
        torch.nn.ReLU(),
        )

        self.sv_cls_classifier = nn.Linear(self.aud_dim,self.n_classes)

    def forward(self, audio_feature, video_feature_map):
        """
        :param f1: other modality (e.g. audio or vision)
        :param f2: video modality
        :return:
        """

        # audio_feature = F.normalize(audio_feature, dim=-1)

        video_feature =  video_feature_map.mean(dim = 1)

        # video_feature_map = F.normalize(video_feature_map, dim=-1)

        audio_weight = self.gated_feature_composer(torch.cat([audio_feature,video_feature],dim = -1))
        audio_weight = F.sigmoid(audio_weight)

        fc_weight0 = self.gated_feature_composer[0].weight
        fc_weight1 = self.gated_feature_composer[3].weight
        
        fc_weights = [fc_weight0,fc_weight1]
        # get the selected audio feature
        selected_audio_feature = audio_weight * audio_feature

        query_audio_feature = selected_audio_feature.unsqueeze(1)
        query_audio_feature = query_audio_feature.permute(1,0,2)
        key_video_feature_map = video_feature_map.permute(1,0,2)
        selected_video_feature,attn_output_weights = self.attn(query_audio_feature,key_video_feature_map,key_video_feature_map)

        selected_video_feature = selected_video_feature.permute(1,0,2).squeeze(1)

        # TODO:change it  
        selected_video_feature = self.trans_sv(selected_video_feature)
        # use the same video feature

        selected_video_pred = self.sv_cls_classifier(selected_video_feature)


        return selected_video_pred,selected_video_feature,selected_audio_feature,fc_weights , audio_weight,attn_output_weights

def generate_mnf(opt):
    model = MNF(opt)
    return model