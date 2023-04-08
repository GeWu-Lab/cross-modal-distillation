import copy
import torch
from torch.utils.data.dataloader import default_collate
from .videodataset import VideoDataset
from .av_dataset import AudioVideoDataset
import os

class VideoDatasetMultiClips(VideoDataset):

    def __loading(self, path, video_frame_indices):
        clips = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))

        return clips

    def __getitem__(self, index):
        video_name = self.dataset[index]['video']
        video_path = os.path.join(self.video_dir,video_name + ".hdf5")
        
        video_frame_indices = self.dataset[index]['frame_indices']
        if self.temporal_transform is not None:
            video_frame_indices = self.temporal_transform(video_frame_indices)

        clips = self.__loading(video_path, video_frame_indices)

        target = self.dataset[index]['label']

        targets = [target for _ in range(len(clips))]

        sample = {
            'video_name' : video_name,
            'clip' : clips,
            'target' : targets,
            'selected':index
        }

        return sample


