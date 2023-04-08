import json
import os
import torch
import torch.utils.data as data
from pathlib import Path
from .loader import VideoLoader
from .loader import AudioFeatureLoader
from random import randrange
import numpy as np
import h5py



def get_dataset(annotation_data,subset):
    video_names = []
    video_labels = []

    for key in annotation_data.keys():
        if annotation_data[key]['subset'] == subset:
            video_names.append(key)
            video_labels.append(annotation_data[key]['label'])
    return video_names,video_labels


class AudioVideoDataset(data.Dataset):
    def __init__(self,
                video_dir,
                audio_dir,
                annotation_path,
                subset,
                fps,
                spatial_transform=None,
                temporal_transform=None,
                video_loader = None):
        
        self.fps = fps
        self.video_dir = video_dir
        self.audio_dir = audio_dir

        self.dataset,self.idx_to_class,self.n_videos = self.__make_dataset(video_dir,annotation_path,subset)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

        self.loader = video_loader
        
    
    def __make_dataset(self,video_dir,annotation_path,subset):
        with open(annotation_path) as f:
            annotation_data = json.load(f)
        class_labels = annotation_data['labels']
        annotation_data = annotation_data['database']
        # get the video name and labels
        video_names , video_labels = get_dataset(annotation_data,subset)
        

        # get the label and idx map
        class_to_idx = {label : i for i,label in enumerate(class_labels)}
        idx_to_class = {i : label for i,label in enumerate(class_labels)}

        n_videos = len(video_names)

        dataset = []
        max_len = 0

        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_names)))
            
            label = video_labels[i]
            label_id = class_to_idx[label]

            video_path = os.path.join(video_dir,video_names[i] + ".hdf5")
            audio_path = os.path.join(self.audio_dir,video_names[i] + ".npy")
            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                print(video_path)
                continue

            
            audio = np.load(audio_path)
            max_len = max(max_len,len(audio))

            video = h5py.File(video_path)
            index = len(video['video'])
            frame_indices = list(range(index))

            sample = {
                'video': video_names[i],
                'label': label_id,
                'frame_indices': frame_indices,
            }

            dataset.append(sample)
        self.max_len = max_len

        return dataset,idx_to_class,n_videos



    def __len__(self):
        return len(self.dataset)
    

    def __loading(self, path, frame_indices):

        try:
            clip = self.loader(path, frame_indices)
        except Exception as e:
            print("path {} has error".format(path))
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip

    def __load_audio(self,audio_path,frame_indices):

        audio = torch.from_numpy(np.load(audio_path))
        audio_features = audio
        mask = torch.zeros(1)


        return audio_features,mask



    def __getitem__(self, index):

        video_name = self.dataset[index]['video']

        video_path = os.path.join(self.video_dir,video_name + ".hdf5")
        target = self.dataset[index]['label']
        frame_indices = self.dataset[index]['frame_indices']

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        
        clip = self.__loading(video_path, frame_indices)

        audio_path = os.path.join(self.audio_dir,video_name + ".npy")
        
        audio,mask = self.__load_audio(audio_path,frame_indices)


        sample = {
            'video_name':video_name,
            'audio':audio,
            'mask':mask,
            'clip':clip,
            'target':target,
            'selected':index
        }
        return sample



