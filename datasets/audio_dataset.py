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



def get_dataset(annotation_data,subset,audio_dir):
    video_names = []
    video_labels = []

    for key in annotation_data.keys():
        if annotation_data[key]['subset'] == subset:
            audio_path = os.path.join(audio_dir,key + ".npy")
            if os.path.exists(audio_path):

                video_names.append(key)
                video_labels.append(annotation_data[key]['label'])
    return video_names,video_labels


class AudioDataset(data.Dataset):
    def __init__(self,
                audio_dir,
                annotation_path,
                subset):
        

        self.audio_dir = audio_dir
        self.dataset,self.idx_to_class,self.n_videos = self.__make_dataset(annotation_path,subset)
        
    
    def __make_dataset(self,annotation_path,subset):
        with open(annotation_path) as f:
            annotation_data = json.load(f)
        class_labels = annotation_data['labels']
        annotation_data = annotation_data['database']
        # get the video name and labels
        video_names , video_labels = get_dataset(annotation_data,subset,self.audio_dir)
        

        # get the label and idx map
        class_to_idx = {label : i for i,label in enumerate(class_labels)}
        idx_to_class = {i : label for i,label in enumerate(class_labels)}

        n_videos = len(video_names)

        dataset = []
        max_len = 0
        
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_names)))
            
            audio_path = os.path.join(self.audio_dir,video_names[i] + ".npy")
            audio = np.load(audio_path)

            max_len = max(max_len,len(audio))

            label = video_labels[i]
            label_id = class_to_idx[label]

            sample = {
                'video': video_names[i],
                'label': label_id,
            }

            dataset.append(sample)
        
        self.max_len = max_len

        print(self.max_len)
        return dataset,idx_to_class,n_videos



    def __len__(self):
        return len(self.dataset)
    

    def __load_audio(self,audio_path):
        audio = torch.from_numpy(np.load(audio_path))
        
        # audio_features = audio
        # audio_features = audio.reshape(-1)
        # mask = torch.zeros(1)

        audio_features = torch.zeros((self.max_len,512))
        audio_features[:len(audio)] = audio
        mask = torch.ones(self.max_len)
        mask[:len(audio)] = 0.
        
        return audio_features,mask

    def __getitem__(self, index):

        video_name = self.dataset[index]['video']

        target = self.dataset[index]['label']

        audio_path = os.path.join(self.audio_dir,video_name + ".npy")
        
        audio,mask = self.__load_audio(audio_path)


        sample = {
            'video_name':video_name,
            'audio':audio,
            'mask':mask,
            'target':target,
            'selected':index
        }

        return sample



