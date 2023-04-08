from transform.spatial_transforms import (Compose, Normalize, Resize, CenterCrop, RandomCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from transform.temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from transform.temporal_transforms import Compose as TemporalCompose

from datasets.video_dataset_multiclips import VideoDatasetMultiClips
from datasets.av_dataset_multiclips import AudioVideoDatasetMultiClips


from utils import Logger,worker_init_fn,get_lr
import torch
from datasets.loader import  VideoLoaderHDF5
from torch.utils.data.dataloader import default_collate

def collate_fn(batch):

    # video_name,batch_clips, batch_targets = zip(*batch)
    new_batchs = {
        'video_name':[],
        'clip':[],
        'target':[]
    }

    for sample in batch:
        for idx,clip in enumerate(sample['clip']):
            new_batchs['video_name'].append(sample['video_name'])
            new_batchs['clip'].append(sample['clip'][idx])
            new_batchs['target'].append(sample['target'][idx])

    new_batchs['clip'] = default_collate(new_batchs['clip'])
    new_batchs['target'] = default_collate(new_batchs['target'])
    # new_batchs['video_name'] = default_collate(new_batchs['video_name'])
    return new_batchs

def get_normalize_method():
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    return Normalize(mean,std)

def get_val_utils(opt):
    normalize = get_normalize_method()

    if opt.train_crop == 'other':
        spatial_transform = [
            Resize((opt.scale_h, opt.scale_w)),
            RandomCrop(opt.sample_size),
            ToTensor()
        ]
    else:
        spatial_transform = [
            Resize(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor()
        ]

    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))

    temporal_transform.append(
        TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))

    temporal_transform = TemporalCompose(temporal_transform)

    loader = VideoLoaderHDF5()
    val_data =  VideoDatasetMultiClips(opt.video_dir,
                                             opt.annotation_path,
                                             opt.val_subset,
                                             opt.fps,
                                             spatial_transform=spatial_transform,
                                             temporal_transform=temporal_transform,
                                             video_loader=loader)

    val_sampler = None

    g = torch.Generator()
    g.manual_seed(opt.manual_seed)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size // opt.n_val_samples),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn,
                                             generator=g)


    val_logger = Logger(opt.result_path / 'val.log',
                        ['epoch', 'loss', 'acc', 'acc_num'])
    return val_loader, val_logger


def get_inference_utils(opt,inference_subset):
    assert opt.inference_crop in ['center', 'nocrop']

    normalize = get_normalize_method()

    if opt.train_crop == 'other':
        spatial_transform = [
            Resize((opt.scale_h, opt.scale_w)),
            RandomCrop(opt.sample_size),
            ToTensor()
        ]
    else:
        spatial_transform = [Resize(opt.sample_size)]

    if opt.inference_crop == 'center':
        spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.append(ToTensor())
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)

    loader = VideoLoaderHDF5()

    inference_data =  VideoDatasetMultiClips(opt.video_dir,
                                             opt.annotation_path,
                                             inference_subset,
                                             opt.fps,
                                             spatial_transform=spatial_transform,
                                             temporal_transform=temporal_transform,
                                             video_loader=loader)

    g = torch.Generator()
    g.manual_seed(opt.manual_seed)
    inference_loader = torch.utils.data.DataLoader(inference_data,
                                                   batch_size=opt.inference_batch_size,
                                                   shuffle=False,
                                                   num_workers=opt.n_threads,
                                                   pin_memory=True,
                                                   worker_init_fn=worker_init_fn,
                                                   collate_fn=collate_fn,
                                                   generator=g)

    return inference_loader,inference_data.idx_to_class