from transform.spatial_transforms import (Compose, Normalize, Resize, CenterCrop, RandomCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from transform.temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from transform.temporal_transforms import Compose as TemporalCompose
import torch
from datasets.loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5
from datasets.av_dataset import AudioVideoDataset
from datasets.videodataset import VideoDataset
from datasets.video_dataset_multiclips import VideoDatasetMultiClips
from datasets.av_dataset_multiclips import AudioVideoDatasetMultiClips

from utils import Logger,worker_init_fn,get_lr
from torch.optim import SGD,Adam,lr_scheduler

def get_normalize_method():
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    return Normalize(mean,std)


def get_spatial_transform(opt):
    assert opt.train_crop in ['random', 'corner', 'center', 'other']
    spatial_transform = []
    if opt.train_crop == 'random':
        print('random crop')
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    elif opt.train_crop == 'other':
        print('other')
        spatial_transform.append(Resize((opt.scale_h, opt.scale_w)))
        spatial_transform.append(RandomCrop(opt.sample_size))

    normalize = get_normalize_method()
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())
    spatial_transform.append(ToTensor())

    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    return spatial_transform

def get_temporal_transform(opt):
    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    return temporal_transform

def get_train_dataset(opt,spatial_transform,temporal_transform):
    loader = VideoLoaderHDF5()
    # whether additional audio features are used
    if opt.use_audio:
        train_data = AudioVideoDataset(
            opt.video_dir,opt.audio_dir,opt.annotation_path,
            opt.train_subset,opt.fps,spatial_transform,temporal_transform,video_loader = loader
        )
    else:
        train_data = VideoDataset(
            opt.video_dir,opt.annotation_path,opt.train_subset,opt.fps,
            spatial_transform,temporal_transform,video_loader = loader
        )

    train_sampler = None
    
    g = torch.Generator()
    g.manual_seed(opt.manual_seed)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn,
                                               generator=g)
    return train_loader,train_sampler

def get_train_module_utils(opt,parameters):
    spatial_transform = get_spatial_transform(opt)
    temporal_transform = get_temporal_transform(opt)

    train_loader , train_sampler = get_train_dataset(opt,spatial_transform,temporal_transform)    

    train_logger = Logger(opt.result_path / 'train.log',
                            ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(
        opt.result_path / 'train_batch.log',
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

    video_params = parameters['video_params']
    audio_params = parameters['audio_params']

    if opt.optim_type == 'adam':
        video_optimizer = Adam(
            video_params,
            lr = opt.learning_rate,
            weight_decay = opt.weight_decay
        )

        audio_optimizer = Adam(
            audio_params,
            lr = opt.learning_rate * 0.2,
            weight_decay = opt.weight_decay
        )
    else:
        video_optimizer = SGD(
            video_params,
            lr = opt.learning_rate,
            weight_decay = opt.weight_decay
        )

        audio_optimizer = SGD(
            audio_params,
            lr = opt.learning_rate * 0.2,
            weight_decay = opt.weight_decay
        )

    optimizers = {
        "video_optimizer":video_optimizer,
        "audio_optimizer":audio_optimizer
    }

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        video_scheduler = lr_scheduler.ReduceLROnPlateau(video_optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)
        audio_scheduler = lr_scheduler.ReduceLROnPlateau()
    else:
        audio_scheduler = lr_scheduler.MultiStepLR(audio_optimizer, opt.multistep_milestones)
        video_scheduler = lr_scheduler.MultiStepLR(video_optimizer, opt.multistep_milestones)
    
    schedulers = {
        'video_scheduler' : video_scheduler,
        'audio_scheduler' : audio_scheduler
    }

    return (train_loader, train_sampler, train_logger, train_batch_logger, optimizers, schedulers)