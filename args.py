import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed',
                        default=1,
                        type=int,
                        help='Manually set random seed')
    # the dataset parameters
    parser.add_argument('--video_dir',type=Path)
    parser.add_argument('--annotation_path',type=Path)
    parser.add_argument('--subset',default = 'training')
    parser.add_argument('--audio_dir',type=Path)
    parser.add_argument('--fps',type=int,default = 4)
    # the model parameters
    parser.add_argument('--result_path',type=Path)
    parser.add_argument('--resume_path',type=Path)

    parser.add_argument("--train_subset",default = "training")
    parser.add_argument("--val_subset",default = "validation")
    parser.add_argument("--inference_subset",default = "validation")
    # the transform parameters

    parser.add_argument('--sample_size',type = int,default = 112)
    parser.add_argument('--sample_t_stride',type = int,default = 1)
    parser.add_argument('--train_crop',
                        default='random',
                        type=str,
                        help=('Spatial cropping method in training. '
                              'random is uniform. '
                              'corner is selection from 4 corners and 1 center. '
                              '(random | corner | center)'))
    parser.add_argument('--value_scale',
                        default=1,
                        type=int,
                        help=
                        'If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
    parser.add_argument("--scale_h", type=int, default=128,
                        help="Scale image height to")
    parser.add_argument("--scale_w", type=int, default=171,
                        help="Scale image width to")
    parser.add_argument('--train_crop_min_scale',
                        default=0.25,
                        type=float,
                        help='Min scale for random cropping in training')
    parser.add_argument('--train_crop_min_ratio',
                        default=0.75,
                        type=float,
                        help='Min aspect ratio for random cropping in training')
    parser.add_argument('--no_hflip',
                        action='store_true',
                        help='If true holizontal flipping is not performed.')
    parser.add_argument('--colorjitter',
                        action='store_true',
                        help='If true colorjitter is performed.')
    parser.add_argument('--train_t_crop',
                        default='random',
                        type=str,
                        help=('Temporal cropping method in training. '
                              'random is uniform. '
                              '(random | center)'))
    # optimizer parameters
    parser.add_argument('--optim_type',default = 'adam')
    parser.add_argument('--learning_rate',type = float,default = 3e-4)
    parser.add_argument('--lr_scheduler',
                        default='multistep',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau)')
    parser.add_argument('--multistep_milestones',
                        default=[150], 
                        type=int,
                        nargs='+',
                        help='Milestones of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument('--weight_decay',type = float, default = 5e-4)
    parser.add_argument('--n_epochs',type = int,default = 200)
    
    # model paramters
    parser.add_argument("--model_depth",type = int, default = 18)
    parser.add_argument('--n_classes',type = int,default = 51)
    parser.add_argument("--n_head",type = int, default = 4)
    parser.add_argument("--with_norm",action = "store_true")
    parser.add_argument("--no_cuda",action = "store_true")
    # loader parameters
    
    parser.add_argument('--n_threads',
                        default=8,
                        type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='Batch Size')
    parser.add_argument('--checkpoint',
                        default=50,
                        type=int)

      # val loader params
    parser.add_argument("--val_freq",type = int,default = 5)
    parser.add_argument('--n_val_samples',
                        default=3,
                        type=int,
                        help='Number of validation samples for each activity')
    parser.add_argument('--sample_duration',
                        default=16,
                        type=int,
                        help='Temporal duration of inputs')
      # inference loader params

    parser.add_argument('--inference_batch_size',
                        default=1,
                        type=int,
                        help='Batch Size for inference. 0 means this is the same as batch_size.')
    parser.add_argument('--inference_stride',
                        default=16,
                        type=int,
                        help='Stride of sliding window in inference.')
    parser.add_argument('--output_topk',
                        default=5,
                        type=int,
                        help='Top-k scores are saved in json file.')
    parser.add_argument('--inference_no_average',
                        action='store_true',
                        help='If true, outputs for segments in a video are not averaged.')
    parser.add_argument('--inference_crop',
                        default='center',
                        type=str,
                        help=('Cropping method in inference. (center | nocrop)'
                              'When nocrop, fully convolutional inference is performed,'
                              'and mini-batch consists of clips of one video.'))
    
    # for weight
    parser.add_argument("--cls_n_weight",type = float,default = 1.)
    parser.add_argument("--adj_loss",type = float,default = 1.)
    parser.add_argument("--cls_v_weight",type = float,default = 1.)
    # train params
    parser.add_argument("--use_audio",action = "store_true")
    parser.add_argument("--margin",type = float,default = 0.2)
    parser.add_argument("--threshold",type = float,default = 0.07)
    # model params
    parser.add_argument("--vid_dim",type = int,default = 512)
    parser.add_argument("--aud_dim",type = int,default = 512 )
    parser.add_argument("--with_adj",action = "store_true")
    parser.add_argument("--use_batch_cos",action = "store_true")
    parser.add_argument("--use_norm",action = "store_true")
    parser.add_argument("--use_triplet",action = "store_true")
    parser.add_argument("--temp_param",type = float, default = 0.4)
    
    parser.add_argument("--train",action = "store_true")
    parser.add_argument("--inference",action = "store_true")

    parser.add_argument('--tensorboard',
                        action='store_true',
                        help='If true, output tensorboard log file.')
    args = parser.parse_args()

    return args