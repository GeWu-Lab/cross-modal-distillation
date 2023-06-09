import csv
import random
from functools import partialmethod

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def save_checkpoint(save_file_path, epoch, video_model,audio_model, optimizer, scheduler):
    
    if hasattr(video_model, 'module'):
        video_model_state_dict = video_model.module.state_dict()
    else:
        video_model_state_dict = video_model.state_dict()
    
    if hasattr(audio_model, 'module'):
        audio_model_state_dict = audio_model.module.state_dict()
    else:
        audio_model_state_dict = audio_model.state_dict()
    save_states = {
        'epoch': epoch,
        'state_dict': video_model_state_dict,
        'audio_state_dict': audio_model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler
    }
    torch.save(save_states, save_file_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def calculate_precision_and_recall(outputs, targets, pos_label=1):
    with torch.no_grad():
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        precision, recall, _, _ = precision_recall_fscore_support(
            targets.view(-1, 1).cpu().numpy(),
            pred.cpu().numpy())

        return precision[pos_label], recall[pos_label]


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)


def partialclass(cls, *args, **kwargs):

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass


def write_to_batch_logger(batch_logger, epoch, i, data_loader, losses, accuracies, current_lr):
    if batch_logger is not None:
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses,
            'acc': accuracies,
            'lr': current_lr
        })


def write_to_epoch_logger(epoch_logger, epoch, losses, accuracies, current_lr):
    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses,
            'acc': accuracies,
            'lr': current_lr
        })
