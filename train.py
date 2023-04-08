import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import AverageMeter, calculate_accuracy, write_to_batch_logger, write_to_epoch_logger
from validation import val_epoch
from utils import Logger, worker_init_fn, get_lr,save_checkpoint
from loss.csc_loss import CSC_Loss

def py_softmax(x,dim = None):
    return torch.exp(x - torch.logsumexp(x,dim = dim,keepdim = True))

def train(train_loader,val_loader,video_model,distillation_model,optimizers,
            train_logger,val_logger,train_batch_logger,tb_writer,schedulers,opt):

    N = len(train_loader.dataset)

    cls_criterion = nn.CrossEntropyLoss().to(opt.device)

    cos_adj_criterion = CSC_Loss(opt.with_adj,opt.threshold)

    criterions = {
        'cls_criterion' : cls_criterion,
        "cos_adj_criterion": cos_adj_criterion
    }


    pre_val_acc = 0.0
    
    for i in range(opt.begin_epoch, opt.n_epochs + 1): 

        train_av_epoch(epoch=i, data_loader=train_loader, video_model=video_model,
                        distillation_model=distillation_model,
                        criterions=criterions,
                        optimizers=optimizers,
                        device=opt.device,
                        epoch_logger=train_logger, 
                        batch_logger=train_batch_logger,
                        tb_writer=tb_writer, opt=opt)

        audio_scheduler = schedulers['audio_scheduler']
        video_scheduler = schedulers['video_scheduler']
        video_optimizer = optimizers['video_optimizer']

        if i % opt.checkpoint == 0 :
            save_file_path = opt.result_path / 'save_{}.pth'.format(i)
            save_checkpoint(save_file_path, i,  video_model,distillation_model, video_optimizer, schedulers)
        if i % opt.val_freq == 0:
            prev_val_loss, val_acc = val_epoch(i, val_loader, video_model, criterions, opt.device, val_logger, tb_writer)
            if pre_val_acc < val_acc:
                pre_val_acc = val_acc
                save_file_path = opt.result_path / 'save_model.pth'
                save_checkpoint(save_file_path, i, video_model,distillation_model, video_optimizer, schedulers)
        
        if opt.train and opt.lr_scheduler == 'multistep':
            audio_scheduler.step()
            video_scheduler.step()
        elif opt.train and opt.lr_scheduler == 'plateau':
            if prev_val_loss is not None:
                audio_scheduler.step(prev_val_loss)
                video_scheduler.step(prev_val_loss)

def get_similarity(video_feature,audio_feature):
    norm_video = F.normalize(video_feature, dim = 1)
    norm_audio = F.normalize(audio_feature, dim = 1)

    similarity = torch.mm(norm_video,norm_audio.T)


    return similarity

def get_triplet_loss(anchor_feature,negative_feature,positive_feature,margin):
    norm_anchor_feature = F.normalize(anchor_feature, dim = 1)
    norm_negative_feature = F.normalize(negative_feature,dim = 1)
    norm_positive_feature = F.normalize(positive_feature, dim = 1)

    cos_an = torch.mm(norm_anchor_feature,norm_negative_feature.T).diag()
    cos_ap = torch.mm(norm_anchor_feature,norm_positive_feature.T).diag()

    zeros = torch.zeros(anchor_feature.shape[0]).cuda()
    loss = torch.max(cos_an - cos_ap + margin,zeros)
    loss = torch.mean(loss,dim = 0)
    return loss

def train_av_epoch(epoch,
                data_loader,
                video_model,
                distillation_model,
                criterions,
                optimizers,
                device,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                opt=None):
    print('train at epoch {}'.format(epoch))

    video_model.train()
    distillation_model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # classification loss
    losses_cls_v = AverageMeter()
    losses_cls_n = AverageMeter()
    triplet_losses = AverageMeter()
    norm_losses= AverageMeter()
    accuracies_n = AverageMeter()
    accuracies = AverageMeter()

    # contrastive lossw
    losses_adj = AverageMeter()

    end_time = time.time()

    video_optimizer = optimizers['video_optimizer']
    audio_optimizer = optimizers['audio_optimizer']

    # get the criterions
    cls_criterion = criterions['cls_criterion']
    cos_adj_criterion = criterions["cos_adj_criterion"]

    current_lr = get_lr(video_optimizer)

    for i, batch in enumerate(data_loader):
        video = batch['clip']
        targets = batch['target']
        audio = batch['audio'].cuda()


        data_time.update(time.time() - end_time)
        targets = targets.to(device, non_blocking=True)
        audio_feature = audio.to(device, non_blocking=True)

        # get video feature
        video_feature_map,video_feature,p_v = video_model(video)

        p_sv, selected_video_feature, selected_audio_feature,fc_weights,_,_ = distillation_model(audio_feature,video_feature_map)
        if opt.use_norm:
            norm_loss = 0
            for fc_weight in fc_weights:
                norm_loss = 0.001 * fc_weight.abs().sum(1).mean()
        else:
            norm_loss = torch.zeros(1).cuda()
        after_cos = get_similarity(video_feature,selected_audio_feature)
        detach_cos = after_cos.detach()

        loss_cls_n = opt.cls_n_weight * cls_criterion(p_sv,targets)
            
        loss_cls_v = opt.cls_v_weight * cls_criterion(p_v, targets) 

        if opt.use_triplet:
            triplet_loss =  get_triplet_loss(video_feature,audio,selected_audio_feature,opt.margin)
        else:
            triplet_loss = torch.zeros(1).cuda()
        
        adj_loss = opt.adj_loss * cos_adj_criterion(video_feature,selected_video_feature,targets,detach_cos)

        acc = calculate_accuracy(p_v, targets)
        Acc_n = calculate_accuracy(p_sv,targets)

        a_losses = sum([loss_cls_n,norm_loss,triplet_loss])
        v_losses = sum([loss_cls_v,adj_loss])

        losses_cls_v.update(loss_cls_v.item(), video.size(0))
        losses_cls_n.update(loss_cls_n.item(), video.size(0))

        losses_adj.update(adj_loss.item(),video.size(0))
        norm_losses.update(norm_loss.item(),video.size(0))
        triplet_losses.update(triplet_loss.item(),video.size(0))
        accuracies.update(acc, video.size(0))
        accuracies_n.update(Acc_n, video.size(0))

        audio_optimizer.zero_grad()
        a_losses.backward(retain_graph = True)
        video_optimizer.zero_grad()
        v_losses.backward()
        video_optimizer.step()
        audio_optimizer.step()


        #####################################################################################
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        write_to_batch_logger(batch_logger, epoch, i, data_loader, losses_cls_v.val, accuracies.val, current_lr)

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss_cls_v {loss_cls_v.val:.3f} ({loss_cls_v.avg:.3f})\t'
              'loss_cls_n {loss_cls_n.val:.3f} ({loss_cls_n.avg:.3f})\t'
              'loss_adj {loss_adj.val:.3f} ({loss_adj.avg:.3f})\t'
              'loss_norm {loss_norm.val:.3f} ({loss_norm.avg:.3f})\t'
              'Triplet_Loss {triplet.val:.3f} ({triplet.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'Acc_n {Acc_n.val:.3f} ({Acc_n.avg:.3f})\t'.

              format(epoch, i + 1, len(data_loader),
                     batch_time=batch_time,
                     data_time=data_time,
                     loss_cls_v=losses_cls_v,
                     loss_cls_n=losses_cls_n,
                     loss_adj=losses_adj,
                     loss_norm = norm_losses,
                     triplet = triplet_losses,
                     acc=accuracies,
                     Acc_n=accuracies_n), flush=True)

    write_to_epoch_logger(epoch_logger, epoch, losses_cls_v.avg, accuracies.avg, current_lr)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss_cls_v', losses_cls_v.avg, epoch)
        tb_writer.add_scalar('train/loss_cls_n', losses_cls_n.avg, epoch)
        tb_writer.add_scalar('train/loss_adj', losses_adj.avg, epoch)
        tb_writer.add_scalar('train/loss_norm', norm_losses.avg, epoch)
        tb_writer.add_scalar('train/loss_triplet', triplet_losses.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)
        tb_writer.add_scalar('train/Acc_n', accuracies_n.avg, epoch)

def train_v_epoch(epoch,
                data_loader,
                video_model,
                distillation_model,
                criterions,
                optimizers,
                device,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                opt=None):
    print('train at epoch {}'.format(epoch))

    video_model.train()
    distillation_model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # classification loss
    losses_cls_v = AverageMeter()
    accuracies = AverageMeter()

    # contrastive lossw

    end_time = time.time()

    video_optimizer = optimizers['video_optimizer']
    # get the criterions

    cls_criterion = criterions['cls_criterion']

    current_lr = get_lr(video_optimizer)

    for i, batch in enumerate(data_loader):
        video = batch['clip']
        targets = batch['target']

        data_time.update(time.time() - end_time)
        targets = targets.to(device, non_blocking=True)

        # get video feature
        video_feature_map,video_feature,p_v = video_model(video)
        
        loss_cls_v = opt.cls_v_weight * cls_criterion(p_v, targets) 


        acc = calculate_accuracy(p_v, targets)
        v_losses = sum([loss_cls_v])

        losses_cls_v.update(loss_cls_v.item(), video.size(0))

        accuracies.update(acc, video.size(0))

        video_optimizer.zero_grad()
        v_losses.backward()
        video_optimizer.step()

        #####################################################################################
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        write_to_batch_logger(batch_logger, epoch, i, data_loader, losses_cls_v.val, accuracies.val, current_lr)

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss_cls_v {loss_cls_v.val:.3f} ({loss_cls_v.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'.

              format(epoch, i + 1, len(data_loader),
                     batch_time=batch_time,
                     data_time=data_time,
                     loss_cls_v=losses_cls_v,
                     acc=accuracies), flush=True)

    write_to_epoch_logger(epoch_logger, epoch, losses_cls_v.avg, accuracies.avg, current_lr)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss_cls_v', losses_cls_v.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)