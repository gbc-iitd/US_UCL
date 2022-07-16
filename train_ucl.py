import argparse
import builtins
import datetime
import math
import os
import random
import shutil
import time
import warnings
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import ucl.loader
import ucl.builder_gb
from ucl.datasets import ImageFolderInstance
from ucl.datasets import R2V2Dataset
from ucl.datasets_gb import GbVideoDataset
from utils.util import is_main_process
#import neptune.new as neptune

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='gbc',
                    help='dataset name')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.06, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', default=10, type=int,
                    help='save frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--cycle-k', default=81920, type=int,
                    help='cycle queue size; number of negative keys (default: 65536)')

parser.add_argument('--negative-N', default=32, type=int,
                    help='Negatives buffer size (default: 32)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for cycle_contrast v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# options for cycle contrastive
parser.add_argument('--soft-nn', action='store_true')
parser.add_argument('--soft-nn-loss-weight', default=0.1, type=float)
parser.add_argument('--moco-loss-weight', default=1., type=float)
parser.add_argument('--soft-nn-support', default=16384, type=int)
parser.add_argument('--sep-head', action='store_true')
parser.add_argument('--cycle-neg-only', dest='cycle_neg_only', action='store_true')
parser.add_argument('--no-cycle-neg-only', dest='cycle_neg_only', action='store_false')
parser.add_argument('--soft-nn-t', default=-1., type=float)
parser.add_argument('--cycle-back-cls', action='store_true')
parser.add_argument('--cycle-back-cls-video-as-pos', action='store_true')

parser.add_argument('--resizecropsize', default=0.2, type=float)

parser.add_argument('--cycle-back-candidates', action='store_true')

parser.add_argument('--num-classes', default=100, type=int)
parser.add_argument('--moco-random-video-frame-as-pos', action='store_true')
parser.add_argument('--detach-target', action='store_true')
parser.add_argument('--multi-crops', default=2, type=int)
parser.add_argument('--num-of-sampled-frames', default=1, type=int)
parser.set_defaults(cycle_neg_only=True)
parser.add_argument('--save-dir', default='../../../scratch/cyclecontrast/output', type=str)
parser.add_argument('--soft-nn-topk-support', action='store_true')
parser.add_argument('--exp_name',default='integrating_changes',type=str)
parser.add_argument('--adam', action='store_true')
parser.add_argument('--pretrained-models',action='store_true')
parser.add_argument('--constant-lr',action='store_true')
parser.add_argument('--negatives',action='store_true')
parser.add_argument('--intranegs-only-two',action='store_true')
parser.add_argument('--convex-combo-loss',action='store_true')
parser.add_argument('--cross-neg-topk-mining',action='store_true')
parser.add_argument('--cross-neg-topk-support-size',default=4, type=int)
parser.add_argument('--anchor-reverse-cross',action='store_true')
parser.add_argument('--single-loss-intra-inter',action='store_true')
parser.add_argument('--qcap-include',action='store_true')
parser.add_argument('--cosine-curriculum',action='store_true')
parser.add_argument('--cosine-clipping',action='store_true')
parser.add_argument('--mean-neighbors',action='store_true')
parser.add_argument('--single-loss-ncap-support-size',default=4, type=int)
parser.add_argument('--num-negatives',default=2, type=int)
parser.add_argument('--local_rank', type=int)

def main():
    args = parser.parse_args()

    if os.path.isdir(os.path.join(args.save_dir,args.exp_name)):
        shutil.rmtree(os.path.join(args.save_dir,args.exp_name),ignore_errors=True)
    os.mkdir(os.path.join(args.save_dir,args.exp_name))
    #####################
    num_var = 32
    args.batch_size = num_var
    #####################
    args.cycle_k = num_var
    args.moco_k = num_var
    args.soft_nn_support = 4
    args.negative_N = args.batch_size * args.num_negatives
    args.epochs = 60
    #####################

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = 4
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu !=0:
        args.gpu +=3

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print('rank', dist.get_rank())

    # init_distributed_mode(args)
    #print(args)

    # create model
    
    print("=> creating model '{}'".format(args.arch))
    model = ucl.builder_gb.CycleContrast(
        args.arch, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, soft_nn=args.soft_nn,
        soft_nn_support=args.soft_nn_support,
        sep_head=args.sep_head,
        cycle_neg_only=args.cycle_neg_only,
        soft_nn_T=args.soft_nn_t,
        cycle_back_cls=args.cycle_back_cls,
        cycle_back_cls_video_as_pos=args.cycle_back_cls_video_as_pos,
        moco_random_video_frame_as_pos=args.moco_random_video_frame_as_pos,
        cycle_K=args.cycle_k,
        pretrained_on_imagenet=args.pretrained_models,
        soft_nn_topk_support= args.soft_nn_topk_support,
        negative_use = args.negatives,
        neg_queue_size=args.negative_N,
        intranegs_only_two = args.intranegs_only_two,
        cross_neg_topk_mining = args.cross_neg_topk_mining,
        cross_neg_topk_support_size = args.cross_neg_topk_support_size,
        anchor_reverse_cross = args.anchor_reverse_cross,
        single_loss_intra_inter = args.single_loss_intra_inter,
        single_loss_ncap_support_size = args.single_loss_ncap_support_size,
        qcap_include = args.qcap_include,
        tsne_name = args.exp_name,
        mean_neighbors = args.mean_neighbors
    )
    print(model)

    if args.gpu == 0:
        writer = SummaryWriter(args.save_dir)
    else:
        writer = None

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    if args.cycle_back_cls:
        criterion = [nn.CrossEntropyLoss().cuda(args.gpu),
                     nn.CrossEntropyLoss().cuda(args.gpu)]
    else:
        criterion = [nn.CrossEntropyLoss().cuda(args.gpu),
                     nn.MSELoss().cuda(args.gpu)]

    
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(),args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #print(optimizer)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            msg = model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}), {}"
                  .format(args.resume, checkpoint['epoch'], msg))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resizecropsize = args.resizecropsize

    augmentation = [
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize
    ]
    

    
    print("Start training")

    """
    Curriculum = [Epoch_Number_till_which_this_will_happen, Negs_low, Negs_High, Only_use_qcap]
    """
    curriculum = [[10,0.2,0.4,True],[30,0.2,0.4,False],[50,0.1,0.15,False],[60,0.03,0.07,False]]
    """
    curriculum = [[20,0.2,0.4,False],[40,0.1,0.15,False],[50,0.03,0.07,False]]
    """
    curriculum_ptr = 0

    if args.cosine_curriculum:
        low_n = 0.2
        high_n = 0.4

        if args.cosine_clipping:
            min_low_n = 0.03
            min_high_n = 0.07

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        """ Getting the correct Dataloader as per epoch requirements"""

        
        if epoch==curriculum[curriculum_ptr][0]:
            curriculum_ptr+=1
        
        if args.dataset == 'r2v2':
            train_dataset = R2V2Dataset(args.data,
                                        transforms.Compose(augmentation),
                                        return_all_video_frames=args.cycle_back_candidates
                                                                or args.moco_random_video_frame_as_pos,
                                        num_of_sampled_frames=args.num_of_sampled_frames,
                                        )
            paths_ds_tsne = train_dataset.path_info
            paths_ds_dict ={}
            for i in range(len(paths_ds_tsne)):
                paths_ds_dict[i]=paths_ds_tsne[i][0]
            with open('vid_index_mapping_tsne.json', 'w') as fp:
                json.dump(paths_ds_dict, fp)
    
        elif args.dataset == 'gbc': 
            if not args.cosine_curriculum:
                train_dataset = GbVideoDataset(args.data,
                                    transforms.Compose(augmentation),neg_dists=[curriculum[curriculum_ptr][1], curriculum[curriculum_ptr][2]], num_neg_samples=args.num_negatives)

            else:
                if epoch >= curriculum[1][0]:
                    epochs_done = curriculum[1][0]
                    low_n , high_n = adjust_negs(low_n,high_n,epoch,args,epochs_done)
                if args.cosine_clipping:
                    low_n = max(min_low_n, low_n)
                    high_n = max(min_high_n, high_n)

                print(low_n,high_n)
                train_dataset = GbVideoDataset(args.data,
                                    transforms.Compose(augmentation),neg_dists=[low_n, high_n], num_neg_samples=args.num_negatives)

        else:
            crops_transform = ucl.loader.TwoCropsTransform(transforms.Compose(augmentation))
            train_dataset = ImageFolderInstance(traindir,
                                            crops_transform)
            print('class name to idx', train_dataset.dataset.class_to_idx)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)


        ep_start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if  not args.adam:
            if not args.constant_lr:
                adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        if args.single_loss_intra_inter:
            total_loss_n,loss_moco_n, top1_n = train(train_loader, model, criterion, optimizer, epoch, args, only_qcap= curriculum[curriculum_ptr][3], writer=writer)
        else:
            if not args.intranegs_only_two:
                total_loss_n,loss_moco_n, loss_softnn_n, top1_n,top5_n = train(train_loader, model, criterion, optimizer, epoch, args,writer)
            else:
                total_loss_n,loss_moco_n, loss_softnn_n, top1_n = train(train_loader, model, criterion, optimizer, epoch, args,writer)

        if args.gpu==0:
            learning_rate_neptune = 0
            for param_group in optimizer.param_groups:
                learning_rate_neptune=param_group['lr'] 
                ##print(learning_rate_neptune)

            #print(optimizer.state_dict())
            
        
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if epoch % args.save_freq == args.save_freq - 1 and is_main_process():
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename='checkpoint_{:04d}'.format(epoch)+"_"+args.exp_name+".pth.tar", path=os.path.join(args.save_dir,args.exp_name))
        epoch_time_str = str(datetime.timedelta(seconds=int(time.time() - ep_start_time)))
        print('Train Epoch {} time {}'.format(epoch, epoch_time_str))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train(train_loader, model, criterion, optimizer, epoch, args,only_qcap=False, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    softnn_losses = AverageMeter('Cycle Loss', ':.4e')
    total_loss = AverageMeter('Total Loss', ':.4e')

    if args.single_loss_intra_inter:
        log_stats = [batch_time, data_time, losses, top1]
    else:
        if not args.intranegs_only_two:
            log_stats = [batch_time, data_time, losses, softnn_losses, top1, top5]
                
        else:
            log_stats = [batch_time, data_time, losses, softnn_losses, top1]

    progress = ProgressMeter(
        len(train_loader),
        log_stats,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    
    model.train()

    end = time.time()

    for i, data_pack in enumerate(train_loader):
    
        if len(data_pack) == 3:
            images, cls_labels, indices = data_pack
            video_frames = None
        elif len(data_pack) == 4:
            images, cls_labels, indices, video_frames = data_pack
        elif len(data_pack) == 5:
            images, cls_labels, indices, video_frames, is_same_frame = data_pack
        elif len(data_pack) ==6:
            images, cls_labels, indices, video_frames,metadata_q,metadata_can = data_pack
        else:
            assert False, 'unsupported data pack of len {}'.format(len(data_pack))
        data_time.update(time.time() - end)

        if args.negatives:
            cls_candidates = video_frames[:,0].unsqueeze(1)

            im_negs = None
            for itr_var in range(args.num_negatives):
                if itr_var == 0:
                        im_negs = video_frames[:,itr_var+1].unsqueeze(1)
                else:
                        im_negs = torch.cat((im_negs,video_frames[:,itr_var+1].unsqueeze(1)),axis=1)
            
            
            outputs = model(im_q=images[0], im_k=images[1], cls_labels=cls_labels, indices=indices,
                        cls_candidates=cls_candidates,im_negs=im_negs,tsne=(epoch==args.epochs-1),only_qcap=only_qcap)
        else:
            outputs = model(im_q=images[0], im_k=images[1], cls_labels=cls_labels, indices=indices,
                        cls_candidates=video_frames,tsne=(epoch==args.epochs-1))
        
        if args.soft_nn:
            if not args.single_loss_intra_inter:
                output, target, softnn_feat, q_feat, meta = outputs
            else:
                output, target, meta = outputs
        else:
            output, target, meta = outputs

        loss_moco = criterion[0](output, target)

        if args.soft_nn:
            """
            if not args.cycle_back_k and args.detach_target:
                softnn_feat = softnn_feat.detach()
            """
            if not args.single_loss_intra_inter:
                loss_softnn = criterion[1](softnn_feat, q_feat)
                if args.moco_loss_weight == 0:
                    loss_moco = loss_moco.detach()
                if args.soft_nn_loss_weight == 0:
                    loss_softnn = loss_softnn.detach()
                if not args.convex_combo_loss:
                    loss = loss_moco * args.moco_loss_weight + loss_softnn * args.soft_nn_loss_weight
                else:
                    loss = loss_moco * (1-args.soft_nn_loss_weight) + loss_softnn * args.soft_nn_loss_weight
            else:
                if args.moco_loss_weight == 0:
                    loss_moco = loss_moco.detach()
        
                loss = loss_moco 
        else:
            if args.moco_loss_weight == 0:
                loss_moco = loss_moco.detach()
            loss = loss_moco * args.moco_loss_weight

        losses.update(loss_moco.item(), images[0].size(0))
        total_loss.update(loss.item(),images[0].size(0))
        
        if args.soft_nn:
            if not args.single_loss_intra_inter:
                softnn_losses.update(loss_softnn.item(), images[0].size(0))

        if not args.intranegs_only_two:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0].cpu().item(), images[0].size(0))
            top5.update(acc5[0].cpu().item(), images[0].size(0))
        else:
            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0].cpu().item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    if args.single_loss_intra_inter:
        return total_loss,losses,top1
    else:
        if not args.intranegs_only_two:
            return total_loss,losses,softnn_losses,top1, top5
        else:
            return total_loss,losses,softnn_losses,top1


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', path=None):
    if path is not None:
        filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_negs(low_n,high_n,epoch,args,epochs_done):
    epochs = args.epochs
    low_n *= 0.5 * (1. + math.cos(math.pi * (epoch-epochs_done) / (epochs-epochs_done)))
    high_n *= 0.5 * (1. + math.cos(math.pi * (epoch-epochs_done) / (epochs-epochs_done)))
    return low_n, high_n

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_nn(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
