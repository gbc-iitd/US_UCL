#!/usr/bin/env python
# Just to set git
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import datetime
import warnings
import copy

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
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

from dataloader import GbUsgDataSet

from ucl.datasets import ImageNetVal

#import neptune.new as neptune


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('data', metavar='DIR',
#                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--img_dir', dest="img_dir", default="data/gb_imgs")
parser.add_argument('--train_list', dest="train_list", default="data/cls_split/train.txt")
parser.add_argument('--val_list', dest="val_list", default="data/cls_split/val.txt")
parser.add_argument('--step', dest="step", default=1, type=int)
parser.add_argument('--warmup', dest="warmup", default=5, type=int)
parser.add_argument('--gradual_unfreeze', action='store_true')
parser.add_argument('--cos_lr', action='store_true')

parser.add_argument('--num_classes', default=2, type=int, metavar='NC',
                    help='number of output classes for finetuning')
parser.add_argument('--fc_type', default=1, type=int, metavar='FC',
                    help='fc_layer type')
parser.add_argument('--last_layer', action="store_true",
                    help='whether only last layer is trainable')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
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

parser.add_argument('--pretrained', default='', type=str,
                    help='path to cycle_contrast pretrained checkpoint')

parser.add_argument('--dataset', default='gbc', type=str)
parser.add_argument('--save-dir', default='', type=str)

parser.add_argument('--eval-interval', default=1, type=int)

best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    #if args.gpu is not None:
    #    warnings.warn('You have chosen a specific GPU. This will completely '
    #                  'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
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
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    #if args.gpu is not None:
    #    print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    #print("=> creating model '{}'".format(args.arch))
    
    num_classes = args.num_classes
    model = models.__dict__[args.arch](pretrained=True)
    num_ftrs = model.fc.in_features
    if args.last_layer or args.gradual_unfreeze:
        for name, param in model.named_parameters():
            param.requires_grad = False

    if args.fc_type==1:
        model.fc = nn.Linear(num_ftrs, num_classes)
        # init the fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
    else:
        model.fc = nn.Sequential(
                          nn.Linear(num_ftrs, 256), 
                          nn.ReLU(inplace=True), 
                          nn.Dropout(0.4),
                          nn.Linear(256, num_classes)
                        )
        # init the fc layer
        for i in [0, 3]:
            model.fc[i].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[i].bias.data.zero_()
    if args.arch == "resnet50":
        num_layers=16
    
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            #print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            if not args.evaluate:
            # rename cycle_contrast pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                        #print(k[len("module.encoder_q."):], k)

                    if k.startswith('module.target_encoder.net') and not k.startswith('module.target_encoder.net.fc'):
                        # remove prefix
                        state_dict[k[len("module.target_encoder.net."):]] = state_dict[k]

                    # delete renamed or unused k
                    del state_dict[k]

                args.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
            
            else:
                state_dict = checkpoint['state_dict']
                args.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
            
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    #assert len(parameters) == 2*int(args.fc_type)  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.cos_lr:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

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
            best_acc = checkpoint['metrics'][0]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc = best_acc.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #cudnn.benchmark = True

    # Data loader
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_dataset = GbUsgDataSet(data_dir=args.img_dir,
                            image_list_file=args.train_list,
                            #df=df,
                            #train=True,
                            bin_classify=(args.num_classes==2),
                            transform=transforms.Compose([
                                #transforms.Resize((224,224)),
                                transforms.Resize(224),
                                transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    #train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
    #                            shuffle=True, num_workers=0)

    val_dataset = GbUsgDataSet(data_dir=args.img_dir,
                            image_list_file=args.val_list,
                            #df=df,
                            #train=True,
                            bin_classify=(args.num_classes==2),
                            transform=transforms.Compose([
                                #transforms.Resize((224,224)),
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    #val_loader = DataLoader(dataset=val_dataset, batch_size=1,
    #                            shuffle=False, num_workers=0)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        y_true, y_pred = validate(val_loader, model, criterion, args)
        cfm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        if args.num_classes == 2:
            spec = cfm[0][0]/np.sum(cfm[0])
            sens = cfm[1][1]/np.sum(cfm[1])
            print("%.4f %.4f %.4f"%(acc, spec, sens))
        else:
            spec = (cfm[0][0]+cfm[0][1]+cfm[1][0]+cfm[1][1])/(np.sum(cfm[0])+np.sum(cfm[1]))
            sens = cfm[2][2]/np.sum(cfm[2])
            acc_2 = (cfm[0][0]+cfm[0][1]+cfm[1][0]+cfm[1][1]+cfm[2][2])/np.sum(cfm)
            print("%.4f %.4f %.4f %.4f"%(acc_2, spec, sens, acc))

        print(cfm)
        return

    """
    run = neptune.init(
        project="sbasu276/cycle-contrast",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNjgzZTk5Yi0xNmFlLTQ4YTAtODBhZS0xOGRmNzdlMTFhMmEifQ==",
        mode="offline",
    )  # your credentials

    params = {
            "learning_rate": args.lr, 
            "weight decay": args.weight_decay,
            "batch size": args.batch_size,
            "fc_type": args.fc_type,
            "last layer": args.last_layer, 
            "save_dir": args.save_dir,
            "gradual unfreeze": args.gradual_unfreeze,
            "pretrained model path": args.pretrained}
    run["parameters"] = params
    """

    os.makedirs(args.save_dir, exist_ok=True)
    #start_time = time.time()
    best_f1, best_acc = 0, 0
    is_best = True
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        epoch_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
        
        if args.gradual_unfreeze:
            if epoch >= args.warmup:
                if epoch % args.step == 0:
                    unfreeze_layers(model, epoch-args.warmup, args.step) #, num_layers)
        #run["Train/Loss"].log(epoch_loss)
        
        # evaluate on validation set
        if epoch % args.eval_interval == args.eval_interval - 1:
            y_true, y_pred = validate(val_loader, model, criterion, args)
            run=None
            acc, spec, sens, cfm = log_stats(run, y_true, y_pred, args)
            f1 = 2*(spec*sens)/(spec+sens)
            
            # remember best mcc and save checkpoint
            #is_best = acc > best_acc
            is_best = f1 > best_f1
            best_f1 = max(f1, best_f1)
            #if args.num_classes == 2:
            #    print("Epoch: %s\t Acc: %.4f\t Spec: %.4f\t Sens: %.4f\t Loss: %.4f"%(epoch, acc, spec, sens, epoch_loss))
            #else:
            #    acc_2 = (cfm[0][0]+cfm[0][1]+cfm[1][0]+cfm[1][1]+cfm[2][2])/np.sum(cfm)
            #    print("Epoch: %s\t Acc-2: %.4f\t Spec: %.4f\t Sens: %.4f\t Acc-3: %.4f\t Loss: %.4f"%(epoch, acc_2, spec, sens, acc, epoch_loss))
            #best_mcc = max(mcc, best_mcc)
            if is_best:
                best_cfm = copy.deepcopy(cfm)
                best_epoch = epoch

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'metrics': [acc, spec, sens],
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename='lincls_ep_%s.pth.tar'%(epoch), path=args.save_dir)

        if epoch>=args.warmup and args.cos_lr:
            lr_scheduler.step()

    #print('best cfm\n', best_cfm)
    cfm = best_cfm
    spec = (cfm[0][0]+cfm[0][1]+cfm[1][0]+cfm[1][1])/(np.sum(cfm[0])+np.sum(cfm[1]))
    sens = cfm[2][2]/np.sum(cfm[2])
    acc_2 = (cfm[0][0]+cfm[0][1]+cfm[1][0]+cfm[1][1]+cfm[2][2])/np.sum(cfm)
    acc = (cfm[0][0]+cfm[1][1]+cfm[2][2])/np.sum(cfm)
    print("%s %.4f %.4f %.4f %.4f"%(best_epoch, acc_2, spec, sens, acc))


def unfreeze_layers(model, epoch, steps, num_layers=16):
    LAYERS = [
                ("layer4", 2),
                ("layer4", 1),
                ("layer4", 0),
                ("layer3", 5),
                ("layer3", 4),
                ("layer3", 3),
                ("layer3", 2),
                ("layer3", 1),
                ("layer3", 0),
                ("layer2", 3),
                ("layer2", 2),
                ("layer2", 1),
                ("layer2", 0),
                ("layer1", 2),
                ("layer1", 1),
                ("layer1", 0)
            ]
    for j in range(num_layers):
        if epoch >= j*steps and epoch < (j+1)*steps:
            layer, conv = LAYERS[j]
            layername = "%s.%s"%(layer, conv)
            for name, params in model.named_parameters():
                if layername in name and params.requires_grad == False:
                    params.requires_grad = True
    if epoch == num_layers*steps:
        for name, params in model.named_parameters():
            if params.requires_grad == False:
                params.requires_grad = True

    
def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    #model.eval()
    running_loss = 0.0
    for i, (images, target, _) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data.item()
    
    return running_loss/len(train_loader)

def mcc_score(cfm):
    tp = cfm[1][1]
    tn = cfm[0][0]
    fp = cfm[0][1]
    fn = cfm[1][0]
    numer = (tp*tn)-(fp*fn)
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return numer/denom


def validate(val_loader, model, criterion, args):
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        for i, (images, target, _) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            _, pred = torch.max(output, dim=1)
            
            #loss = criterion(output, target)
            y_true.append(target.tolist()[0])
            y_pred.append(pred.item())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

    return y_true, y_pred

def log_stats(logobj, y_true, y_pred, args, label="Eval"):
    acc = accuracy_score(y_true, y_pred)
    cfm = confusion_matrix(y_true, y_pred)
    if args.num_classes == 2:
        spec = cfm[0][0]/np.sum(cfm[0])
        sens = cfm[1][1]/np.sum(cfm[1])
    else:
        spec = (cfm[0][0]+cfm[0][1]+cfm[1][0]+cfm[1][1])/(np.sum(cfm[0])+np.sum(cfm[1]))
        sens = cfm[2][2]/np.sum(cfm[2])
        acc_2 = (cfm[0][0]+cfm[0][1]+cfm[1][0]+cfm[1][1]+cfm[2][2])/np.sum(cfm)
        #logobj["%s/Acc-2cls"%label].log(acc_2)
    #mcc = mcc_score(cfm)
    
    #logobj["%s/Accuracy"%label].log(acc)
    #logobj["%s/MCC"%label].log(mcc)
    #logobj["%s/Specificity"%label].log(spec)
    #logobj["%s/Sensitivity"%label].log(sens)
    
    return acc, spec, sens, cfm 


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', path=None):
    if path is not None:
        filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_lincls.pth.tar')


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    #print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    #print("=> sanity check passed.")

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
