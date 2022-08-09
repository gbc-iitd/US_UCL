import os
import sys
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim

from tools.my_dataset import COVIDDataset
from resnet_uscl import ResNetUSCL


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


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp
    print("Apex on, run on mixed precision.")
    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\nRunning on:", device)

if device == 'cuda':
    device_name = torch.cuda.get_device_name()
    print("The device name is:", device_name)
    cap = torch.cuda.get_device_capability(device=None)
    print("The capability of this device is:", cap, '\n')
"""

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(device,split):
    # ============================ step 1/5 data ============================
    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
    ])

    # MyDataset
    train_data = COVIDDataset(data_dir=data_dir, train=True, transform=train_transform)
    valid_data = COVIDDataset(data_dir=data_dir, train=False, transform=valid_transform)

    # DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

    # ============================ step 2/5 model ============================

    if pretrained:
        print('\nThe ImageNet pretrained parameters are loaded.')
    else:
        print('\nThe ImageNet pretrained parameters are not loaded.')
    
    num_classes = 3

    net = models.__dict__["resnet50"](pretrained=pretrained)
    num_ftrs = net.fc.in_features
    
    net.fc = nn.Linear(num_ftrs, num_classes)
    # init the fc layer
    net.fc.weight.data.normal_(mean=0.0, std=0.01)
    net.fc.bias.data.zero_()
    
    num_layers=16

    # load from pre-trained, before DistributedDataParallel constructor
    if selfsup:
        #print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(state_dict_path, map_location="cpu")

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
            msg = net.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(state_dict_path))

    else:
        print('\nThe self-supervised trained parameters are not loaded.\n')
            
    

    # frozen all convolutional layers
    # for param in net.parameters():
    #     param.requires_grad = False

    # fine-tune last 3 layers
    for name, param in net.named_parameters():
        if not name.startswith('layer4.1'):
            param.requires_grad = False

    # add a classifier for linear evaluation
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 3)
    #net.fc = nn.Linear(3, 3)

    for name, param in net.named_parameters():
        print(name, '\t', 'requires_grad=', param.requires_grad)

    net.to(device)
    print(net)

    # ============================ step 3/5 loss function ============================
    criterion = nn.CrossEntropyLoss()       # choose loss function

    # ============================ step 4/5 optimizer ============================
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=weight_decay)      # choose optimizer
    """
    optimizer = torch.optim.SGD(net.parameters(), LR,
                                momentum=0.9,
                                weight_decay=weight_decay)
    """
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=MAX_EPOCH, 
                                                     eta_min=0,
                                                     last_epoch=-1)     # set learning rate decay strategy


    # ============================ step 5/5 training ============================
    print('\nTraining start!\n')
    start = time.time()
    train_curve = list()
    valid_curve = list()
    max_acc = 0.
    reached = 0    # which epoch reached the max accuracy

    # the statistics of classification result: classification_results[true][pred]
    classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    best_classification_results = None

    if apex_support and fp16_precision:
        net, optimizer = amp.initialize(net, optimizer,
                                        opt_level='O2',
                                        keep_batchnorm_fp32=True)
    for epoch in range(MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            if apex_support and fp16_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # update weights
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()

            # print training information
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("\nTraining:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.

        print('Learning rate this epoch:', scheduler.get_last_lr()[0])
        scheduler.step()  # updata learning rate

        # validate the model
        if (epoch+1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).cpu().squeeze().sum().numpy()
                    for k in range(len(predicted)):
                        classification_results[labels[k]][predicted[k]] += 1    # "label" is regarded as "predicted"

                    loss_val += loss.item()

                acc = correct_val / total_val
                if acc > max_acc:   # record best accuracy
                    max_acc = acc
                    reached = epoch
                    best_classification_results = classification_results
                    torch.save(net.state_dict(),'/home/somanshu/scratch/POCUS_ours_'+str(split)+'.pth.tar')
                classification_results = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                valid_curve.append(loss_val/valid_loader.__len__())
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, acc))

    
    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(MAX_EPOCH, round(time.time() - start)))
    print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc, reached))

    print('\nThe best prediction results of the dataset:')
    print('Class 0 predicted as class 0:', best_classification_results[0][0])
    print('Class 0 predicted as class 1:', best_classification_results[0][1])
    print('Class 0 predicted as class 2:', best_classification_results[0][2])
    print('Class 1 predicted as class 0:', best_classification_results[1][0])
    print('Class 1 predicted as class 1:', best_classification_results[1][1])
    print('Class 1 predicted as class 2:', best_classification_results[1][2])
    print('Class 2 predicted as class 0:', best_classification_results[2][0])
    print('Class 2 predicted as class 1:', best_classification_results[2][1])
    print('Class 2 predicted as class 2:', best_classification_results[2][2])

    acc0 = best_classification_results[0][0] / sum(best_classification_results[i][0] for i in range(3))
    recall0 = best_classification_results[0][0] / sum(best_classification_results[0])
    print('\nClass 0 accuracy:', acc0)
    print('Class 0 recall:', recall0)
    print('Class 0 F1:', 2 * acc0 * recall0 / (acc0 + recall0))

    acc1 = best_classification_results[1][1] / sum(best_classification_results[i][1] for i in range(3))
    recall1 = best_classification_results[1][1] / sum(best_classification_results[1])
    print('\nClass 1 accuracy:', acc1)
    print('Class 1 recall:', recall1)
    print('Class 1 F1:', 2 * acc1 * recall1 / (acc1 + recall1))

    acc2 = best_classification_results[2][2] / sum(best_classification_results[i][2] for i in range(3))
    recall2 = best_classification_results[2][2] / sum(best_classification_results[2])
    print('\nClass 2 accuracy:', acc2)
    print('Class 2 recall:', recall2)
    print('Class 2 F1:', 2 * acc2 * recall2 / (acc2 + recall2))
    
    return best_classification_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='linear evaluation')
    parser.add_argument('-p', '--path', default='checkpoint', help='folder of ckpt')
    parser.add_argument('-g', '--gpu', default=2,type=int, help='GPU ID to use')
    parser.add_argument('-c', '--ckpt', default='59', help='ckpt to use')
    args = parser.parse_args()

    set_seed(1)  # random seed

    # parameters
    MAX_EPOCH = 100       # default = 100
    BATCH_SIZE = 32      # default = 32
    LR = 0.01       # default = 0.01
    weight_decay = 1e-4   # default = 1e-4
    log_interval = 10
    val_interval = 1
    base_path = "./eval_pretrained_model/"
    state_dict_path = args.path
    device = args.gpu

    if torch.cuda.is_available():
        device = "cuda:"+str(device)
    else:
        device = "cpu"

    print(device)
    state_dict_path = os.path.join("/home/somanshu/scratch/cyclecontrast/output",args.path, args.ckpt)
    print('State dict path:', state_dict_path)
    fp16_precision = True
    pretrained = True
    selfsup = True

    # save result
    """
    save_dir = os.path.join('result')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    resultfile = save_dir + '/my_result.txt'
    """

    print(os.getcwd())
    print(os.path.exists(state_dict_path))
    #print(os.path.exists(resultfile))
    #print(os.path.isdir())
    if (os.path.exists(state_dict_path)):
        confusion_matrix = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for i in range(1, 6):
            print('\n' + '='*20 + 'The training of fold {} start.'.format(i) + '='*20)
            data_dir = "./scratch/pocus/covid_data{}.pkl".format(i)
            best_classification_results = main(device,i)
            confusion_matrix = confusion_matrix + np.array(best_classification_results)

        print('\nThe confusion matrix is:')
        print(confusion_matrix)
        print('\nThe precision of class 0 is:', confusion_matrix[0,0] / sum(confusion_matrix[:,0]))
        print('The precision of class 1 is:', confusion_matrix[1,1] / sum(confusion_matrix[:,1]))
        print('The precision of class 2 is:', confusion_matrix[2,2] / sum(confusion_matrix[:,2]))
        print('\nThe recall of class 0 is:', confusion_matrix[0,0] / sum(confusion_matrix[0]))
        print('The recall of class 1 is:', confusion_matrix[1,1] / sum(confusion_matrix[1]))
        print('The recall of class 2 is:', confusion_matrix[2,2] / sum(confusion_matrix[2]))
        
        print("****************************")
        print("*****************")
        
        print('\nTotal acc is:', (confusion_matrix[0,0]+confusion_matrix[1,1]+confusion_matrix[2,2])/confusion_matrix.sum())


        print('\nCOVID acc is:',(confusion_matrix[0][0]/np.sum(confusion_matrix[0])))

        print('\nPneumonia acc is:',(confusion_matrix[1][1]/np.sum(confusion_matrix[1]))) 

        print('\nNormal acc is:',(confusion_matrix[2][2]/np.sum(confusion_matrix[2])))

        print("****************************")
        print("*****************")

        