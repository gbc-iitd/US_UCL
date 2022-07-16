import torch
import torch.nn as nn
import torch.nn.functional as F
import ucl.resnet as models
import cv2
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CycleContrast(nn.Module):
    """
    Build a CycleContrast model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/2105.06463
    """
    def __init__(self, arch, dim=128,
                 K=65536, m=0.999,
                 T=0.07, mlp=False,
                 soft_nn=False, soft_nn_support=-1,
                 sep_head=False, cycle_neg_only=True,
                 soft_nn_T=-1.,
                 cycle_back_cls=False,
                 cycle_back_cls_video_as_pos=False,
                 cycle_K=None,
                 moco_random_video_frame_as_pos=False,
                 pretrained_on_imagenet=False,
                 soft_nn_topk_support=True,
                 negative_use = True,
                 neg_queue_size=32,
                 intranegs_only_two=True,
                 cross_neg_topk_mining= True,
                 cross_neg_topk_support_size = 4,
                 anchor_reverse_cross = True,
                 single_loss_intra_inter= True,
                 single_loss_ncap_support_size = 4,
                 qcap_include = False,
                 tsne_name="",
                 mean_neighbors= False
                 ):
        super(CycleContrast, self).__init__()

        self.K = K  ## 65536 size of queue
        self.tsne_name = tsne_name
        if cycle_K is not None:
            self.cycle_K = cycle_K  ## 81920
        else:
            self.cycle_K = K
        self.m = m ## 0.999
        self.T = T ## 0.07

        if soft_nn_T != -1:
            self.soft_nn_T = soft_nn_T
        else:
            self.soft_nn_T = T   ## soft_nn_T gets the value of T for us [0.07]

        self.soft_nn = soft_nn  ## True
        self.soft_nn_support = soft_nn_support ## 16384
        self.soft_nn_topk_support = soft_nn_topk_support
        self.sep_head = sep_head ## True
        self.cycle_back_cls = cycle_back_cls ## True
        self.cycle_neg_only = cycle_neg_only ## True
        self.cycle_back_cls_video_as_pos = cycle_back_cls_video_as_pos ## True
        self.moco_random_video_frame_as_pos = moco_random_video_frame_as_pos ## True
        self.neg_queue_size = neg_queue_size
        self.negative_use = negative_use
        self.intranegs_only_two = intranegs_only_two
        self.cross_neg_topk_mining = cross_neg_topk_mining
        self.cross_neg_topk_support_size = cross_neg_topk_support_size
        self.anchor_reverse_cross = anchor_reverse_cross
        self.single_loss_intra_inter = single_loss_intra_inter
        self.single_loss_ncap_support_size = single_loss_ncap_support_size
        self.qcap_include = qcap_include
        self.mean_neighbors = mean_neighbors
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = models.__dict__[arch](pretrained=pretrained_on_imagenet, progress= pretrained_on_imagenet,num_classes=dim,
                                               return_inter=True)
        self.encoder_k = models.__dict__[arch](pretrained=pretrained_on_imagenet, progress= pretrained_on_imagenet,num_classes=dim,
                                               return_inter=sep_head)

        ## return inter is True because they use a seperate MLP Head

        # sep head
        if self.sep_head:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.q_cycle_fc = nn.Linear(dim_mlp, dim)
            self.k_cycle_fc = nn.Linear(dim_mlp, dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

            if self.sep_head:
                dim_mlp = self.q_cycle_fc.weight.shape[1]
                self.q_cycle_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.q_cycle_fc)
                self.k_cycle_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.k_cycle_fc)

        ## This code above defines the head/fc layer for us

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if self.sep_head:
            for param_q, param_k in zip(self.q_cycle_fc.parameters(), self.k_cycle_fc.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))  ## initialize the queue by random K vectors
        self.queue = nn.functional.normalize(self.queue, dim=0)

        if self.sep_head:
            self.register_buffer("queue_cycle", torch.randn(dim, self.cycle_K))
            self.queue_cycle = nn.functional.normalize(self.queue_cycle, dim=0)
            self.register_buffer("queue_cycle_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_labels", torch.zeros(1, K, dtype=torch.long))

        self.register_buffer("queue_indices", torch.zeros(1, K, dtype=torch.long))

        #########
        self.register_buffer("neg_vecs_labels",torch.zeros(1, self.cycle_K,dtype=torch.long))
        self.register_buffer("label_ptr", torch.zeros(1, dtype=torch.long))
        #########

        self.register_buffer("negatives",torch.zeros(dim, self.neg_queue_size))
        self.register_buffer("negative_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        if self.sep_head:
            for param_q, param_k in zip(self.q_cycle_fc.parameters(), self.k_cycle_fc.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_cycle=None, labels=None, indices=None, cluster=None,metadata_q_tsne=None):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        if keys_cycle is not None:
            keys_cycle = concat_all_gather(keys_cycle)
        if labels is not None:
            labels = concat_all_gather(labels)
        if indices is not None:
            indices = concat_all_gather(indices)
        if cluster is not None:
            cluster = concat_all_gather(cluster)        
        
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[0])
        # if some gpu of the machine is dead, K might not able to be divided by batch size
        assert self.K % batch_size == 0, 'K {} can not be divided by batch size {}'.format(self.K, batch_size)  # for simplicity

        if self.sep_head:
            cycle_ptr = int(self.queue_cycle_ptr[0])
            # if some gpu of the machine is dead, K might not able to be divided by batch size
            assert self.cycle_K % batch_size == 0, \
                'K {} can not be divided by batch size {}'.format(self.cycle_K,
                                                                  batch_size)  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        try:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        except Exception:
            print('enqueue size', ptr, batch_size, self.K)
            enqueue_size = min(self.K-ptr, batch_size)
            self.queue[:, ptr:ptr + batch_size] = keys[:enqueue_size].T
        try:
            if self.sep_head and keys_cycle is not None:
                self.queue_cycle[:, cycle_ptr:cycle_ptr + batch_size] = keys_cycle.T
        except Exception:
            print('enqueue size', ptr, keys_cycle.shape[0], self.K)
            enqueue_size = min(self.K - ptr, keys_cycle.shape[0])
            self.queue_cycle[:, ptr:ptr + keys_cycle.shape[0]] = keys_cycle[:enqueue_size].T
        try:
            if labels is not None:
                self.queue_labels[:, ptr:ptr + batch_size] = labels
            if indices is not None:
                self.queue_indices[:, ptr:ptr + batch_size] = indices
            if cluster is not None:
                self.queue_cluster[:, ptr:ptr + batch_size] = cluster.T
        except Exception:
            enqueue_size = min(self.K-ptr, batch_size)
            if labels is not None:
                self.queue_labels[:, ptr:ptr + batch_size] = labels[:enqueue_size]
            if indices is not None:
                self.queue_indices[:, ptr:ptr + batch_size] = indices[:enqueue_size]
            if cluster is not None:
                self.queue_cluster[:, ptr:ptr + batch_size] = cluster[:enqueue_size].T


        #############################
        label_ptr = self.label_ptr[0]
        try:
            if self.sep_head and metadata_q_tsne is not None:
                self.neg_vecs_labels[:,label_ptr:label_ptr + batch_size] = metadata_q_tsne.T
        except Exception:
            print('error')
        ##############################            

        ptr = (ptr + batch_size) % self.K  # move pointer
        assert ptr < self.K, 'ptr: {}, batch_size: {}, K: {}'.format(ptr, batch_size, self.K)

        self.queue_ptr[0] = ptr

        if self.sep_head:
            cycle_ptr = (cycle_ptr + batch_size) % self.cycle_K  # move pointer
            assert cycle_ptr < self.cycle_K, 'cycle ptr: {}, batch_size: {}, cycle K: {}'.format(
                cycle_ptr, batch_size, self.cycle_K)

            self.queue_cycle_ptr[0] = cycle_ptr

        ################################
        if self.sep_head:
            label_ptr = (label_ptr + batch_size) % self.cycle_K  # move pointer
            assert label_ptr < self.cycle_K, 'label ptr: {}, batch_size: {}, cycle K: {}'.format(
                cycle_ptr, batch_size, self.cycle_K)

            self.label_ptr[0] = label_ptr
        #################################



    
    @torch.no_grad()
    def _dequeue_and_enqueue_negs(self,negatives=None):

        # gather keys before updating queue
        if negatives is not None:
            negatives = concat_all_gather(negatives)
    
        neg_queue_ptr = self.negative_queue_ptr[0]
        num_negs = negatives.shape[0]
        try:
            if  negatives is not None:
                if (neg_queue_ptr+num_negs) <= self.neg_queue_size:
                    self.negatives[:,neg_queue_ptr:neg_queue_ptr + num_negs] = negatives.T
                elif num_negs > self.neg_queue_size:
                    num_negs = self.neg_queue_size
                    self.negatives[:,neg_queue_ptr:neg_queue_ptr + num_negs] = negatives.T[:,0:num_negs]
                else:
                    diff = (neg_queue_ptr+num_negs - self.neg_queue_size)
                    fit = self.neg_queue_size-neg_queue_ptr
                    self.negatives[:,neg_queue_ptr:neg_queue_ptr+fit] = negatives.T[:,0:fit]
                    self.negatives[:,0:diff]=negatives.T[:,fit:fit+diff]

        except Exception:
            print('error')
            
        if negatives is not None:
            neg_queue_ptr = (neg_queue_ptr + num_negs) % self.neg_queue_size  # move pointer
            assert neg_queue_ptr < self.neg_queue_size, 'Neg ptr: {}, Num Negs: {}, Neg Queue Size: {}'.format(
                neg_queue_ptr, num_negs, self.neg_queue_size)

            self.negative_queue_ptr[0] = neg_queue_ptr


    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k=None,
                cls_labels=None, indices=None,
                cls_candidates=None, im_negs=None, tsne=False, only_qcap= False
                ):

        ## im_negs is of the shape 2x2x3x224x224
        ## want to convert it to 4x3x224x224

        im_negs_temp = None
        num_samples_to_handle = im_negs.shape[0]
        for itr_var in range(num_samples_to_handle):
            if itr_var ==0:
                im_negs_temp = im_negs[itr_var]
            else:
                temporary_negatives = im_negs[itr_var]
                im_negs_temp = torch.cat((im_negs_temp,temporary_negatives),axis=0)

        im_negs = im_negs_temp ## 4x3x224x224

        if self.negative_use is not None:
            assert im_negs is not None, "No negatives found even when negative flag activated"
            unnormalize_negs = self.encoder_q(im_negs)  # queries: NxC
            
            unnormalize_negs, negs_avgpool = unnormalize_negs
            if self.sep_head:
                negs_cycle = self.q_cycle_fc(negs_avgpool)
                negs_cycle = nn.functional.normalize(negs_cycle, dim=1)
            negs = nn.functional.normalize(unnormalize_negs, dim=1)
            negs_curr_gpu = negs.clone()
            
            self._dequeue_and_enqueue_negs(negs)
        
        # compute query features
        unnormalize_q = self.encoder_q(im_q)  # queries: NxC
        
        unnormalize_q, q_avgpool = unnormalize_q
        if self.sep_head:
            q_cycle = self.q_cycle_fc(q_avgpool)
            q_cycle = nn.functional.normalize(q_cycle, dim=1)
        q = nn.functional.normalize(unnormalize_q, dim=1)
        #print(q.shape)
        meta = {}
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            if (self.cycle_back_cls or self.moco_random_video_frame_as_pos) and cls_candidates is not None:
                n, k_can = cls_candidates.shape[:2]
                concat_im = torch.cat((im_k, cls_candidates.flatten(0, 1)), dim=0)
                
                
                im_k, idx_unshuffle = self._batch_shuffle_ddp(concat_im)
                k = self.encoder_k(im_k)  # keys: NxC
                
                if self.sep_head:
                    k, k_avgpool = k
                    k_cycle = self.k_cycle_fc(k_avgpool)
                    k_cycle = nn.functional.normalize(k_cycle, dim=1)
                    k_cycle = self._batch_unshuffle_ddp(k_cycle, idx_unshuffle)
                    k_cycle, can_cycle = torch.split(k_cycle, [n, n * k_can], dim=0)
                    can_cycle = can_cycle.view(n, k_can, -1)
                k = nn.functional.normalize(k, dim=1)
                
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                k, can = torch.split(k, [n, n * k_can], dim=0)
                can = can.view(n, k_can, -1)

            else:
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                k = self.encoder_k(im_k)  # keys: NxC
                if self.sep_head:
                    k, k_avgpool = k
                    k_cycle = self.k_cycle_fc(k_avgpool)
                    k_cycle = nn.functional.normalize(k_cycle, dim=1)
                    k_cycle = self._batch_unshuffle_ddp(k_cycle, idx_unshuffle)
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        if self.sep_head:
            l_pos_cycle = torch.einsum('nc,nc->n', [q_cycle, k_cycle]).unsqueeze(-1)
            # negative logits: NxK
            l_neg_cycle = torch.einsum('nc,ck->nk', [q_cycle, self.queue_cycle.clone().detach()])
        else:
            l_pos_cycle = l_pos
            l_neg_cycle = l_neg
        

        if self.soft_nn:
            if self.cycle_neg_only:
                """ In case we want topk based sampling"""
                if self.soft_nn_topk_support:
                    nn_embs, perm_indices,sampled_indices_tsne,wts_tsne = \
                        self.cycle_back_topk_queue_without_self(l_neg_cycle, self.queue_cycle \
                            if self.sep_head else self.queue,can.device)
                else:
                    nn_embs, perm_indices,sampled_indices_tsne,wts_tsne = \
                        self.cycle_back_queue_without_self(l_neg_cycle, self.queue_cycle \
                            if self.sep_head else self.queue,can.device)
            else:
                pos_neigbor = q_cycle if self.sep_head else q
                nn_embs, perm_indices = \
                    self.cycle_back_queue_with_self(l_pos_cycle, l_neg_cycle,
                                                    pos_neigbor,
                                                    self.queue_cycle if self.sep_head else self.queue,
                                                    meta=meta,
                                                    q=q, k=k
                                                    )
            
            nn_embs = F.normalize(nn_embs, dim=1)

        

        sampled_indices_logits = sampled_indices_tsne.clone() 

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        if self.cycle_back_cls:
            assert cls_candidates is not None, "cls candidates can not be None"
            """ In case we want topk based sampling"""
            if self.negative_use:
                if not self.cross_neg_topk_mining:
                    back_logits, cycle_back_labels = \
                            self.get_logits_cycle_back_cls_video_as_pos_negs(nn_embs,
                                                                        can_cycle if self.sep_head else can,
                                                                        self.negatives)
                else:
                    if not self.anchor_reverse_cross:
                        back_logits, cycle_back_labels = \
                                self.get_logits_cycle_back_cls_video_as_pos_topk_negs(nn_embs,
                                                                            can_cycle if self.sep_head else can,
                                                                            self.negatives)
                    else:
                        back_logits, cycle_back_labels = \
                                self.get_logits_cycle_back_cls_video_as_pos_topk_negs_anchor_rev(nn_embs,
                                                                            can_cycle if self.sep_head else can,
                                                                            self.negatives)
            else:
                if self.soft_nn_topk_support:
                    back_logits, cycle_back_labels = \
                        self.get_logits_topk_cycle_back_cls_video_as_pos(nn_embs,
                                                                    can_cycle if self.sep_head else can,
                                                                    self.queue_cycle if self.sep_head else self.queue,
                                                                    sampled_indices_logits,
                                                                    indices=indices)
                else:
                    back_logits, cycle_back_labels = \
                        self.get_logits_cycle_back_cls_video_as_pos(nn_embs,
                                                                    can_cycle if self.sep_head else can,
                                                                    self.queue_cycle if self.sep_head else self.queue,
                                                                    perm_indices,
                                                                    indices=indices)

        # random sample one frame from the same video of im_q / im_k as positive, i.e. intra-video objective
        if self.moco_random_video_frame_as_pos:
            if self.negative_use:
                if not self.intranegs_only_two:
                    """ Using N complete """
                    #  can: n x k x 128
                    pos_indices = torch.randint(high=can.shape[1], size=(q.shape[0],), device=can.device)
                    pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can.shape[2])

                    logits_neg = torch.matmul(q, self.negatives.clone().detach())
                    logits_pos = torch.einsum('nc, nc->n', [q, torch.gather(can, 1, pos_indices).squeeze()]).unsqueeze(-1)
                    logits = torch.cat([logits_pos, logits_neg], dim=1)
                    logits /= self.T
                    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                
                if self.intranegs_only_two and (not self.qcap_include):
                    """ Using n1,n2 .... """
                    pos_indices = torch.randint(high=can.shape[1], size=(q.shape[0],), device=can.device)
                    pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can.shape[2])

                    logits_neg = None
                    num_samples_curr_gpu = q.shape[0]
                    num_negs_curr_gpu = negs_curr_gpu.shape[0]
                    negs_per_q = int(num_negs_curr_gpu/num_samples_curr_gpu)

                    for loop_variable in range(num_samples_curr_gpu):
                        q_sample = q[loop_variable].unsqueeze(0) ## 1,128

                        if loop_variable ==0:
                            negatives_to_be_used = negs_curr_gpu[:loop_variable+negs_per_q].T  ## 128x2
                            logits_neg = torch.matmul(q_sample,negatives_to_be_used)

                        else:
                            negatives_to_be_used = negs_curr_gpu[(loop_variable*negs_per_q):(loop_variable*negs_per_q)+negs_per_q].T  ## 128x2
                            logits_curr_sample = torch.matmul(q_sample,negatives_to_be_used)
                            logits_neg = torch.cat((logits_neg,logits_curr_sample),axis=0) ## (loop_variable+1)x 2

                    logits_pos = torch.einsum('nc, nc->n', [q, torch.gather(can, 1, pos_indices).squeeze()]).unsqueeze(-1)  ## 2,128   x 2,128 kind of multiplication
                    logits = torch.cat([logits_pos, logits_neg], dim=1)
                    logits /= self.T
                    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                

                if self.single_loss_intra_inter and self.intranegs_only_two and self.qcap_include and only_qcap:
                    """ Using only n_cap"""

                    pos_indices = torch.randint(high=can.shape[1], size=(q.shape[0],), device=can.device)
                    pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can.shape[2])
                    N_set = self.negatives.detach().clone()

                    logits_neg = None
                    num_samples_curr_gpu = q.shape[0]
                    num_negs_curr_gpu = negs_curr_gpu.shape[0]
                    negs_per_q = int(num_negs_curr_gpu/num_samples_curr_gpu)

                    for loop_variable in range(num_samples_curr_gpu):
                        q_sample = q[loop_variable].unsqueeze(0) ## 1,128

                        if loop_variable ==0:
                            negatives_to_be_used = negs_curr_gpu[:loop_variable+negs_per_q].T  ## 128x2
                            
                            sim_wts = torch.matmul(q_sample,N_set)
                            (wts_selected,index_used) = sim_wts.topk(self.single_loss_ncap_support_size,dim=1)
                            wts_selected = wts_selected/self.soft_nn_T
                            wts_selected = torch.nn.functional.softmax(wts_selected,dim=1)
                            selected_negs = N_set[:,index_used[0]]
                            n_cap = torch.matmul(wts_selected,selected_negs.T)
                            if self.mean_neighbors:
                                n_cap = torch.mean(N_set,1,True).T
                            logits_neg = torch.matmul(q_sample,n_cap.T)

                        else:
                            negatives_to_be_used = negs_curr_gpu[(loop_variable*negs_per_q):(loop_variable*negs_per_q)+negs_per_q].T  ## 128x2
                            
                            sim_wts = torch.matmul(q_sample,N_set)
                            (wts_selected,index_used) = sim_wts.topk(self.single_loss_ncap_support_size,dim=1)
                            wts_selected = wts_selected/self.soft_nn_T
                            wts_selected = torch.nn.functional.softmax(wts_selected,dim=1)
                            selected_negs = N_set[:,index_used[0]]
                            n_cap = torch.matmul(wts_selected,selected_negs.T)
                            if self.mean_neighbors:
                                n_cap = torch.mean(N_set,1,True).T
                            logits_curr_sample = torch.matmul(q_sample,n_cap.T)
                            logits_neg = torch.cat((logits_neg,logits_curr_sample),axis=0) ## (loop_variable+1)x 2

                    logits_pos = torch.einsum('nc, nc->n', [q, torch.gather(can, 1, pos_indices).squeeze()]).unsqueeze(-1)  ## 2,128   x 2,128 kind of multiplication
                    logits = torch.cat([logits_pos, logits_neg], dim=1)
                    logits /= self.T
                    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()



                if self.single_loss_intra_inter and self.intranegs_only_two and self.qcap_include and not(only_qcap):
                    """ Using n1,n2.... with n_cap"""
                    
                    pos_indices = torch.randint(high=can.shape[1], size=(q.shape[0],), device=can.device)
                    pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can.shape[2])
                    N_set = self.negatives.detach().clone()

                    logits_neg = None
                    num_samples_curr_gpu = q.shape[0]
                    num_negs_curr_gpu = negs_curr_gpu.shape[0]
                    negs_per_q = int(num_negs_curr_gpu/num_samples_curr_gpu)

                    for loop_variable in range(num_samples_curr_gpu):
                        q_sample = q[loop_variable].unsqueeze(0) ## 1,128

                        if loop_variable ==0:
                            negatives_to_be_used = negs_curr_gpu[:loop_variable+negs_per_q].T  ## 128x2
                            
                            sim_wts = torch.matmul(q_sample,N_set)
                            (wts_selected,index_used) = sim_wts.topk(self.single_loss_ncap_support_size,dim=1)
                            wts_selected = wts_selected/self.soft_nn_T
                            wts_selected = torch.nn.functional.softmax(wts_selected,dim=1)
                            selected_negs = N_set[:,index_used[0]]
                            n_cap = torch.matmul(wts_selected,selected_negs.T)
                            if self.mean_neighbors:
                                n_cap = torch.mean(N_set,1,True).T
                            negatives_to_be_used = torch.cat((negatives_to_be_used,n_cap.T),axis=1)
                    
                            logits_neg = torch.matmul(q_sample,negatives_to_be_used)

                        else:
                            negatives_to_be_used = negs_curr_gpu[(loop_variable*negs_per_q):(loop_variable*negs_per_q)+negs_per_q].T  ## 128x2
                            
                            sim_wts = torch.matmul(q_sample,N_set)
                            (wts_selected,index_used) = sim_wts.topk(self.single_loss_ncap_support_size,dim=1)
                            wts_selected = wts_selected/self.soft_nn_T
                            wts_selected = torch.nn.functional.softmax(wts_selected,dim=1)
                            selected_negs = N_set[:,index_used[0]]
                            n_cap = torch.matmul(wts_selected,selected_negs.T)
                            if self.mean_neighbors:
                                n_cap = torch.mean(N_set,1,True).T
                            negatives_to_be_used = torch.cat((negatives_to_be_used,n_cap.T),axis=1)

                            logits_curr_sample = torch.matmul(q_sample,negatives_to_be_used)
                            logits_neg = torch.cat((logits_neg,logits_curr_sample),axis=0) ## (loop_variable+1)x 2

                    logits_pos = torch.einsum('nc, nc->n', [q, torch.gather(can, 1, pos_indices).squeeze()]).unsqueeze(-1)  ## 2,128   x 2,128 kind of multiplication
                    logits = torch.cat([logits_pos, logits_neg], dim=1)
                    logits /= self.T
                    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            
            else:
                pos_indices = torch.randint(high=can.shape[1], size=(q.shape[0],), device=can.device)
                pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can.shape[2])

                logits_neg = torch.matmul(q, self.queue_cycle.clone().detach())
                logits_pos = torch.einsum('nc, nc->n', [q, torch.gather(can, 1, pos_indices).squeeze()]).unsqueeze(-1)
                logits = torch.cat([logits_pos, logits_neg], dim=1)
                logits /= self.T
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(k,
                                  keys_cycle=k_cycle if self.sep_head else None,
                                  labels=cls_labels, indices=indices
                                  )
        
        if self.soft_nn:
            if not self.single_loss_intra_inter:
                return logits, labels, back_logits, cycle_back_labels, meta
            else:
                return logits, labels, meta
        else:
            return logits, labels, meta

    def cycle_back_queue_without_self(self, l_neg_cycle, queue_cycle,device):
        if self.soft_nn_support != -1:
            perm_indices = torch.randperm(self.cycle_K if self.sep_head else self.K)
            sampled_indices = perm_indices[:self.soft_nn_support]
            sampled_l_neg = l_neg_cycle[:, sampled_indices]
            weights = nn.functional.softmax(sampled_l_neg / self.soft_nn_T, dim=1)
            nn_embs = torch.matmul(weights, queue_cycle.clone().detach().transpose(1, 0)[sampled_indices])
        else:
            weights = nn.functional.softmax(l_neg_cycle / self.soft_nn_T, dim=1)
            nn_embs = torch.matmul(weights, queue_cycle.clone().detach().transpose(1, 0))
            perm_indices = None

        sampled_indices_ret = sampled_indices.to(device)
        sampled_indices_ret = torch.reshape(sampled_indices_ret,(1,sampled_indices_ret.shape[0]))
        sampled_indices_ret = torch.cat((sampled_indices_ret,sampled_indices_ret),0)
        return nn_embs, perm_indices, sampled_indices_ret,weights

    def cycle_back_topk_queue_without_self(self, l_neg_cycle, queue_cycle,device):
        assert self.soft_nn_support != -1  #TODO: what to do when soft_nn_support == -1 i.e all elements
                                           #are to be used for soft nn calculation
        
        perm_indices = None

        ## Starting with all the vectors at first
        sampled_l_neg = l_neg_cycle
        weights = nn.functional.softmax(sampled_l_neg / self.soft_nn_T, dim=1)
        (weights,indices) = weights.topk(self.soft_nn_support,dim=1)

        weights = weights.unsqueeze(1)  ## Num_samples x 1 x self.soft_nn_support
        """ Getting Correct Weights for each sample in the batch based on topk"""
        num_samples = weights.shape[0]
        queue_cycle_wts = None

        
        for i in range(num_samples):
            if i ==0:
                queue_cycle_wts = queue_cycle.clone().detach().transpose(1, 0)[indices[i]].unsqueeze(0)
            else:
                temp_var = queue_cycle.clone().detach().transpose(1, 0)[indices[i]].unsqueeze(0)
                queue_cycle_wts = torch.cat((queue_cycle_wts,temp_var),0)


        nn_embs = torch.bmm(weights, queue_cycle_wts).squeeze()
        weights = torch.squeeze(weights)
        
        """ Some non important stuff for tsne plots/book keeping"""
        sampled_indices_ret = indices.to(device)

        return nn_embs, perm_indices, sampled_indices_ret,weights

    def cycle_back_queue_with_self(self, l_pos_cycle, l_neg_cycle, q_cycle, queue_cycle, meta=None, q=None, k=None):
        if self.soft_nn_support != -1:
            perm_indices = torch.randperm(self.cycle_K if self.sep_head else self.K)
            sampled_indices = perm_indices[:self.soft_nn_support]
            sampled_l_neg = l_neg_cycle[:, sampled_indices]
            logits_cycle = torch.cat([l_pos_cycle, sampled_l_neg], dim=1)
        else:
            logits_cycle = torch.cat([l_pos_cycle, l_neg_cycle], dim=1)
            perm_indices = None

        weights = nn.functional.softmax(logits_cycle / self.soft_nn_T, dim=1)
        num_neg = self.soft_nn_support if self.soft_nn_support != -1 \
            else (self.cycle_K if self.sep_head else self.K)
        weights_pos, weights_neg = torch.split(weights, [1, num_neg], dim=1)
        nn_embs_pos = weights_pos * q_cycle
        if self.soft_nn_support != -1:
            nn_embs_neg = torch.matmul(weights_neg, queue_cycle.clone().detach().transpose(1, 0)[sampled_indices])
        else:
            nn_embs_neg = torch.matmul(weights_neg, queue_cycle.clone().detach().transpose(1, 0))
        nn_embs = nn_embs_pos + nn_embs_neg  #
        return nn_embs, perm_indices

    def get_logits_cycle_back_cls_video_as_pos(self, nn_embs, can_cycle, queue_cycle, perm_indices, indices=None):
        if self.soft_nn_support == -1:
            back_logits_neg = torch.matmul(nn_embs, queue_cycle.clone().detach())
        else:
            remain_indices = perm_indices[self.soft_nn_support:]
            back_logits_neg = torch.matmul(nn_embs, queue_cycle.clone().detach()[:, remain_indices])
        pos_indices = torch.randint(high=can_cycle.shape[1], size=(nn_embs.shape[0],), device=can_cycle.device)
        pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can_cycle.shape[2])
        back_logits_pos = torch.einsum('nc, nc->n',
                                       [nn_embs, torch.gather(can_cycle, 1, pos_indices).squeeze()]).unsqueeze(-1)

        back_logits = torch.cat([back_logits_pos, back_logits_neg], dim=1)
        back_logits /= self.T

        cycle_back_labels = torch.zeros(back_logits.shape[0], dtype=torch.long).cuda()
        return back_logits, cycle_back_labels

    def get_logits_cycle_back_cls_video_as_pos_negs(self, nn_embs, can_cycle, negatives ):
        
        back_logits_neg = torch.matmul(nn_embs, negatives.clone().detach())
        pos_indices = torch.randint(high=can_cycle.shape[1], size=(nn_embs.shape[0],), device=can_cycle.device)
        pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can_cycle.shape[2])
        back_logits_pos = torch.einsum('nc, nc->n',
                                       [nn_embs, torch.gather(can_cycle, 1, pos_indices).squeeze()]).unsqueeze(-1)

        back_logits = torch.cat([back_logits_pos, back_logits_neg], dim=1)
        back_logits /= self.T

        cycle_back_labels = torch.zeros(back_logits.shape[0], dtype=torch.long).cuda()
        return back_logits, cycle_back_labels

    def get_logits_cycle_back_cls_video_as_pos_topk_negs(self, nn_embs, can_cycle, negatives ):
        
        similarity_scores = torch.matmul(nn_embs,negatives.clone().detach())  ## 2x32
        (wts,indices_hard_negs) = similarity_scores.topk(self.cross_neg_topk_support_size,dim=1)

        #negatives_copy = negatives.clone().T  #128,32
        topk_negatives = None
        num_samples = nn_embs.shape[0]

        for itr in range(num_samples):
            if itr ==0:
                topk_negatives = negatives[:,indices_hard_negs[itr]].unsqueeze(0)
            else:
                temp_element = negatives[:,indices_hard_negs[itr]].unsqueeze(0)
                topk_negatives = torch.cat((topk_negatives,temp_element),axis=0)

        nn_embs_reshaped = nn_embs.unsqueeze(1)
        back_logits_neg = torch.bmm(nn_embs_reshaped, topk_negatives)
        back_logits_neg = back_logits_neg.squeeze()


        pos_indices = torch.randint(high=can_cycle.shape[1], size=(nn_embs.shape[0],), device=can_cycle.device)
        pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can_cycle.shape[2])
        back_logits_pos = torch.einsum('nc, nc->n',
                                       [nn_embs, torch.gather(can_cycle, 1, pos_indices).squeeze()]).unsqueeze(-1)

        back_logits = torch.cat([back_logits_pos, back_logits_neg], dim=1)
        back_logits /= self.T

        cycle_back_labels = torch.zeros(back_logits.shape[0], dtype=torch.long).cuda()
        return back_logits, cycle_back_labels


    def get_logits_cycle_back_cls_video_as_pos_topk_negs_anchor_rev(self, nn_embs, can_cycle, negatives ):
        
        #print(can_cycle.shape)
        can_cycle_sim = can_cycle.clone().squeeze()
        
        similarity_scores = torch.matmul(can_cycle_sim,negatives.clone().detach())  ## 2x32
        (wts,indices_hard_negs) = similarity_scores.topk(self.cross_neg_topk_support_size,dim=1)

        #negatives_copy = negatives.clone().T  #128,32
        topk_negatives = None
        num_samples = nn_embs.shape[0]

        for itr in range(num_samples):
            if itr ==0:
                topk_negatives = negatives[:,indices_hard_negs[itr]].unsqueeze(0)
            else:
                temp_element = negatives[:,indices_hard_negs[itr]].unsqueeze(0)
                topk_negatives = torch.cat((topk_negatives,temp_element),axis=0)

        can_copy = can_cycle.clone()
        back_logits_neg = torch.bmm(can_copy, topk_negatives)
        back_logits_neg = back_logits_neg.squeeze()

        pos_indices = torch.randint(high=can_cycle.shape[1], size=(nn_embs.shape[0],), device=can_cycle.device)
        pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can_cycle.shape[2])
        back_logits_pos = torch.einsum('nc, nc->n',
                                       [nn_embs, torch.gather(can_cycle, 1, pos_indices).squeeze()]).unsqueeze(-1)

        back_logits = torch.cat([back_logits_pos, back_logits_neg], dim=1)
        back_logits /= self.T

        cycle_back_labels = torch.zeros(back_logits.shape[0], dtype=torch.long).cuda()
        return back_logits, cycle_back_labels

    def get_logits_topk_cycle_back_cls_video_as_pos(self, nn_embs, can_cycle, queue_cycle, sampled_indices, indices=None):
        assert self.soft_nn_support != -1  #TODO: what to do when soft_nn_support == -1 i.e all elements
                                           #are to be used for soft nn calculation
        
        num_samples = nn_embs.shape[0]

        """ Getting the leftover sample indices from queue"""
        remain_indices = None

        for i in range(num_samples):
            used_indices = sampled_indices[i]
            use_indices_np = used_indices.cpu().detach().numpy()
            total_possible_indices = np.arange(0,queue_cycle.shape[1],1)
            remain_indices_temp = [i for i in total_possible_indices if i not in use_indices_np] 

            remain_indices_temp = torch.Tensor(remain_indices_temp).unsqueeze(0)
            remain_indices_temp = remain_indices_temp.long()

            if i ==0:
                remain_indices = remain_indices_temp
            else:
                remain_indices = torch.cat((remain_indices,remain_indices_temp),0)
        

        """ Getting the leftover sample embeddings based on the indices generated above"""
        queue_cycle_wts= None
        for i in range(num_samples):
            remain_index_sample = remain_indices[i]

            if i ==0:
                queue_cycle_wts = queue_cycle.clone().detach()[:, remain_index_sample].unsqueeze(0)
            else:
                temp_var = queue_cycle.clone().detach()[:, remain_index_sample].unsqueeze(0)
                queue_cycle_wts = torch.cat((queue_cycle_wts,temp_var),0)

        nn_embs = nn_embs.unsqueeze(1)
        
        back_logits_neg = torch.bmm(nn_embs, queue_cycle_wts)

        nn_embs = nn_embs.squeeze()
        back_logits_neg = back_logits_neg.squeeze()
        
        pos_indices = torch.randint(high=can_cycle.shape[1], size=(nn_embs.shape[0],), device=can_cycle.device)
        pos_indices = pos_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, can_cycle.shape[2])
        back_logits_pos = torch.einsum('nc, nc->n',
                                       [nn_embs, torch.gather(can_cycle, 1, pos_indices).squeeze()]).unsqueeze(-1)

        back_logits = torch.cat([back_logits_pos, back_logits_neg], dim=1)
        back_logits /= self.T

        cycle_back_labels = torch.zeros(back_logits.shape[0], dtype=torch.long).cuda()
        return back_logits, cycle_back_labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output