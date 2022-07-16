# ------------------------------------------------------------------------------
# Copyright (c) by contributors 
# Licensed under the MIT License.
# Written by Haiping Wu
# ------------------------------------------------------------------------------
import os
import glob
import random

from PIL import Image
import numpy as np
from skimage import color
import scipy.io as sio
import tqdm
import pickle
from collections import OrderedDict, defaultdict

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import multiprocessing
from torch.multiprocessing import Pool
import cv2
import math


import ucl.loader


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageNetVal(torch.utils.data.Dataset):
    # the class name and idx do not necessarily follows the standard one
    def __init__(self, root, class_names, class_to_idx, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        # super(ImageNetVal, self).__init__(root, transform=transform,
        #                                     target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        samples = self._make_dataset(class_names, class_to_idx)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 samples"))

        self.loader = default_loader

        self.classes = list(class_names)
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _make_dataset(self, class_names, class_to_idx):
        meta_file = os.path.join(self.root, 'meta_clsloc.mat')
        meta = sio.loadmat(meta_file, squeeze_me=True)['synsets']
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}

        annot_file = os.path.join(self.root, 'ILSVRC2012_validation_ground_truth.txt')
        with open(annot_file, 'r') as f:
            val_idcs = f.readlines()
        val_idcs = [int(val_idx) for val_idx in val_idcs]
        pattern = os.path.join(self.root, 'ILSVRC2012_val_%08d.JPEG')
        samples = []
        for i in range(50000):
            # filter class names needed
            gt_wnid = idx_to_wnid[val_idcs[i]]
            if gt_wnid in class_names:
                samples.append([pattern%(i+1), class_to_idx[gt_wnid]])
        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)



class ImageFolderInstance(torch.utils.data.Dataset):
    """Folder datasets which returns the index of the image as well
    """
    def __init__(self, root, transform=None, target_transform=None):
        self.dataset = datasets.ImageFolder(root, transform)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

SAMPLE_NAME = "AA/AA2pFq9pFTA_000001.jpg"
LEN_SAMPLE_NAME = len(SAMPLE_NAME)
LEN_VID_NAME = len("AA2pFq9pFTA")
LEN_NUM_NAME = len("000001")
LEN_CLIP_NAME = len("0001")


class GbVideoDataset(torch.utils.data.Dataset):
    """Folder datasets which returns the index of the image as well
    """
    def __init__(self, root, transform=None, target_transform=None, \
                    data_split='train', return_all_video_frames=False, \
                    num_of_sampled_frames=4, return_same_frame_indicator=False, \
                    pos_dists=[3] , neg_dists=[0.2, 0.4], num_neg_samples=2, \
                    return_neg_frame=True):

        self.root = root
        self.transform = transform
        self.two_crops_transform = ucl.loader.TwoCropsTransform(transform)
        #self.crops_transform = cycle_contrast.loader.CropsTransform(transform)
        self.target_transform = target_transform ## None
        self.data_split = data_split
        self.return_all_video_frames = return_all_video_frames ## True
        self.num_of_sampled_frames = num_of_sampled_frames ##1
        self.return_same_frame_indicator = return_same_frame_indicator ## False
        self.return_neg_frame = return_neg_frame  ## True
        self.pos_dists = pos_dists
        self.neg_dists = neg_dists
        self.num_neg_samples = num_neg_samples

        self._get_annotations()

        self.loader = default_loader

    @staticmethod
    def get_video_name(name):
        return name.split("/")[-2]

    @staticmethod
    def get_frame_id(name):
        return int(name.split("/")[-1][:-4])

    def get_image_paths(self):
        print('path ############', self.data_basepath)
        return sorted(list(tqdm.tqdm(glob.iglob(os.path.join(self.data_basepath, "*/*.jpg")))))

    def get_image_name(self, key: str, ind: int):
        return os.path.join(self.data_split_path, key,  "%05d.jpg" % ind)

    def video_id_frame_id_split(self, name):
        return self.get_video_name(name), self.get_frame_id(name)

    def _get_single_frame(self, path_key, ind):
        return self.transform(self.loader(self.get_image_name(path_key, ind)))

    def _get_annotations(self):
        self.data_basepath = self.root
        self.data_split_path = os.path.join(self.data_basepath)
        pickle_path = os.path.join(self.data_basepath, self.data_split+ "_names.pkl")
        
        if not os.path.exists(pickle_path):
            print('creat new cache')
            images = self.get_image_paths()
            path_info = OrderedDict()
            video_names = sorted([self.video_id_frame_id_split(name) for name in images])
            for vid_id, ind in video_names:
                if vid_id not in path_info:
                    path_info[vid_id] = []
                path_info[vid_id].append(ind)
            path_info = sorted([(key, val) for key, val in path_info.items()])
            os.makedirs(self.data_split_path, exist_ok=True)
            pickle.dump(path_info, open(pickle_path, "wb"))
        self.path_info = pickle.load(open(pickle_path, "rb"))
        num_frames = int(np.sum([len(p_info[1]) for p_info in self.path_info]))
        print("Num for %s videos %d frames %d" % (self.data_split, len(self.path_info), num_frames))

    def __getitem__(self, index):
        path_key, frame_ids = self.path_info[index]
        target = index
        ## index is the video number
        ## ind is the frame number
        num_frames = len(frame_ids)
        pos_dists = self.pos_dists
        neg_dists = self.neg_dists
        low = int(math.ceil(num_frames*0.2))
        high = int(math.ceil(num_frames*0.5))
        anchor_frame = np.random.randint(low, high)

        pos_indices = [elem for elem in range(max(0, anchor_frame-pos_dists[0]), \
                        min(num_frames, anchor_frame+pos_dists[0]+1))]
        pos_indices.remove(anchor_frame)

        left_low = max(0, min(anchor_frame-int(neg_dists[1]*num_frames), \
                              anchor_frame-pos_dists[0]-3))
        left_high = max(0, min(anchor_frame-int(neg_dists[0]*num_frames), \
                               anchor_frame-pos_dists[0]-1))
        right_low = min(num_frames, max(anchor_frame+int(neg_dists[0]*num_frames), \
                                        anchor_frame+pos_dists[0]+1))
        right_high = min(num_frames, max(anchor_frame+int(math.ceil(neg_dists[1]*num_frames)), \
                                         anchor_frame+pos_dists[0]+3))
        neg_indices = [elem for elem in range(left_low, left_high)] \
                        + [elem for elem in range(right_low, right_high)]

        pos_ind = np.random.choice(pos_indices)
        if self.return_neg_frame:
            #ind = anchor_frame + np.random.randint(self.pos_dists[0], self.pos_dists[1]+1)
            #neg_ = np.random.randint(int(self.neg_dists[0]*num_frames), \
            #                            int(self.neg_dists[1]*num_frames)+1, size=self.num_neg_samples)
            #neg_inds = [(anchor_frame+d)%num_frames for d in neg_]
            neg_inds = np.random.choice(neg_indices, size=self.num_neg_samples)
        
        ## ind gets a random frame out of allthe frames

        q_img = self.loader(self.get_image_name(path_key, anchor_frame))
        k_img = self.loader(self.get_image_name(path_key, pos_ind))
        n_imgs = [self.loader(self.get_image_name(path_key, idx)) for idx in neg_inds]

        ## Loading the image at the chosen random index 
        if self.transform is not None:
            sample = self.two_crops_transform(q_img) # sample is [q, k, [n1, n2]]
            #q_frame = sample[1] #self.transform(q_img) ## for the neighbor set
            k_frame = self.transform(k_img)
            n_frames = [self.transform(n) for n in n_imgs]
            video_frames = [k_frame] + n_frames
            video_frames = torch.stack(video_frames, dim=0)
            ## sample --> two augs of q
            ## video_frames --> augs of [k, n1, n2]
            return sample, target, index, video_frames
        
        elif self.return_all_video_frames:
            video_frames = [self.loader(self.get_image_name(path_key, _ind))
                            for _ind in frame_ids if _ind != ind]
            if self.transform is not None:
                video_frames = [self.transform(video_frame) for video_frame in video_frames]
            video_frames = torch.stack([self.transform(image), *video_frames], dim=0)
            return sample, target, index, video_frames
        
        else:
            return sample, target, index

    def __len__(self):
        # path_info: dictionary; video_name, frame_ids in video
        return len(self.path_info)


def parse_file(dataset_adr, categories):
    dataset = []
    with open(dataset_adr) as f:
        for line in f:
            line = line[:-1].split("/")
            category = "/".join(line[2:-1])
            file_name = "/".join(line[2:])
            if not category in categories:
                continue
            dataset.append([file_name, category])
    return dataset


def get_class_names(path):
    classes = []
    with open(path) as f:
        for line in f:
            categ = "/".join(line[:-1].split("/")[2:])
            classes.append(categ)
    class_dic = {classes[i]: i for i in range(len(classes))}
    return class_dic

