import os, cv2
import random

# import h5py
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import imageio
from torch.utils.data import Dataset
from torchvision import transforms
import  torchvision.transforms.functional as TF
from pathlib import Path
import torch.nn.functional as F
from utils import *

def normalize_numpy(data):
    m = 0.5
    std = 0.5
    return (data - m)/std

def rggb_pack_aveg(im):
    im = (im*255).astype('uint8')
    im = im/255.
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]   
    ## r g2 g1 b
    out = np.concatenate((im[0:H:2,0:W:2,:],  #r
                          (im[0:H:2,1:W:2,:]+im[1:H:2,0:W:2,:])/2,     #(g1+g2)/2
                          im[1:H:2,1:W:2,:]), axis=2)  #b
    return out


def pack_rggb_raw(im):
    #pack RGGB Bayer raw to 4 channels

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :]), axis=2)
    return out


def default_loader(path,type_img):
    if type_img == 'm_raw':
        raw_img = np.load(path)
        raw_data = raw_img['patch_data']
        #norm_factor = raw_img['white_level'] - raw_img['black_level_per_channel'][0]
        #img = (raw_data- raw_img['black_level_per_channel'][0])/norm_factor
        img = raw_data/4095.0
    elif type_img == 'rgb':
        img = np.array(Image.open(path).convert('RGB'))/255.0
    elif type_img == 'gt_raw':
        raw_img = np.load(path)
        raw_data = raw_img['patch_data']
        img = raw_data/4095.0

    return img


class Moire_dataset(Dataset):
    def __init__(self, root, loader = default_loader):
        moire_raw_root = os.path.join(root, 'moire_RAW_npz')
        gt_raw_root = os.path.join(root, 'gt_RAW_npz')
        gt_rgb_root = os.path.join(root, 'gt_RGB')
        label_data_root = 'label_res.txt'

        image_names = os.listdir(moire_raw_root)
        image_names = ["_".join(i.split("_")[:-1]) for i in image_names]

        self.moire_raw_images = [os.path.join(moire_raw_root, x + '_m.npz') for x in image_names]
        self.gt_raw_images = [os.path.join(gt_raw_root, x + '_gt.npz') for x in image_names]
        self.gt_rgb_images = [os.path.join(gt_rgb_root, x + '_gt.png') for x in image_names]
        self.labels_dict = make_labels(label_data_root)
        
        self.loader = loader
        self.labels = image_names

    def __getitem__(self, index):
        moire_raw_path = self.moire_raw_images[index]
        gt_raw_path = self.gt_raw_images[index]
        gt_rgb_path = self.gt_rgb_images[index]
        moire_raw_img = self.loader(moire_raw_path, 'm_raw')
        gt_raw_img = self.loader(gt_raw_path, 'gt_raw')
        gt_rgb_img = self.loader(gt_rgb_path, 'rgb')

        moire_raw_img = pack_rggb_raw(moire_raw_img)
        gt_raw_img = pack_rggb_raw(gt_raw_img)
        
        if np.random.randint(2,size=1)[0] == 1:  # random flip 
            moire_raw_img = np.flip(moire_raw_img, axis=1)
            gt_raw_img = np.flip(gt_raw_img, axis=1)
            gt_rgb_img = np.flip(gt_rgb_img, axis=1)
        if np.random.randint(2,size=1)[0] == 1: 
            moire_raw_img = np.flip(moire_raw_img, axis=0)
            gt_raw_img = np.flip(gt_raw_img, axis=0)
            gt_rgb_img = np.flip(gt_rgb_img, axis=0)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose 
            moire_raw_img = np.transpose(moire_raw_img, (1,0,2))
            gt_raw_img = np.transpose(gt_raw_img, (1,0,2))
            gt_rgb_img = np.transpose(gt_rgb_img, (1,0,2))	
        
        moire_raw_img = torch.from_numpy(moire_raw_img.copy())
        gt_raw_img = torch.from_numpy(gt_raw_img.copy())
        gt_rgb_img = torch.from_numpy(gt_rgb_img.copy())
        
        moire_raw_img = moire_raw_img.type(torch.FloatTensor).permute(2,0,1).cuda() 
        gt_raw_img = gt_raw_img.type(torch.FloatTensor).permute(2,0,1).cuda()
        gt_rgb_img = gt_rgb_img.type(torch.FloatTensor).permute(2,0,1).cuda() 
        label = int(self.labels_dict[self.labels[index]])
        
        return moire_raw_img, gt_raw_img, gt_rgb_img, label

    def __len__(self):
        return len(self.moire_raw_images)
        

class Moire_dataset_test(Dataset):
    def __init__(self, root, loader = default_loader):
        moire_raw_root = os.path.join(root, 'moire_RAW_npz')
        gt_rgb_root = os.path.join(root, 'gt_RGB')

        image_names = os.listdir(moire_raw_root)
        image_names = ["_".join(i.split("_")[:-1]) for i in image_names]

        self.moire_raw_images = [os.path.join(moire_raw_root, x + '_m.npz') for x in image_names]
        self.gt_rgb_images = [os.path.join(gt_rgb_root, x + '_gt.png') for x in image_names]
        
        self.loader = loader
        self.labels = image_names

    def __getitem__(self, index):
        moire_raw_path = self.moire_raw_images[index]
        gt_rgb_path = self.gt_rgb_images[index]
        moire_raw_img = self.loader(moire_raw_path, 'm_raw')
        gt_rgb_img = self.loader(gt_rgb_path, 'rgb')

        class_img = rggb_pack_aveg(moire_raw_img)
        class_img = normalize_numpy(class_img)
        moire_raw_img = pack_rggb_raw(moire_raw_img)	
        
        moire_raw_img = torch.from_numpy(moire_raw_img)
        class_img = torch.from_numpy(class_img)
        gt_rgb_img = torch.from_numpy(gt_rgb_img)
        
        moire_raw_img = moire_raw_img.type(torch.FloatTensor).permute(2,0,1).cuda() 
        class_img = class_img.type(torch.FloatTensor).permute(2,0,1).cuda()
        gt_rgb_img = gt_rgb_img.type(torch.FloatTensor).permute(2,0,1).cuda() 
        num_img = self.labels[index]
        
        return moire_raw_img, class_img, gt_rgb_img, num_img#, label

    def __len__(self):
        return len(self.moire_raw_images)


def default_loader_full(path,type_img):
    if type_img == 'm_raw':
        raw_img = np.load(path)
        raw_data = raw_img['data']
        img = raw_data/4095.0
    elif type_img == 'rgb':
        img = np.array(Image.open(path).convert('RGB'))/255.0
    elif type_img == 'gt_raw':
        raw_img = np.load(path)
        raw_data = raw_img['data']
        img = raw_data/4095.0

    return img

class Moire_dataset_test_full(Dataset):
    def __init__(self, root, loader=default_loader_full):
        moire_raw_root = os.path.join(root, 'moire_RAW_npz')

        image_names = os.listdir(moire_raw_root)
        image_names = ["_".join(i.split("_")[:-1]) for i in image_names]

        self.moire_raw_images = [os.path.join(moire_raw_root, x + '_ds.npz') for x in image_names]
        self.loader = loader
        self.labels = image_names

    def __getitem__(self, index):
        moire_raw_path = self.moire_raw_images[index]
        moire_raw_img = self.loader(moire_raw_path, 'm_raw')
        H, M = moire_raw_img.shape
        x = (M - 1792) / 2 - 2
        y = (H - 1280) / 2 - 2
        moire_raw_img = moire_raw_img[int(y):int(y) + 1280, int(x):int(x) + 1792]

        class_img = rggb_pack_aveg(moire_raw_img)
        class_img = normalize_numpy(class_img)
        moire_raw_img = pack_rggb_raw(moire_raw_img)

        moire_raw_img = torch.from_numpy(moire_raw_img)
        class_img = torch.from_numpy(class_img)

        moire_raw_img = moire_raw_img.type(torch.FloatTensor).permute(2, 0, 1).cuda()
        class_img = class_img.type(torch.FloatTensor).permute(2, 0, 1).cuda()
        num_img = self.labels[index]

        return moire_raw_img, class_img, num_img  # , label

    def __len__(self):
        return len(self.moire_raw_images)