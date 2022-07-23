from __future__ import division
import os, scipy.io
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import cv2
from scipy.stats import poisson
from skimage.measure import compare_psnr,compare_ssim
import time
from PIL import Image
from math import log10

def pack_gbrg_raw(raw):
    #pack GBRG Bayer raw to 4 channels
    black_level = 240
    white_level = 2**12-1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[0:H:2, 0:W:2, :]), axis=2)
    return out

def depack_gbrg_raw(raw):
    H = raw.shape[1]
    W = raw.shape[2]
    output = np.zeros((H*2,W*2))
    for i in range(H):
        for j in range(W):
            output[2*i,2*j]=raw[0,i,j,3]
            output[2*i,2*j+1]=raw[0,i,j,2]
            output[2*i+1,2*j]=raw[0,i,j,0]
            output[2*i+1,2*j+1]=raw[0,i,j,1]
    return output

def pack_rggb_raw(raw):
    #pack RGGB Bayer raw to 4 channels
    black_level = 240
    white_level = 2**12-1
    im = raw.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level-black_level)

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def generate_noisy_raw(gt_raw, a, b):
    """
    a: sigma_s^2
    b: sigma_r^2
    """
    gaussian_noise_var = b
    poisson_noisy_img = poisson((gt_raw-240)/a).rvs()*a
    gaussian_noise = np.sqrt(gaussian_noise_var)*np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    noisy_img = poisson_noisy_img + gaussian_noise + 240
    noisy_img = np.minimum(np.maximum(noisy_img,0), 2**12-1)
    
    return noisy_img

def generate_name(number):
    name = list('000000_raw.tiff')
    num_str = str(number)
    for i in range(len(num_str)):
        name[5-i] = num_str[-(i+1)]
    name = ''.join(name)
    return name

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()

def reduce_mean_with_weight(im1, im2, noisy_level_data):
    result = torch.abs(im1 - im2) * noisy_level_data * 0.1
    return result.mean()

def preprocess(raw):
    input_full = raw.transpose((0, 3, 1, 2))
    input_full = torch.from_numpy(input_full)
    input_full = input_full.cuda()
    return input_full

def postprocess(output):
    output = output.cpu()
    output = output.detach().numpy().astype(np.float32)
    output = np.transpose(output, (0, 2, 3, 1))
    output = np.clip(output,0,1)
    return output

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'demoire_model_epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*demoire_model_epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def bayer_preserving_augmentation(raw, aug_mode):
    if aug_mode == 0:  # horizontal flip
        aug_raw = np.flip(raw, axis=1)[:,1:-1]
    elif aug_mode == 1: # vertical flip
        aug_raw = np.flip(raw, axis=0)[1:-1,:]
    else:  # random transpose
        aug_raw = np.transpose(raw, (1, 0))
    return aug_raw

def pack_gbrg_raw_for_compute_ssim(raw):

    im = raw.astype(np.float32)
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[0:H:2, 0:W:2, :]), axis=2)
    return out

def compute_ssim_for_packed_raw(raw1, raw2):
    raw1_pack = pack_gbrg_raw_for_compute_ssim(raw1)
    raw2_pack = pack_gbrg_raw_for_compute_ssim(raw2)
    test_raw_ssim = 0
    for i in range(4):
        test_raw_ssim += compare_ssim(raw1_pack[:,:,i], raw2_pack[:,:,i], data_range=1.0)

    return test_raw_ssim/4

def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        # im = Image.open(filelist).convert('L')
        im = Image.open(filelist)
        return np.array(im).reshape(1, im.size[1], im.size[0], 3)
    data = []
    for file in filelist:
        # im = Image.open(file).convert('L')
        im = Image.open(file)
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 3))
    return data

def load_images_raw(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        # im = Image.open(filelist).convert('L')
        im = Image.open(filelist)
        return np.array(im).reshape(1, im.size[1], im.size[0])
    data = []
    for file in filelist:
        # im = Image.open(file).convert('L')
        im = Image.open(file)
        data.append(np.array(im).reshape(1, im.size[1], im.size[0]))
    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    # im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im = Image.fromarray(cat_image.astype('uint8'))
    im.save(filepath, 'png')

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False

def color_loss(x1, x2):
	x1_norm = torch.sqrt(torch.sum(torch.pow(x1, 2)) + 1e-8)
	x2_norm = torch.sqrt(torch.sum(torch.pow(x2, 2)) + 1e-8)
	# 内积
	x1_x2 = torch.sum(torch.mul(x1, x2))
	cosin = 1 - x1_x2 / (x1_norm * x2_norm)
	return cosin

def cal_psnr(img1, img2):
    #_, _, h, w = img1.size()
    mse = torch.sum((img1/1.0 - img2/1.0) ** 2) / img1.numel()
    psnr = 10 * log10(1/mse)
    return psnr

def make_labels(dir):
    D = {}
    f = open(dir)
    line = f.readline()
    while line:
        key, v = line.split('_')
        v2 = re.findall("\d+",v)[0]
        #print(v2)
        #v2 = v2.strip('\n')
        #v2 = int(v2)
        D[key] = v2
        line = f.readline()
    f.close()
    return D

def pack_raw(im):
    
    im = np.expand_dims(im, axis=3)
    img_shape = im.shape
    H = img_shape[1]
    W = img_shape[2]   
    ## R G G B
    out = np.concatenate((im[:,0:H:2,0:W:2,:], 
                       im[:,0:H:2,1:W:2,:],
                       im[:,1:H:2,0:W:2,:],
                       im[:,1:H:2,1:W:2,:]), axis=3)
    return out

def unpack_raw(im):
    b,h,w,chan = im.shape 
    H, W = h*2, w*2
    img2 = np.zeros((b,H,W))
    img2[:,0:H:2,0:W:2]=im[:,:,:,0] #r
    img2[:,0:H:2,1:W:2]=im[:,:,:,1] #g1
    img2[:,1:H:2,0:W:2]=im[:,:,:,2] #g2
    img2[:,1:H:2,1:W:2]=im[:,:,:,3] #b
    return img2

def normalize_numpy(data):
    m = 0.5
    std = 0.5
    #return [(float(i) - m) / (mx - mn) for i in data]
    return (data - m)/std

def rggb_pack_aveg(im):
    im = np.expand_dims(im, axis=3)
    img_shape = im.shape
    H = img_shape[1]
    W = img_shape[2]   
    ## r g2 g1 b
    out = np.concatenate((im[:,0:H:2,0:W:2,:],  #r
                       (im[:,0:H:2,1:W:2,:]+im[:,1:H:2,0:W:2,:])/2,     #(g1+g2)/2
                       im[:,1:H:2,1:W:2,:]), axis=3)  #b
    return out


# def loss_mae(y_true, y_pred):
#     return torch.abs(y_true - y_pred)

def a_sobel_loss(y_true, y_pred):
    mae = torch.nn.L1Loss()
    def sobel_process(y_true, y_pred, sobel_func):
        sobel_pred = sobel_func(y_pred) * 0.25
        sobel_true = sobel_func(y_true) * 0.25
        dx_loss = mae(sobel_pred[:, :, :, :, 0], sobel_true[:, :, :, :, 0])
        dy_loss = mae(sobel_pred[:, :, :, :, 1], sobel_true[:, :, :, :, 1])
        dr_loss = mae(sobel_pred[:, :, :, :, 2], sobel_true[:, :, :, :, 2])
        dl_loss = mae(sobel_pred[:, :, :, :, 3], sobel_true[:, :, :, :, 3])
        return dx_loss + dy_loss + dr_loss + dl_loss
    sobel_d1_loss = sobel_process(y_true, y_pred, sobel_edges)
    #sobel_d2_loss = sobel_process(y_true, y_pred, sobel_edges_d2)
    #sobel_d3_loss = sobel_process(y_true, y_pred, sobel_edges_d3)
    # sobel_d4_loss = sobel_process(y_true, y_pred, sobel_edges_d4)
    return sobel_d1_loss# + sobel_d2_loss + sobel_d3_loss + sobel_d4_loss

def sobel_edges(image):
    """Returns a tensor holding Sobel edge maps.

    Arguments:
    image: Image tensor with shape [batch_size, d, h, w] and type float32 or
    float64.  The image(s) must be 2x2 or larger.

    Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
    """

    # Define vertical and horizontal Sobel filters.
    torch_image_shape = image.shape
    torch_kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],#dy
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],#dx
             [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],#
             [[0, -1, -2], [1, 0, -1], [2, 1, 0]]]#
    num_kernels = len(torch_kernels)
    torch_kernels = np.asarray(torch_kernels)
    torch_kernels = np.expand_dims(torch_kernels, 0)
    torch_kernels = torch.from_numpy(torch_kernels)
    torch_kernels = torch_kernels.repeat(torch_image_shape[1], 1, 1, 1)
    torch_padded = torch.nn.functional.pad(image, [1, 1, 1, 1], mode='reflect')
    torch_padded_shape = torch_padded.shape
    input_torch = torch.Tensor(torch_padded_shape[0], torch_padded_shape[1] * num_kernels, torch_padded_shape[2],
                               torch_padded_shape[3])
    for i in range(torch_padded_shape[1]):
        for j in range(num_kernels):
            input_torch[:, i * num_kernels + j, :, :] = torch_padded[:, i, :, :]
    shape_torch_kernels = torch_kernels.shape
    torch_kernels = torch_kernels.reshape(shape_torch_kernels[0] * shape_torch_kernels[1], 1, shape_torch_kernels[2],
                                          shape_torch_kernels[3])
    torch_output = torch.nn.functional.conv2d(input=input_torch, weight=torch_kernels.type(torch.FloatTensor), stride=1,
                                              groups=input_torch.shape[1])
    torch_output = torch_output.permute(0, 2, 3, 1).reshape(torch_image_shape[0], torch_image_shape[2],
                                                            torch_image_shape[3], torch_image_shape[1], num_kernels)

    return torch_output

def sobel_edges_d2(image):
    """Returns a tensor holding Sobel edge maps.

    Arguments:
    image: Image tensor with shape [batch_size, d, h, w] and type float32 or
    float64.  The image(s) must be 2x2 or larger.

    Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
    """

    # Define vertical and horizontal Sobel filters.
    torch_image_shape = image.shape
    torch_kernels = [[[-1, 0, -2, 0, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 2, 0, 1]],
             [[-1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [-2, 0, 0, 0, 2], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 1]],
             [[-2, 0, -1, 0, 0], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 1, 0, 2]],
             [[0, 0, -1, 0, -2], [0, 0, 0, 0, 0], [1, 0, 0, 0, -1], [0, 0, 0, 0, 0], [2, 0, 1, 0, 0]]]
    num_kernels = len(torch_kernels)
    torch_kernels = np.asarray(torch_kernels)
    torch_kernels = np.expand_dims(torch_kernels, 0)
    torch_kernels = torch.from_numpy(torch_kernels)
    torch_kernels = torch_kernels.repeat(torch_image_shape[1], 1, 1, 1)
    torch_padded = torch.nn.functional.pad(image, [2, 2, 2, 2], mode='reflect')
    torch_padded_shape = torch_padded.shape
    input_torch = torch.Tensor(torch_padded_shape[0], torch_padded_shape[1] * num_kernels, torch_padded_shape[2],
                               torch_padded_shape[3])
    for i in range(torch_padded_shape[1]):
        for j in range(num_kernels):
            input_torch[:, i * num_kernels + j, :, :] = torch_padded[:, i, :, :]
    shape_torch_kernels = torch_kernels.shape
    torch_kernels = torch_kernels.reshape(shape_torch_kernels[0] * shape_torch_kernels[1], 1, shape_torch_kernels[2],
                                          shape_torch_kernels[3])
    torch_output = torch.nn.functional.conv2d(input=input_torch, weight=torch_kernels.type(torch.FloatTensor), stride=1,
                                              groups=input_torch.shape[1])
    torch_output = torch_output.permute(0, 2, 3, 1).reshape(torch_image_shape[0], torch_image_shape[2],
                                                            torch_image_shape[3], torch_image_shape[1], num_kernels)

    return torch_output

def sobel_edges_d3(image):
    """Returns a tensor holding Sobel edge maps.

    Arguments:
    image: Image tensor with shape [batch_size, d, h, w] and type float32 or
    float64.  The image(s) must be 2x2 or larger.

    Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
    """

    # Define vertical and horizontal Sobel filters.
    torch_image_shape = image.shape
    torch_kernels = [[[-1, 0, 0, -2, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 2, 0, 0, 1]],
             [[-1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [-2, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 1]],
             [[-2, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 2]],
             [[0, 0, 0, -1, 0, 0, -2], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 1, 0, 0, 0]]]
    num_kernels = len(torch_kernels)
    torch_kernels = np.asarray(torch_kernels)
    torch_kernels = np.expand_dims(torch_kernels, 0)
    torch_kernels = torch.from_numpy(torch_kernels)
    torch_kernels = torch_kernels.repeat(torch_image_shape[1], 1, 1, 1)
    torch_padded = torch.nn.functional.pad(image, [3, 3, 3, 3], mode='reflect')
    torch_padded_shape = torch_padded.shape
    input_torch = torch.Tensor(torch_padded_shape[0], torch_padded_shape[1] * num_kernels, torch_padded_shape[2],
                               torch_padded_shape[3])
    for i in range(torch_padded_shape[1]):
        for j in range(num_kernels):
            input_torch[:, i * num_kernels + j, :, :] = torch_padded[:, i, :, :]
    shape_torch_kernels = torch_kernels.shape
    torch_kernels = torch_kernels.reshape(shape_torch_kernels[0] * shape_torch_kernels[1], 1, shape_torch_kernels[2],
                                          shape_torch_kernels[3])
    torch_output = torch.nn.functional.conv2d(input=input_torch, weight=torch_kernels.type(torch.FloatTensor), stride=1,
                                              groups=input_torch.shape[1])
    torch_output = torch_output.permute(0, 2, 3, 1).reshape(torch_image_shape[0], torch_image_shape[2],
                                                            torch_image_shape[3], torch_image_shape[1], num_kernels)

    return torch_output

def gradient_1order(x,h_x=None,w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    return xgrad