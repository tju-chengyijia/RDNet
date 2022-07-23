import argparse
import os
import random
import sys
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from torchnet import meter
from skimage.metrics import peak_signal_noise_ratio
import torchvision



def val(model, model_isp, test_dataloader, args):
   
    model_c = torch.load('classify_model/res_epoch20.pth').cuda()
    
    model.eval()
    model_isp.eval()
    model_c.eval()
    	
    psnr = []

    for ii ,(moire_img, c_img, gt_img, num_img) in enumerate(test_dataloader):
    
        with torch.no_grad():
            y_pred = model_c(c_img)
            _, pred = torch.max(y_pred.data, 1)
            dm_raw = model(moire_img, flag = pred)
            dm_rgb = model_isp(dm_raw)

            psnr_output = peak_signal_noise_ratio(torch.clamp(dm_rgb,0,1).cpu().numpy(), gt_img.cpu().numpy())
            psnr.append(psnr_output)
    psnr_mean = np.mean(psnr)

    return psnr_mean