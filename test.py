from __future__ import division
import os, scipy.io
import torch
import torch.nn as nn
import numpy as np
import glob
import cv2
import argparse
from PIL import Image
from utils import *
from thop import profile
import time
from torch.utils.data import DataLoader
from dataset import Moire_dataset, Moire_dataset_test
from skimage.metrics import peak_signal_noise_ratio


parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./Demoire_model/demoire_model_epoch43.pth', help='the model file to load')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=1, help='gpu id')
parser.add_argument('--num_worker', dest='num_worker', type=int, default=0, help='number of workers when loading data')
parser.add_argument('--test_path', dest='test_path', default='./data/testset/', help='path of test data')
parser.add_argument('--save_test_dir', dest='save_test_dir', default='./out/', help='storage path of output data')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

if not os.path.exists(args.save_test_dir):
    os.makedirs(args.save_test_dir)


Moire_data_test = Moire_dataset_test(args.test_path)
test_dataloader = DataLoader(Moire_data_test,
                             batch_size=1,
                             shuffle=True,
                             num_workers=args.num_worker,
                             drop_last=False)


model = torch.load(args.checkpoint_dir).cuda()
model_isp = torch.load('ISP_model/isp_model_epoch20.pth').cuda()
model_c = torch.load('classify_model/res_epoch20.pth').cuda()

model.eval()
model_isp.eval()
model_c.eval()	
psnr = []
time1 = []
time2 = []
time3 = []

with torch.no_grad():
    for ii ,(moire_img, c_img, gt_img, num_img) in enumerate(test_dataloader):      
        y_pred = model_c(c_img)
        _, pred = torch.max(y_pred.data, 1)
        dm_raw = model(moire_img, flag = pred)
        dm_rgb = model_isp(dm_raw)

        ##calculate psnr and save
        psnr_output = peak_signal_noise_ratio(torch.clamp(dm_rgb,0,1).cpu().numpy(), gt_img.cpu().numpy())
        psnr.append(psnr_output)
        print("num_img = "+str(num_img)+" PSNR = "+str(psnr_output)+" pred = "+str(pred))
        
        psnr_txt="num_img = "+str(num_img)+" PSNR = "+str(psnr_output)+" pred = "+str(pred)
        with open(args.save_test_dir+'psnr.txt','a') as psnr_file:   
            psnr_file.write(psnr_txt)    
            psnr_file.write('\n') 

        with open(args.save_test_dir+'psnr_woidx.txt','a') as psn_woidx_file:   
            psn_woidx_file.write(str(psnr_output))    
            psn_woidx_file.write('\n')

        with open(args.save_test_dir+'psnr_idx.txt','a') as psn_woidx_file:   
            psn_woidx_file.write(str(psnr_output))    
            psn_woidx_file.write('\n') 

        # ##save image			
        dm_rgb = dm_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()
        dm_rgb = np.clip(255 * dm_rgb, 0, 255).astype('uint8')
        save_images(os.path.join(args.save_test_dir,str(num_img[0])+'_dm.png'), dm_rgb)
        
    psnr_mean = np.mean(psnr)
    print("[*] the average of PSNR is "+str(psnr_mean))  
    with open(args.save_test_dir+'psnr.txt','a') as psnr_file: 
        psnr_file.write("Average PSNR = "+str(psnr_mean))          
    print("[*] Finish testing.")

ave_t1 = np.mean(time1)
ave_t2 = np.mean(time2)
ave_t3 = np.mean(time3)
print('AVERAGE c TIME = ')
print(ave_t1)
print('AVERAGE dm TIME = ')
print(ave_t2)
print('AVERAGE isp TIME = ')
print(ave_t3)
  
