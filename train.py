from __future__ import division
import torch
import torch.nn as nn
from torch.optim import *
import torch.nn.functional as F
import os, time
import argparse
import numpy as np
import h5py
import random
from tensorboardX import SummaryWriter
from model import Demoire
import Vgg19
from utils import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import Moire_dataset, Moire_dataset_test
from val_func import val

parser = argparse.ArgumentParser(description='Training demoire module')
parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--max_epoch', dest='max_epoch', type=int, default=100, help='maximum number of epoch')
parser.add_argument('--num_worker', dest='num_worker', type=int, default=0, help='number of workers when loading data')
parser.add_argument('--bestperformance', dest='bestperformance', type=float, default=0, help='best performance')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=256, help='patch_size')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--train_path', dest='train_path', default='/media/sdb1/MY/RAW_img_dm/data/trainset/patch_data6/', help='path of training data')
parser.add_argument('--test_path', dest='test_path', default='/media/sdb1/MY/RAW_img_dm/data/testset/testset6/', help='path of test data')
parser.add_argument('--save_path', dest='save_path', default='./Demoire_model/', help='storage path of model file')
parser.add_argument('--logs_path', dest='logs_path', default='./logs/demoire/', help='storage path of log file')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
torch.backends.cudnn.benchmark = True

save_dir = args.save_path
log_dir = args.logs_path

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

Moire_data_train = Moire_dataset(args.train_path)
train_dataloader = DataLoader(Moire_data_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_worker,
                              drop_last=True)
num_batch = len(train_dataloader)

Moire_data_test = Moire_dataset_test(args.test_path)
test_dataloader = DataLoader(Moire_data_test,
                             batch_size=1,
                             shuffle=True,
                             num_workers=args.num_worker,
                             drop_last=False)

    
print("[*] data load successfully...") 

model_isp = torch.load('ISP_model/isp_model_epoch20.pth').cuda()
for k,v in model_isp.named_parameters():
    v.requires_grad=False  

vgg19 = Vgg19.Vgg19(requires_grad=False).cuda()
model = Demoire().cuda()

initial_epoch = findLastCheckpoint(save_dir=save_dir) 
if initial_epoch > 0:
    print('[*] resuming by loading epoch %03d' % initial_epoch)
    model = torch.load(os.path.join(save_dir, 'demoire_model_epoch%d.pth' % initial_epoch))
    #initial_epoch += 1
    print("[*] Model restore success!")	
else:
    print("[*] Not find pretrained model!")

lr_list = [2e-4, 1e-4, 5e-5, 2.5e-5, 1.25e-5]
lr_index = 0
fails_threshold = 4
fail_count = 0
compile_flag = False
bestperformance = args.bestperformance    
opt = Adam(model.parameters(), lr=lr_list[lr_index])
total_step = 0
warm_step = 1 * num_batch

psnr_output = val(model, model_isp, test_dataloader, args)
print('Test set : \t' + 'PSNR = {:0.4f}'.format(psnr_output))
writer.add_scalar('psnr',psnr_output,initial_epoch)

for epoch in range(initial_epoch, args.max_epoch):
    cnt = 0
    #print(opt.state_dict()['param_groups'][0]['lr'])
    
    for ii, (batch_moire, batch_raw, batch_rgb, label) in enumerate(train_dataloader):

        total_step += 1
        
        # draw images
        #print(label)
        #plt.imshow(batch_rgb[0].permute(1,2,0).cpu().numpy())
        #plt.show()
        
        if epoch==0:
            opt.param_groups[0]['lr']=lr_list[lr_index] * total_step / warm_step
            writer.add_scalar('lr',opt.param_groups[0]['lr'],total_step)
        
        #loss
        L1_loss = nn.L1Loss()
        L2_loss = nn.MSELoss()

        model.zero_grad()
        dm_raw = model(batch_moire, flag = label)
        dm_rgb = model_isp(dm_raw)

        #loss3
        loss3_l1 = L1_loss(dm_rgb, batch_rgb)
        isp_vgg = vgg19(dm_rgb)
        with torch.no_grad():
            rgb_vgg = vgg19(batch_rgb.detach())
        loss3_vgg = L1_loss(isp_vgg, rgb_vgg)
        loss3_color = color_loss(dm_rgb, batch_rgb)
        loss3 = loss3_l1 + loss3_vgg + loss3_color
        if (cnt % 1000 == 0):
            writer.add_images('dm_gt', [torch.clamp(dm_rgb[0],0,1),batch_rgb[0]], total_step,dataformats='CHW')
            print("loss3=%.3f loss3_l1=%.3f loss3_vgg=%.3f loss3_color=%.3f" % (
                loss3.data, loss3_l1.data, loss3_vgg.data,loss3_color.data))


        if label == 1:            
            loss1_l1 = L1_loss(dm_raw, batch_raw)
            loss1_color = color_loss(dm_raw, batch_raw) 
            loss1_dm = loss1_l1 + loss1_color
            loss1_sum = loss1_dm + loss3
            loss1_sum.backward()
            opt.step()           
            cnt += 1
            
            if (cnt % 1000 ==0):
                writer.add_scalars('loss1_dm', {'loss1_dm':loss1_dm.item(),'loss3':loss3.item(),'loss1_sum':loss1_sum.item()}, total_step)
                print("lr:%f epoch:%d iter_sum%d iter%d loss1_dm=%.3f loss3=%.3f loss1_sum=%.3f" % (opt.state_dict()['param_groups'][0]['lr'], epoch+1, num_batch, cnt, loss1_dm.data, loss3.data, loss1_sum.data))

        elif label == 0:           
            loss2_l1 = L1_loss(dm_raw, batch_raw)
            loss2_dm = loss2_l1
            loss2_sum = loss2_dm + loss3
            loss2_sum.backward()
            opt.step()       
            cnt += 1
            
            if (cnt % 1000 == 0):
                writer.add_scalars('loss2_dm', {'loss2_dm':loss1_dm.item(),'loss3':loss3.item(),'loss2_sum':loss2_sum.item()}, total_step) 
                print("lr:%f epoch:%d iter_sum%d iter%d loss2_dm=%.3f loss3=%.3f loss2_sum=%.3f" % (opt.state_dict()['param_groups'][0]['lr'], epoch+1, num_batch, cnt, loss2_dm.data, loss3.data, loss2_sum.data))


    epoch_save = epoch + 1
    print('[*] Saving model...')	
    torch.save(model, os.path.join(save_dir, 'demoire_model_epoch%d.pth' % epoch_save))
    writer.add_scalar('lr',opt.param_groups[0]['lr'],total_step)
    psnr_output = val(model, model_isp, test_dataloader, args)
    print('Test set : \t' + 'PSNR = {:0.4f}'.format(psnr_output))
    writer.add_scalar('psnr',psnr_output,epoch+1)
    writer.close()
    
    if psnr_output > bestperformance  :  # bestperformance  
        if psnr_output - bestperformance < 1e-3:
            fail_count += 1
        else:
            fail_count = 0
        bestperformance = psnr_output
        torch.save(model, os.path.join(save_dir, 'Bestperformance_demoire_model_epoch%d_psnr%.3f.pth' % (epoch_save,bestperformance)))
    else:
        fail_count += 1
        
    if fail_count >= 4:
        if lr_index < (len(lr_list)-1):
            fail_count = 0
            lr_index += 1
            compile_flag = True
        else:
            exit()        
    if compile_flag:
        opt.param_groups[0]['lr']=lr_list[lr_index]
        compile_flag = False
    
    
#plt.plot(range(epoch),lr_list,color = 'r')
print('[*] Finish training.')
