import torch
torch.autograd.set_detect_anomaly(True)

import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F 
import scipy.io as sio
import datetime
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
import types
import cv2
import math
from models.unrolling import ISTANetplus
from utils import get_Optimizer
from scipy.io import savemat
import glob
## set the device
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
import numpy as np
from dataset.mridb import MRIData
import time
from physics.mri import FFT_Mask_ForBack

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#### set the parser argument
parser = ArgumentParser(description='ISTANet-Plus')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=700, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=12, help='phase number of ISTA-Net-plus')
## add the mask type of the data
parser.add_argument('--mask_type', type=str, default='2r', help='mask type for the MRI sampling pattern')
parser.add_argument('--ratio', type=float, default=0.25, help='sampling ratio for the MRI mask')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--optimizer', default='Adam', type=str, help='Currently only support SGD, Adam and Adamw')
parser.add_argument('--model_dir', type=str, default='models', help='trained or pre-trained model directory')
## add the momentum to the SGD optimizer
## about the parameters of my optimizers
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD')
parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1 of Adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of Adam')
parser.add_argument('--eps', default=1e-8, type=float, help='eps of Adam')
parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay of optimizer')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate '
                                       '(default 5e-4 for CT)',
                    dest='lr')

parser.add_argument('--batch_size', type=int, default=2)

parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
layer_num = args.layer_num

## set the batch_size
dtype = torch.FloatTensor

### import the mask
Phi_data_Name = f'/home/ytzhao/TTT/masks/mask{args.mask_type}{args.ratio:.2f}.mat'
print(Phi_data_Name)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['k']
mask = torch.tensor(mask_matrix).type(dtype)
mask = torch.unsqueeze(mask, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)
print(mask.shape)

### Noise is fixed
#Noise_name = f'/home/huanzheng/video/DSO_delivery/CV_Project/TTT/noise_mat/bsd_noise{args.mask_type}{args.ratio:.2f}_new.mat'
Noise_name = f'/home/ytzhao/TTT/noise/noise{args.mask_type}{args.ratio:.2f}.mat'
print(Noise_name)
N = sio.loadmat('%s' % ( Noise_name))
Noise = N['N']
print(Noise.shape)


## define the psnr function
def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


data_loader = torch.utils.data.DataLoader(dataset=MRIData(mode='train'),
                                   batch_size=args.batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=MRIData(mode='test'),
                                   batch_size=1, shuffle=False)
train_dataset = MRIData(mode='train')
val_dataset = MRIData(mode='test')
print(f'Length of train_loader: {len(train_dataset)}, Length of val_loader: {len(val_dataset)}')

#--------------------------------------Loading the model -------------------------------------------
model = ISTANetplus(layer_num)
model = model.to(device)

##### define the physics from the FFT_Mask_ForBack
physics = FFT_Mask_ForBack()
### optimizer, SGD or Adam
optimizer = get_Optimizer(args, model)
model_dir = "./%s/MRI_Sup_layer%d_mask%s_ratio%.4f_lr%.4f_optim%s_fixnoise"% (args.model_dir, args.layer_num, args.mask_type, args.ratio, args.lr, args.optimizer)
log_file_name = "./%s/Log_MRI_Sup_layer%d_mask%s_ratio%.4f_lr%.4f_optim%s_fixnoise.txt" % (args.log_dir,  args.layer_num, args.mask_type, args.ratio, args.lr, args.optimizer)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

with open(f"{args.log_dir}/args.txt", "w") as f:
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

start_time = time.time()
best_status = {'PSNR': 0, 'SSIM': 0}

best_checkpoint = None
## optimizer, we have a choice from the utils.py
for epoch_i in range(args.start_epoch, args.end_epoch):
    ## seth three losses for my training
    loss_all_list, loss_sup_list, loss_aux_list = [], [], []
    psnr_list, ssim_list = [], []
    print('Current epoch: {}'.format(epoch_i))
    for i, (batch_x, batch_n) in enumerate(data_loader):
        ## turn the mask
        batch_x = batch_x.type(torch.FloatTensor).to(device)
        batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])
        batch_n = batch_n.to(torch.complex64).to(device)
        batch_n = batch_n.view(batch_n.shape[0], 1, batch_n.shape[1], batch_n.shape[2])
        ## just do the supervised learning
        y = physics.A(batch_x, mask)
        y = y + batch_n
#        print(batch_n)
        PhiTb = physics.A_dagger(y)
    
        x_output = model(PhiTb, mask)
        loss_sup = torch.mean(torch.pow(batch_x - x_output, 2))
    
        loss_all = loss_sup

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        ## just save the loss lists
        loss_all_list.append(loss_all.item())
        loss_sup_list.append(loss_sup.item())
    
    ## for the scheduler, we temporality just do not use it
    avg_loss_all = np.array(loss_all_list).mean()
    avg_loss_sup = np.array(loss_sup_list).mean()
    output_data = "[%02d/%02d] Total Loss: %.6f, Supervised Loss: %.6f\n" % (epoch_i, args.end_epoch, avg_loss_all, avg_loss_sup)
    print(output_data)
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()


    if epoch_i % save_interval == 0:
        os.makedirs("./%s/" % (model_dir), exist_ok=True)
        torch.save(model.state_dict(),
                   "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters


#### save the best checkpoint
torch.save(best_checkpoint,
                   os.path.join(model_dir, 'best.pth'))  # save only the parameters


total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
sio.savemat(os.path.join(model_dir, 'TrainingLog.mat'), {'loss': loss_all_list})
savemat(os.path.join(model_dir, 'PSNR_SSIM_Curves.mat'), {'PSNR': psnr_list, 'SSIM': ssim_list})
