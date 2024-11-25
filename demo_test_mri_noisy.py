import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import math
from torch.nn import init
import copy
import cv2
from models.unrolling_tf import ISTANetplus
from torch.utils.data import Dataset, DataLoader
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
import types
from utils import *
import numpy as np
from dataset.mridb import MRIData
from physics.mri import FFT_Mask_ForBack

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser(description='Test-MRI')

parser.add_argument('--layer_num', type=int, default=12, help='phase number of ISTA-Net-plus')
## add the mask type of the data
parser.add_argument('--mask_type', type=str, default='2r', help='mask type for the MRI sampling pattern')
parser.add_argument('--ratio', type=float, default=0.25, help='sampling ratio for the MRI mask')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--pre_model_dir', type=str, default='/home/ytzhao/TTT/pth/2r_0.25_mri_noisy.pkl', help='pretrained model directory')

args = parser.parse_args()
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
dtype = torch.FloatTensor
layer_num = args.layer_num



Phi_data_Name = f'/home/ytzhao/TTT/masks/mask{args.mask_type}{args.ratio:.2f}.mat'
print(Phi_data_Name)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['k']
mask = torch.tensor(mask_matrix).type(dtype)
mask = torch.unsqueeze(mask, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)
print(mask.shape)

def fft2(data):
    data = torch.fft.ifftshift(data)
    data = torch.fft.fft2(data)
    data = torch.fft.fftshift(data)
    return data

def ifft2(data):
    # input.shape: [..., h, w], input.dtype: complex
    # output.shape: [..., h, w], output.dtype: complex
    data = torch.fft.ifftshift(data)
    data = torch.fft.ifft2(data)
    data = torch.fft.fftshift(data)
    return data

def cal_ssim(a, b, multichannel=False):
    # a: prediction
    # b: ground-truth
    a = np.float64(a)
    b = np.float64(b)
    ssim_score = ssim(b[0], a[0])

    return ssim_score



physics = FFT_Mask_ForBack()

test_loader = torch.utils.data.DataLoader(dataset=MRIData(mode='test'),
                                   batch_size=1, shuffle=False)


#--------------------------------------Loading the model -------------------------------------------
model = ISTANetplus(layer_num)
model = model.to(device)
model.eval()

## loading checkpoints
ckpt = torch.load(args.pre_model_dir, map_location=device)
#model.load_state_dict(ckpt)
#Load pre-trained model with epoch number
state_dict = model.state_dict()
for k1, k2 in zip(state_dict.keys(), ckpt.keys()):
    state_dict[k1] = ckpt[k2]
model.load_state_dict(state_dict)
model = model.to(device)

psnr_avg = 0
ssim_avg = 0
total_time = 0.0

for i, (x,n) in enumerate(test_loader):
#for batch in test_loader:
    x = x.type(torch.FloatTensor).to(device)
    img_np = x.cpu().detach().numpy()
    print(img_np.shape)
    img_np = img_np.reshape(1, 192, 160)
    x = x.view(1, 1, 192, 160)
    
    n = n.type(torch.FloatTensor).to(device)
    n = n.to(torch.complex64).to(device)
    n = n.view(1, 1, 192, 160)
    y = physics.A(x, mask, sigma=0.1, add_noise=False)
    y = y + n
    
    PhiTb = physics.A_dagger(y)

#    x_output, _ = model(PhiTb, mask, cond, spatial_size)
    x_output = model(PhiTb, mask)
    BB = x_output[0,:,:,:].cpu()
    B = BB.detach().numpy()
    
    psnr = compare_psnr(np.clip(B,0,1),img_np)
    print("psnr:", psnr)
    psnr_avg +=psnr
    ssim_index = cal_ssim(np.clip(B, 0, 1), img_np)
    print(ssim_index)
    ssim_avg += ssim_index  

psnr = psnr_avg / len(test_loader)
print("The average reconstruction psnr is:", psnr)   
ssim = ssim_avg / len(test_loader)
print("The average reconstruction ssim is:", ssim)  