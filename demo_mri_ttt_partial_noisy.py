import torch
import numpy as np
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
import random
import copy
import cv2
from skimage.io import imsave
from torch.utils.data import Dataset, DataLoader
from dataset.mridb import MRIData
from copy import deepcopy
from models.unrolling_scale_new import ISTANet_S_new, ISTANet_S_new_x
#from models.unrolling_conv import ISTANetplus_Conv
import torchvision.utils as vutils
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
import argparse
import types
import time
from skimage import img_as_float
import learn2learn as l2l
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='MRI test.')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--ckp',
                    default='/home/ytzhao/TTT/pth/2r_0.25_mri_noisy.pkl',
                    type=str, metavar='PATH',
                    help='path to checkpoint of a trained model')
parser.add_argument('--mask_type', type=str, default='2r', help='mask type for the MRI sampling pattern')
parser.add_argument('--ratio', type=float, default=0.25, help='sampling ratio for the MRI mask')
parser.add_argument('--model-name', default='Unrolling', type=str, help="name of the trained model (dafault: 'EI')")
parser.add_argument('--layer_num', type=int, default=12, help='phase number of ISTA-Net-plus')
parser.add_argument('--image_dir', default='Image_noisy/admmnet', type=str, metavar='PATH',
                    help='path to checkpoint of a trained model')

args = parser.parse_args()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
seed = 32
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)
"""
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
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

### Noise is fixed
Noise_name = f'/home/ytzhao/TTT/mri_noise/noise{args.mask_type}{args.ratio:.2f}.mat'
print(Noise_name)
N = sio.loadmat('%s' % ( Noise_name))
Noise = N['N']
## For the test-time-adaptation
def add_noise(model, nlevel):
    for n in [x for x in model.parameters() if len(x.size()) == 4]:
        noise = torch.randn(n.size()) * nlevel
        noise = noise.type(dtype).to(device)
        n.data = n.data + noise


def cal_psnr(a, b):
    # a: prediction
    # b: ground-truth
    alpha = np.sqrt(a.shape[-1] * a.shape[-2])
    return 20 * torch.log10(alpha * torch.norm(b, float('inf')) / torch.norm(b - a, 2)).detach().cpu().numpy()


def cal_ssim(a, b, multichannel=False):
    # a: prediction
    # b: ground-truth
    b = img_as_float(b.squeeze().detach().cpu().numpy())
    a = img_as_float(a.squeeze().detach().cpu().numpy())
    return ssim(b, a, data_range=a.max() - a.min(), multichannel=multichannel)


### forward operation
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


class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, full_mask, sigma=0.1, add_noise=True):
        full_mask = full_mask[..., 0]
        x_in_k_space = fft2(x)
        if add_noise:
            noise = torch.randn_like(x) * sigma * torch.max(x)
            noise = fft2(noise)
            ## noise version of k_space
            x_in_k_space = x_in_k_space + noise
        ## add mask to the data
        masked_x_in_kspace = x_in_k_space * full_mask.view(1, 1, *(full_mask.shape))
        masked_x = torch.real(ifft2(masked_x_in_kspace))
        return masked_x

    def A(self, x, full_mask, sigma=0.1, add_noise=False):
        full_mask = full_mask[..., 0]
        x_in_k_space = fft2(x)
        if add_noise:
            noise = torch.randn_like(x) * sigma * torch.max(x)
            noise = fft2(noise)
            ## noise version of k_space
            x_in_k_space = x_in_k_space + noise
        masked_x_in_k_space = x_in_k_space * full_mask.view(1, 1, *(full_mask.shape))
        return masked_x_in_k_space

    def A_dagger(self, y):
        x = torch.real(ifft2(y))
        return x


physics = FFT_Mask_ForBack()

LR = 1e-3
num_iter = 200
burnin_iter = 0
out_avg = None

model_conv = ISTANet_S_new(layer_num)
model_conv = model_conv.to(device)
print("Number of parameters in model_conv:", sum(p.numel() for p in model_conv.parameters() if p.requires_grad))

## load the checkpoint
ckpt = torch.load(args.ckp, map_location=device)

# Remove entries from ckpt that are not in model_conv
ckpt = {k: v for k, v in ckpt.items() if k in model_conv.state_dict()}

# Load the state dict into model_conv
model_conv.load_state_dict(ckpt, strict=False)


def freeze_all_parameters_except_conv_new(model):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    for k, v in model.named_parameters():
        if 'fcs.12' in k:
            v.requires_grad = True
        if 'conv_add' in k:
            v.requires_grad = True


freeze_all_parameters_except_conv_new(model_conv)
print("Number of updated parameters in model:", sum(p.numel() for p in model_conv.parameters() if p.requires_grad))


            
data_loader = torch.utils.data.DataLoader(dataset=MRIData(mode='test'),
                                          batch_size=1, shuffle=False)

psnr_a = np.zeros((len(data_loader),))
ssim_a = np.zeros((len(data_loader),))
psnr_avg = 0.0
ssim_avg = 0.0

psnr_trace = []
ssim_trace = []
psnr_avg_trace = []
ssim_avg_trace = []
loss_trace = []
trace = {}

total_time = 0.0

for idx, (x, n) in enumerate(data_loader):
    MODEL_PATH = "./%s/mask%s_ratio%f_noisy" % (
        args.image_dir, str(args.mask_type),args.ratio)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if len(x.shape) == 3:
        x = x.unsqueeze(1)
    x = x.type(torch.FloatTensor).to(device)
    if len(n.shape) == 3:
        n = n.unsqueeze(1)
    n = n.to(torch.complex64).to(device)
    ##copy the model for partial update
    model_adapt =  deepcopy(model_conv)

    ## model_adapt is for adaption update, we freeze part of its parameters
    #freeze_all_parameters_except_conv_new(model_adapt)
    print("Number of updated parameters in model_adapt:", sum(p.numel() for p in model_adapt.parameters() if p.requires_grad))
    sgld_mean_each = 0
    sgld_mean_tmp = 0


    def closure():
        global iter, num_iter, psnr_total, sgld_mean_each, sgld_mean_tmp
        y = physics.A(x, mask, sigma=0.1, add_noise=True)
        
        gamma = (torch.FloatTensor(y.size()).normal_(mean=0, std=10 / 255).cuda())
        PhiTb = physics.A_dagger(y + gamma)
        
        ## get the output of model_adapt
        x_hat, features = model_adapt(PhiTb, mask)
        y_hat = physics.A(x_hat, mask, sigma=0.1, add_noise=False)
        ### define the range loss
        loss_mc = torch.mean(torch.pow(torch.abs(y_hat - (y - gamma)), 2))
    #    loss_mc = torch.mean(torch.pow(torch.real(y_hat - (y - gamma)), 2))
        
        # first get x_label from pre-trained model, add the poisson noise
    #    alpha = torch.poisson(x_label / 1).cuda()

        with torch.no_grad():
            x_label, features = model_conv(PhiTb, mask)
            alpha = 1*torch.poisson(torch.relu(x_label) / 1)
            ## calculate (I-A^+A)(x_label-alpha)
            x_label_null = (x_label-alpha) - physics.A_dagger(physics.A((x_label-alpha), mask))
        # add noise to the x_label
        alpha = 1*torch.poisson(torch.relu(x_label) / 1)
   
        PhiTb_plus = physics.A_dagger(physics.A((x_label+alpha), mask, sigma=0.1, add_noise=True))
        ## put the PhiTb_plus to the model_adapt
        x_hat_plus, _ = model_adapt(PhiTb_plus, mask)
        ## calculate (I-A^+A)x_hat_plus
        x_hat_null = x_hat_plus - physics.A_dagger(physics.A(x_hat_plus, mask))
        loss_nc = torch.mean(
            torch.pow(x_hat_null - x_label_null, 2))


        print("loss_mc, loss_nc", loss_mc, 25*loss_nc)
        total_loss = loss_mc  + 25 * loss_nc
        loss_data = total_loss.item()
        total_loss.backward()

        ## define the iteration
        if iter > burnin_iter:
            sgld_mean_each += x_hat
            sgld_mean_tmp = sgld_mean_each / (iter - burnin_iter)
        else:
            sgld_mean_tmp = x_hat

        iter += 1

        psrn_gt = cal_psnr(torch.clip(x_hat, 0, 1), x)
        ssim_gt = cal_ssim(torch.clip(x_hat, 0, 1), x)
        psrn_gt_avg = psrn_gt
        if iter > burnin_iter:
            psrn_gt_avg = cal_psnr(x, torch.clip(sgld_mean_tmp, 0, 1))
            ssim_gt_avg = cal_ssim(torch.clip(sgld_mean_tmp, 0, 1), x)

        return total_loss


    iter = 0

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_adapt.parameters()), lr=LR)

    print('Starting optimization with ASGLD')

    start_time = time.time()  # Record start time
    # define the noise level
    nlevel_ = 1e-10

    for j in range(num_iter):
        optimizer.zero_grad()
        loss = closure()
        optimizer.step()
        add_noise(model_adapt, nlevel_)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_time += elapsed_time  # Add the time for this image to the total
    print(f"Time taken for image {idx} during test time adaptation: {elapsed_time:.2f} seconds")

    psnr_single = cal_psnr(x, torch.clip(sgld_mean_tmp, 0, 1))
    ssim_single = cal_ssim(x, torch.clip(sgld_mean_tmp, 0, 1))
    image_data = sgld_mean_tmp.squeeze().cpu().detach().numpy()
    
    image_data = np.clip(image_data, 0, 1)
    image_data = (image_data * 255).astype(np.uint8)

    # Save the grayscale image
#    imsave(os.path.join(MODEL_PATH, 'img_%d_' % (idx) + 'psnr—' + str(psnr_single) + '.png'), image_data)


    psnr_avg += psnr_single
    ssim_avg += ssim_single
    psnr_a[idx] = psnr_single
    ssim_a[idx] = ssim_single

psnr_avg /= len(data_loader)
ssim_avg /=len(data_loader)
print("average psnr, ssim over the image set:", psnr_avg, ssim_avg)
avg_time_per_image = total_time / len(data_loader)
print(f"Average Time taken for each image during test time adaptation: {avg_time_per_image:.2f} seconds")
