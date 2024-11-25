import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import os
from models.network_swinir import SwinIR
from models.unrolling_tf import BasicBlock_tf


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

    def A(self, x, full_mask):
        full_mask = full_mask[..., 0]
        x_in_k_space = fft2(x)
        masked_x_in_k_space = x_in_k_space * full_mask.view(1, 1, *(full_mask.shape))
        return masked_x_in_k_space

    def A_dagger(self, y):
        x = torch.real(ifft2(y))
        return x


class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        #    self.gamma1 = nn.Parameter(torch.ones(1))  # First scaling parameter
        #    self.gamma2 = nn.Parameter(torch.ones(1))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bias = nn.Parameter(torch.full([32], 0.01))
        self.conv4 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bias1 = nn.Parameter(torch.full([32], 0.01))
        self.conv5 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv6 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv7 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask, sigma=0.1, add_noise=False)
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x = F.conv2d(x_D, self.conv1, padding=1)
        #    x = self.gamma1 * x
        ## add the scale factor
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2, bias=self.bias, padding=1)
        x_forward = F.relu(x_forward)
        x_forward = F.conv2d(x_forward, self.conv3, bias=self.bias, padding=1)
        x_forward = F.relu(x_forward)
        x = F.conv2d(x_forward, self.conv4, bias=self.bias, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv5, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv6, padding=1)
        x = F.relu(x)
        ### this is just like the tail conv, which has no norm_layer
        x = F.conv2d(x, self.conv7, padding=1)
        x_G = F.conv2d(x, self.conv_G, padding=1)
        x_pred = x_input + x_G


        return x_pred


class BasicBlock_add(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_add, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        #    self.gamma1 = nn.Parameter(torch.ones(1))  # First scaling parameter
        #    self.gamma2 = nn.Parameter(torch.ones(1))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bias = nn.Parameter(torch.full([32], 0.01))
        self.conv4 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.bias1 = nn.Parameter(torch.full([32], 0.01))
        self.conv5 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv6 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv7 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        self.conv_add1 = nn.Parameter(torch.zeros(1, 1, 3, 3))
        self.conv_add2 = nn.Parameter(torch.zeros(1, 1, 3, 3))
#        self.conv_add = nn.Parameter(torch.zeros(1, 1, 1, 1))
        

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask, sigma=0.1, add_noise=False)
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x = F.conv2d(x_D, self.conv1, padding=1)
        #    x = self.gamma1 * x
        ## add the scale factor
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2, bias=self.bias, padding=1)
        x_forward = F.relu(x_forward)
        x_forward = F.conv2d(x_forward, self.conv3, bias=self.bias, padding=1)
        x_forward = F.relu(x_forward)
        x = F.conv2d(x_forward, self.conv4, bias=self.bias, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv5, padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv6, padding=1)
        x = F.relu(x)
        ### this is just like the tail conv, which has no norm_layer
        x = F.conv2d(x, self.conv7, padding=1)
        x_G = F.conv2d(x, self.conv_G, padding=1)
        x_pred = x_input + x_G
        ## add one small adaptnet to the original one
        x_pred = x_pred - self.lambda_step * fft_forback(x_pred, mask, sigma=0.1, add_noise=False)
        x_pred = x_pred + self.lambda_step * PhiTb
        # for 1x1 set to be 0
        x_pred_res = F.conv2d(x_pred, self.conv_add1, padding=1)
        x_pred_res = F.conv2d(x_pred_res, self.conv_add2, padding=1)
        x_pred = x_pred + x_pred_res


        return x_pred

"""
class ISTANetplus_S(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANetplus_S, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        ## import the forward function from physics
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):

        x = PhiTb
        for i in range(self.LayerNo):
            x = self.fcs[i](x, self.fft_forback, PhiTb, mask)

        x_final = x

        return x_final

"""
class BasicBlock_new_small(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_new_small, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.conv_G1 = nn.Parameter(torch.zeros(1, 1, 3, 3))
        self.conv_G2 = nn.Parameter(torch.zeros(1, 1, 3, 3))

        # Reduced convolution layers

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask, sigma=0.1, add_noise=False)
        x = x + self.lambda_step * PhiTb
        x_input = x
        x_G = F.conv2d(x_input, self.conv_G1, padding=1)
        x_G = F.conv2d(x_G, self.conv_G2, padding=1)
        x_pred = x_input + x_G

        return x_pred


class BasicBlock_new_small_x(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_new_small_x, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.conv_G1 = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 1, 3, 3)))
        self.conv_G2 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 16, 3, 3)))

        # Reduced convolution layers

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask, sigma=0.1, add_noise=False)
        x = x + self.lambda_step * PhiTb
        x_input = x
        x_G = F.conv2d(x_input, self.conv_G1, padding=1)
        x_G = F.conv2d(x_G, self.conv_G2, padding=1)
        x_pred = x_input + x_G

        return x_pred



class BasicBlock_new(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_new, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 1, 3, 3)))

        # Reduced convolution layers
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 16, 3, 3)))
        self.bias = nn.Parameter(torch.full([16], 0.01))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 16, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 16, 3, 3)))

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask, sigma=0.1, add_noise=False)
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x = F.conv2d(x_D, self.conv1, padding=1)
        # Add the scale factor
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2, bias=self.bias, padding=1)
        x_forward = F.relu(x_forward)
        x_G = F.conv2d(x_forward, self.conv_G, padding=1)
        x_pred = x_input + x_G

        return x_pred

class BasicBlock_new_wo(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_new_wo, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 1, 3, 3)))

        # Reduced convolution layers
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 16, 3, 3)))
        self.bias = nn.Parameter(torch.full([16], 0.01))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(16, 16, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 16, 3, 3)))

    def forward(self, x, fft_forback, PhiTb, mask):
        #x = x - self.lambda_step * fft_forback(x, mask, sigma=0.1, add_noise=False)
        #x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        x = F.conv2d(x_D, self.conv1, padding=1)
        # Add the scale factor
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2, bias=self.bias, padding=1)
        x_forward = F.relu(x_forward)
        x_G = F.conv2d(x_forward, self.conv_G, padding=1)
        x_pred = x_input + x_G

        return x_pred

class ISTANet_S(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANet_S, self).__init__()

        onelayer = []
        self.LayerNo = LayerNo

        # Import the forward function from physics
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):  # We subtract 1 because we want to add BasicBlock_tf as the last layer
            onelayer.append(BasicBlock())

        # Add BasicBlock_tf as the last layer
        onelayer.append(BasicBlock_new())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):
        x = PhiTb
        for i in range(self.LayerNo + 1):
            x = self.fcs[i](x, self.fft_forback, PhiTb, mask)

        x_final = x
        return x_final
class ISTANet_S_wo(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANet_S_wo, self).__init__()

        onelayer = []
        self.LayerNo = LayerNo

        # Import the forward function from physics
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):  # We subtract 1 because we want to add BasicBlock_tf as the last layer
            onelayer.append(BasicBlock_add())

        # Add BasicBlock_tf as the last layer
        onelayer.append(BasicBlock_new_wo())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):
        x = PhiTb
        for i in range(self.LayerNo + 1):
            x = self.fcs[i](x, self.fft_forback, PhiTb, mask)

        x_final = x
        return x_final

class ISTANet_S_new(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANet_S_new, self).__init__()

        onelayer = []
        self.LayerNo = LayerNo

        # Import the forward function from physics
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):  # We subtract 1 because we want to add BasicBlock_tf as the last layer
            onelayer.append(BasicBlock_add())

#        onelayer.append(BasicBlock_new())
        onelayer.append(BasicBlock_new_small())


        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):
        
        x = PhiTb
        x_pred_list = []
        for i in range(self.LayerNo+1):
            x = self.fcs[i](x, self.fft_forback, PhiTb, mask)
            x_pred_list.append(x)
        x_final = x

        return x_final, x_pred_list

class ISTANet_S_new_x(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANet_S_new_x, self).__init__()

        onelayer = []
        self.LayerNo = LayerNo

        # Import the forward function from physics
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):  # We subtract 1 because we want to add BasicBlock_tf as the last layer
            onelayer.append(BasicBlock_add())

#        onelayer.append(BasicBlock_new())
        onelayer.append(BasicBlock_new_small_x())


        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):
        x = PhiTb
        x_pred_list = []
        for i in range(self.LayerNo):
            x = self.fcs[i](x, self.fft_forback, PhiTb, mask)
            x_pred_list.append(x)

        x_final = x
        return x_final, x_pred_list
