import torch

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
        full_mask = full_mask[...,0]
        x_in_k_space = fft2(x)
        if add_noise:
            noise = torch.randn_like(x) * sigma * torch.max(x)
            noise = fft2(noise)
            ## noise version of k_space
            x_in_k_space = x_in_k_space + noise
        ## add mask to the data
        masked_x_in_kspace = x_in_k_space * full_mask.view(1, 1, *(full_mask.shape))
        masked_x = torch.real(ifft2(masked_x_in_kspace))
#        masked_x = torch.abs(ifft2(masked_x_in_kspace))
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
#        x = torch.abs(ifft2(y))
        return x