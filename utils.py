import numpy as np 
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

## specify the optimizer
def get_Optimizer(opt,model):
    if opt.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': opt.lr}], lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': opt.lr}], lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    else:
        raise Exception('Unexpected Optimizer of {}'.format(opt.optimizer))
    return optimizer


def getPSNR(ref, recon):
    """
    measures PSNR between the reference and reconstructed images
    """
    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    psnr = 20 * np.log10(np.abs(ref.max())/(np.sqrt(mse) + 1e-10))

    return psnr

def fft(ispace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters:
    ----------
    ispace: coil images of size nrow x ncol x ncoil
    axes: The default is (0, 1).
    norm: The default is None.
    unitary_opt: The default is True.

    Returns:
    --------
    transform image space to k-space.
    """
    ### axes is (0, 1)
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ispace, axes=axes), axes=axes, norm=norm),axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * kspace.shape[axis]
        
        kspace = kspace / np.sqrt(fact)

    return kspace

def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace: image space of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    transform k-space image space.
    """
    ispace = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)
    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * ispace.shape[axis]
        ### is this a kind of normalization?
        ispace = ispace * np.sqrt(fact)

    return ispace


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor: It can be applied in image space or k-space.
    axes: The default is (0, 1, 2)
    keepdims: The default is True.

    Returns:
    tnesor: applies l2 norm
    """

    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)
    
    if not keepdims: return tensor.squeeze()

    return tensor

def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace: nrow x ncol x ncoil.
    axes: The default is (1, 2, 3)

    Returns
    -------
    the center of the k-space
    """
    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]

def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/ vector for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations
    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


def sense1(input_kspace, sens_maps, axes=(0, 1)):
    """
    Parameters:
    -----------
    input_kspace: nrow x ncol x ncoil
    sens_maps : nrow x ncol x ncoil

    axes : The default is (0,1).

    Returns:
    -------
    sense1 image
    """

    image_space = ifft(input_kspace, axes=axes, norm=None, unitary_opt=True)
    Eh_op = np.conj(sens_map) * image_space
    sense1_image = np.sum(Eh_op, axis=axes[-1] + 1)

    return sense1_image

def complex2real(input_data):
    """
    Parameters
    ----------
    input_data: rowxcol
    dtype: The default is np.float32

    Returns
    -------
    output: row x col x2
    """

    return np.stack((input_data.real, input_data.imag), dim=-1)

def real2complex(input_data):
    """
    Parameters
    ----------
    input_data: row x col x 2

    Returns
    -------
    output: row x col
    """

    return input_data[...,0] * input_data[...,1]



def psnr(gt, pred):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred, maxval=None):
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    ssim = structural_similarity(
            gt, pred, data_range=maxval
        )
    return ssim

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count