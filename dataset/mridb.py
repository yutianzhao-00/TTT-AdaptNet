import torch
from torch.utils.data import Dataset
import scipy.io as sio

class MRIData(Dataset):
    def __init__(self, mode='train',
                 root_dir='/home/ytzhao/TTT/data/MRI_data.mat',
                 noise_data_path='/home/ytzhao/TTT/data/noise1s0.25.mat',
                 sample_index=None, tag=100):
        
        x = sio.loadmat(root_dir)['MRI_data']
        x = torch.from_numpy(x.transpose(2, 0, 1))
        # Load the noise data
        noise_data = sio.loadmat(noise_data_path)['N']
        
        if mode == 'train':
            self.x = x[:tag]
            self.noise_data = noise_data[:tag, :]
        elif mode == 'test':
            self.x = x[300:321, ...]
            self.noise_data = noise_data[300:321, :]
            
        if sample_index is not None:
            self.x = self.x[sample_index].unsqueeze(0)
            self.noise_data = self.noise_data[sample_index, :].unsqueeze(0)
            
    def __getitem__(self, index):
        x = self.x[index]
        n = self.noise_data[index, :]
        n = torch.from_numpy(n)
        return x, n
    
    def __len__(self):
        return len(self.x)
