import torch
from skimage.metrics import peak_signal_noise_ratio
dtype = torch.cuda.FloatTensor
def psnr3d(x,y): 
    ps = 0
    for i in range(x.shape[2]):
        ps += peak_signal_noise_ratio(x[:,:,i], y[:,:,i])
    return ps/x.shape[2]

