from torchmetrics.functional.image.ssim import multiscale_structural_similarity_index_measure
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
# import lpips
import torch

class Metrics:

    def __init__(self, net='vgg'):
        # self.lpips_loss_fn = lpips.LPIPS(net=net)
        pass

    def _ssim(self, x, y):
        '''
        Compute the Structural Similarity Index (SSIM) between two images
        '''
        return multiscale_structural_similarity_index_measure(x, y)

    def _psnr(self, x, y):
        '''
        Compute the Peak Signal to Noise Ratio (PSNR) between two images
        '''
        return peak_signal_noise_ratio(x, y)

    def _lpips(self, x, y):
        '''
        Compute the Learned Perceptual Image Patch Similarity (LPIPS) between two images
        '''
        return self.lpips_loss_fn(x, y)
    
    def evaluate(self, renders, target_imgs):
        '''
        Evaluate the renders against the target images
        '''
        ssim = self._ssim(renders, target_imgs)
        psnr = self._psnr(renders, target_imgs)
        lpips = 0
        
        return ssim, psnr, lpips