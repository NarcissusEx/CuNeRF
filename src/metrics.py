from functools import partial, reduce
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

class Metrics:
    def __init__(self, metrics):
        self.metrics = metrics
        if 'lpips' in metrics.keys():
            self._lpips = lpips.LPIPS(net='alex').cuda()

    def compare(self, key, old, new):
        return True if (self.metrics[key] and old < new) or (not self.metrics[key] and old > new) else False

    def getDict(self):
        return self.metrics
    
    def evaluation(self, pds, gts):
        scores = {k : getattr(self, k.lower())(pds, gts) for k, v in filter(lambda x : x[0] != 'avg', self.metrics.items())}
        scores = {**scores, 'avg' : self.avg(scores)}
        return scores

    def psnr(self, pds, gts):
        return peak_signal_noise_ratio(gts, pds, data_range=1)
    
    # NWH
    def ssim(self, pds, gts):
        print (pds.shape, gts.shape)
        return structural_similarity(gts, pds, win_size=11, data_range=0, channel_axis=0)
    
    def lpips(self, pds, gts):
        with torch.no_grad():
            x = torch.from_numpy(pds).cuda()[..., None].expand([*pds.shape, 3]).permute(0, 3, 1, 2)
            y = torch.from_numpy(gts).cuda()[..., None].expand([*gts.shape, 3]).permute(0, 3, 1, 2)

            return self._lpips(x.float(), y.float()).mean().item()
        
    def avg(self, scores):
        slist = []
        if 'psnr' in scores.keys():
            slist.append(np.power(10, -scores['psnr'] / 10))
        
        if 'ssim' in scores.keys():
            slist.append(np.sqrt(1 - scores['ssim']))

        if 'lpips' in scores.keys():
            slist.append(scores['lpips'])

        return np.power(reduce(lambda x1, x2 : x1 * x2, slist), 1./len(slist))

    def mse(self, pds, gts):
        return ((pds - gts) ** 2).mean()

    def loss2psnr(self, value):
        return 10 * np.log10(1. / value)
