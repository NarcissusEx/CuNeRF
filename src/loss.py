import torch
import torch.nn.functional as F

def Adaptive_MSE_LOSS(rgb, rgb0, target):
    weights = torch.sqrt(torch.abs(rgb - target)).detach()
    loss = F.mse_loss(rgb, target)
    loss0 = torch.mean(weights * torch.square(rgb0 - target))
    return loss + loss0