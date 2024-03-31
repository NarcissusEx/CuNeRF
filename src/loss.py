import torch
import torch.nn.functional as F

# def EXP_MSE_LOSS(rgb, rgb0, target, b=0.5):
#     weight = torch.square(rgb - target).detach()
#     loss = torch.mean(torch.square(rgb - target))
#     loss0 = torch.mean(weight * torch.square(rgb0 - target))
#     return loss + loss0

# def FOCAL_MSE_LOSS(rgb, rgb0, target, a=2, b=0.5):
#     weights = torch.pow(torch.abs(rgb - target), a).detach()
#     loss = weights * torch.square(rgb - target)
#     loss0 = weights * torch.square(rgb0 - target)
#     return torch.mean(loss + b * loss0)

def IREM_LOSS(rgb, rgb0, target):
    loss = F.mse_loss(rgb, target)
    return loss

def MSE_LOSS(rgb, rgb0, target):
    loss = F.mse_loss(rgb, target)
    loss0 = F.mse_loss(rgb0, target)
    return loss + loss0

def Adaptive_MSE_LOSS(rgb, rgb0, target):
    weights = torch.sqrt(torch.abs(rgb - target)).detach()
    loss = F.mse_loss(rgb, target)
    loss0 = torch.mean(weights * torch.square(rgb0 - target))
    return loss + loss0