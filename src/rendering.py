import torch
import torch.nn.functional as F

import math

def cube_rendering(raw, pts, cnts, dx, dy, dz):
    norm = torch.sqrt(torch.square(dx) + torch.square(dy) + torch.square(dz))
    raw2beta = lambda raw, dists, rs, act_fn=F.relu : -act_fn(raw) * dists * torch.square(rs) * 4 * math.pi
    raw2alpha = lambda raw, dists, rs, act_fn=F.relu : (1.-torch.exp(-act_fn(raw) * dists)) * torch.square(rs) * 4 * math.pi 
    rs = torch.norm(pts - cnts[:, None], dim=-1)
    sorted_rs, indices_rs = torch.sort(rs)
    dists = sorted_rs[...,1:] - sorted_rs[...,:-1]
    dists = torch.cat([dists, dists[...,-1:]], -1)  # [N_rays, N_samples]
    rgb = torch.gather(torch.sigmoid(raw[...,:-1]), -2, indices_rs[..., None].expand(raw[...,:-1].shape))
    sorted_raw = torch.gather(raw[...,-1], -1, indices_rs)
    beta = raw2beta(sorted_raw, dists, sorted_rs / norm)
    alpha = raw2alpha(sorted_raw, dists, sorted_rs / norm)  # [N_rays, N_samples]
    weights = alpha * torch.exp(torch.cumsum(torch.cat([torch.zeros(alpha.shape[0], 1), beta], -1), -1)[:, :-1])
    rgb_map = torch.sum(weights * rgb.squeeze(), -1)

    return {'rgb' : rgb_map, 'weights' : weights, 'indices_rs' : indices_rs}