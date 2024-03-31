import torch
import math

def cube_sampling(batch, depths, n_samples, is_train, R):
    n_cnts = batch.shape[0]
    (cnts, LR, TB), (near, far) = torch.split(batch, [3, 2, 2], dim=-1), depths

    left, right = torch.split(LR, [1, 1], dim=-1)
    top, bottom = torch.split(TB, [1, 1], dim=-1)
    steps = int(math.pow(n_samples, 1./3) + 1)
    t_vals = torch.cat([v[...,None] for v in torch.meshgrid(torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps), torch.linspace(0., 1., steps=steps))], -1)
    t_vals = t_vals[1:, 1:, 1:].contiguous().view(-1, 3)

    x_l, x_r = left.expand([n_cnts, n_samples]), right.expand([n_cnts, n_samples])
    y_l, y_r = top.expand([n_cnts, n_samples]), bottom.expand([n_cnts, n_samples])
    z_l = torch.full_like(x_l, 1.).view(-1, n_samples) * near[:, None]
    z_r = torch.full_like(x_r, 1.).view(-1, n_samples) * far[:, None]

    if is_train:
        x_vals = x_l + t_vals[:, 0] * (x_r - x_l) * torch.rand(n_cnts, n_samples)
        y_vals = y_l + t_vals[:, 1] * (y_r - y_l) * torch.rand(n_cnts, n_samples)
        z_vals = z_l + t_vals[:, 2] * (z_r - z_l) * torch.rand(n_cnts, n_samples)
            
    else:
        x_vals = x_l + t_vals[:, 0] * (x_r - x_l)
        y_vals = y_l + t_vals[:, 1] * (y_r - y_l)
        z_vals = z_l + t_vals[:, 2] * (z_r - z_l)

    pts = torch.cat([x_vals[..., None], y_vals[..., None], z_vals[..., None]], -1)
    if R is not None:
        pts, cnts = pts @ R, cnts @ R

    return {'pts' : pts, 'cnts' : cnts, 'dx' : (x_r - x_l).mean() / 2, 'dy' : (y_r - y_l).mean() / 2, 'dz' : (z_r - z_l).mean() / 2}