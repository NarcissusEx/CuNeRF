import torch
import math

def cube_imp(weights, indices_rs, pts, cnts, is_train, n_samples, **kwargs):
    pts_rs = cube_sample_pdf(pts, cnts, weights[..., 1:-1], indices_rs, n_samples, is_train).detach()
    pts = torch.cat([pts, pts_rs + cnts[:, None]], 1)
    return {'pts' : pts, 'cnts' : cnts, 'dx' : kwargs['dx'], 'dy' : kwargs['dy'], 'dz' : kwargs['dz']}

def cube_sample_pdf(pts, cnts, weights, indices_rs, N_samples, is_train):
    centers = torch.gather(pts - cnts[:, None], -2, indices_rs[..., None].expand(*pts.shape))
    mids = .5 * (centers[:, 1:] + centers[:, :-1])
    rs_mid = torch.norm(mids, dim=-1)
    # xs_mid, ys_mid, zs_mid = mids[...,0], mids[...,1], mids[...,2]
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)
    # Take uniform samples
    if is_train:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])
    
    else:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(rs_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[...,1] - cdf_g[...,0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[...,0]) / denom

    rs = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])
    ts = torch.rand_like(rs) * math.pi
    ps = torch.rand_like(rs) * 2 * math.pi

    xs = rs * torch.sin(ts) * torch.cos(ps)
    ys = rs * torch.sin(ts) * torch.sin(ps)
    zs = rs * torch.cos(ts)
    samples = torch.cat([xs[...,None], ys[...,None], zs[...,None]], -1)

    return samples