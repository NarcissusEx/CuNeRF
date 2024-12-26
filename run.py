##########################################################################
# The Run Script of CuNeRF
# @Author: Zixuan Chen
# @Email:  chenzx3@mail2.sysu.edu.cn
##########################################################################

import argparse
from tqdm import tqdm
from functools import reduce

import math
import random
import numpy as np

import torch
import torch.nn.functional as F

from src import Cfg, utils

def argParse():
    parser = argparse.ArgumentParser(description='MISR3D')
    # basic settings
    parser.add_argument('expname', type=str)
    parser.add_argument('--cfg', default='configs/example.yaml')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--mode', choices=['train', 'eval', 'test'], default='train')
    parser.add_argument('--file', type=str)
    parser.add_argument('--max_iter', type=int)
    parser.add_argument('--eval_iter', type=int)
    parser.add_argument('--N_eval', type=int) # number of eval imgs
    parser.add_argument('--save_map', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_type', default='current')

    # test options
    parser.add_argument('--zpos', nargs='+', type=float) # pos of the medical volume
    parser.add_argument('--scales', nargs='+', type=float) # rendering scale
    parser.add_argument('--angles', nargs='+', type=int) # init rendering angle
    parser.add_argument('--axis', nargs='+', type=int) # rotation axis, e.g., [1,1,0]
    parser.add_argument('--cam_scale', nargs='+', type=float) # camera size = img size * cam_scale
    parser.add_argument('--is_details', action='store_true') # save the details of rendering
    parser.add_argument('--is_gif', action='store_true') # save gif
    parser.add_argument('--is_video', action='store_true') # save video
    parser.add_argument('--asteps', type=int)

    # other options
    parser.add_argument('--modality', choices=['FLAIR', 'T1w', 't1gd', 'T2w']) # modality for mris
    parser.add_argument('--workers', type=int) # workers for saving imgs
    args = parser.parse_args()

    return args


def train(cfg):
    while cfg.i_step <= cfg.max_iter:
        for batch in cfg.trainloader:
            cfg.optim.zero_grad()
            gts, coords, depths = batch
            gts, coords = gts.squeeze(0), coords.squeeze(0)
            rgb, rgb0 = cfg.Render(coords, depths, is_train=True)
            loss = cfg.loss_fn(rgb, rgb0, gts)
            loss.backward()
            # cfg.Update_grad()
            cfg.optim.step()
            with torch.no_grad():
                cfg.Update(loss, rgb.cpu().numpy(), gts.cpu().numpy())
                if cfg.i_step % cfg.log_iter == 0: cfg.Log()
                if cfg.i_step % cfg.save_iter == 0: cfg.Save()
                if cfg.i_step % cfg.eval_iter == 0: globals()['eval'](cfg)

            # update step and pbar
            if (cfg.i_step > cfg.max_iter) or (cfg.resume and cfg.i_step == cfg.max_iter): return
            cfg.i_step += 1
            cfg.pbar.update(1)

def eval(cfg):
    N, W, H, S = cfg.evalset.__len__(), cfg.evalset.W, cfg.evalset.H, cfg.bs_eval
    pds = np.zeros((N, W * H))
    dataloader = tqdm(cfg.evalloader)
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            dataloader.set_description(f'[EVAL] : {idx}')
            coords, depths = batch
            coords = coords.squeeze(0)
            for cidx in range(math.ceil(W * H / S)):
                select_coords = coords[list(range(S * cidx, min(S * (cidx + 1), len(coords))))]
                rgb, _ = cfg.Render(select_coords, depths, is_train=False)
                pds[idx, S * cidx : S * (cidx + 1)] = rgb.cpu().numpy()
            assert S * (cidx + 1) >= H * W

        pds = pds.reshape(N, W, H)

        cfg.evaluation(pds)

def test(cfg):
    N, W, H, S = cfg.testset.__len__(), int(cfg.cam_scale * cfg.testset.W), int(cfg.cam_scale * cfg.testset.H), cfg.bs_test
    pds = np.zeros((N, H * W))
    dataloader = tqdm(cfg.testloader)
    axis = reduce(lambda x1, x2 : str(x1) + str(x2), [int(axis) for axis in cfg.axis])
    zs, angles, scales = [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            coords, depths, R, zpos, angle, scale = batch
            zpos, angle, scale = zpos.item(), int(angle.item()), scale.item()
            zs.append(zpos)
            angles.append(angle)
            scales.append(scale)
            dataloader.set_description(f'[TEST] pos : {zpos:.2f} | axis : {axis} | angle : {angle} | scale : {scale:.2f}x')
            coords, R = torch.squeeze(coords), torch.squeeze(R)
            flags = utils.judge_range(coords, R)
            for cidx in range(H * W // S + 1):
                select_inds = list(range(S * cidx, min(S * (cidx + 1), len(coords))))
                select_flags = flags[select_inds]
                valid_inds = torch.tensor(select_inds).long()[select_flags]
                select_coords = coords[valid_inds]
                if len(select_coords) > 0:
                    valid_inds = valid_inds.cpu().numpy()
                    rgb, _ = cfg.Render(select_coords, depths, is_train=False, R=R)
                    rgb.cpu().numpy()
                    pds[idx, valid_inds] = rgb.cpu().numpy()
        pds = np.clip(pds.reshape((N, W, H)), 0, 1)

    if cfg.save_map: 
        cfg.Save_test_map(pds, zs, angles, scales)
    
if __name__ == '__main__':

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    seed = 0

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    args = argParse()
    globals()[args.mode](Cfg(args))