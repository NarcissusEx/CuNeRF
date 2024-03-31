import os
import math
# from multiprocessing import Process
from multiprocessing import Process, Manager
from functools import reduce
import queue
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.cpp_extension import load

def dfs_update_configs(cfg):
    q = queue.Queue()
    q.put(cfg)
    while not q.empty():
        tmp = q.get()
        if isinstance(tmp, dict):
            for key in tmp.keys():
                if isinstance(tmp[key], str):
                    if '$' in tmp[key]:
                        klist = [k if i % 2 == 0 else cfg[k] for i, k in enumerate(tmp[key].split('$'))]
                        try:
                            tmp[key] = cfg[reduce(lambda x1, x2 : x1 + x2, filter(lambda x : x != '', klist))]
                        except:
                            tmp[key] = reduce(lambda x1, x2 : x1 + x2, filter(lambda x : x != '', klist))
                q.put(tmp[key])

def merge_configs(args, bcfg):
    for k, v in args.__dict__.items():
        if (k in bcfg.keys() and v is not None) or k not in bcfg.keys():
            bcfg[k] = v

    return bcfg

def clip_grad(self, max_val, max_norm):
    # Clip the gradients of each MLP individually.
    for param_group in self.optim.param_groups:
        if max_val > 0:
            torch.nn.utils.clip_grad_value_(param_group['params'], max_val)

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(param_group['params'], max_norm)

def lr_decay(step, self, lr_init, lr_final, max_iter, lr_delay_steps=0, lr_delay_mult=1):

    def log_lerp(t, v0, v1):
        lv0 = np.log(v0)
        lv1 = np.log(v1)
        return np.exp(np.clip(t, 0, 1) * (lv1 - lv0) + lv0)

    if lr_delay_steps > 0:
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))

    else:
        delay_rate = 1.

    new_lr = delay_rate * log_lerp(step / max_iter, lr_init, lr_final)

    for param_group in self.optim.param_groups:
        param_group['lr'] = new_lr
    
    return new_lr

def mlt_process(rets, const_params, params, function, num_workers=8, is_tqdm=False):
    q = queue.Queue(num_workers)
    for idx, param in enumerate(tqdm(params) if is_tqdm else params):
        if num_workers > 0:
            t = Process(target=function, args=(rets, idx, *const_params, *param))
            t.start()
            q.put(t)
            if q.full():
                while not q.empty():
                    t = q.get()
                    t.join()
        else:
            function(rets, idx, *const_params, *param)

    while not q.empty():
        t = q.get()
        t.join()
    
    return rets

def get_rotate_matrix(axi, a):
    # Rodrigues' Rotation Formulation
    # Inputs: An arbitrary axi and angle a
    # Outputs: A rotate matrix R = cos(a)I + (1-cos(a)) * V.T * V + sin(a) * K
    A = a / 180 * math.pi
    V = torch.tensor(axi)
    V = V / torch.sqrt(torch.sum(torch.square(V)))
    K = torch.tensor([
        [0, -V[2], V[1]],
        [V[2], 0, -V[0]],
        [-V[1], V[0], 0]
    ])
    R = torch.eye(3) * math.cos(A) + (1 - math.cos(A)) * V[:, None] @ V[None] + math.sin(A) * K
    return R

def judge_range(coords, R):
    C, _, _ = torch.split(coords, [3, 2, 2], dim=-1)
    _C = C @ R
    return torch.logical_and(torch.sum(_C < math.pi, -1) == 3, torch.sum(_C > -math.pi, -1) == 3)