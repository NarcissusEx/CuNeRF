##########################################################################
# The Expmental Manager of CuNeRF
# @Author: Zixuan Chen
# @Email:  chenzx3@mail2.sysu.edu.cn
##########################################################################

import os
import copy
import math
from functools import reduce, partial

import yaml
from tqdm import tqdm

import numpy as np
import SimpleITK as sitk
from PIL import Image

import torch
from torch.utils.data import DataLoader
from . import dataset, loss, metrics, rendering, sampling, models, importance
from .utils import dfs_update_configs, clip_grad, lr_decay, merge_configs, mlt_process

from torchsummary import summary

class Cfg:

    def __init__(self, args):
        with open(args.cfg, 'r') as bf:
            bcfg = yaml.safe_load(bf)
        
        cfg = merge_configs(args, bcfg)
        dfs_update_configs(cfg)
        self.load_cfg(cfg)

        if self.resume or self.mode in ['eval', 'test']: 
            self.Resume()

        if self.mode == 'train':
            self.pbar = tqdm(initial=self.i_step, total=self.max_iter, position=0, leave=True)
            self.m_psnr, self.m_loss, self.m_lr = [], [], []

        # self.save_psnr = False

    def load_cfg(self, cfg):

        # loading config setup
        def load_model(self, cfg):
            self.model = getattr(models, cfg.pop('name'))(**cfg).cuda()
            

            summary(self.model, input_size=(64,3))

        def load_model_ft(self, cfg):
            self.model_ft = getattr(models, cfg.pop('name'))(**cfg).cuda()

        def load_optim(self, cfg):
            params = list(self.model.parameters()) + list(self.model_ft.parameters())
            self.optim = getattr(torch.optim, cfg.pop('name'))(params, **cfg)

        def load_dataset(self, cfg):
            mode, name = cfg.pop('mode'), cfg.pop('dname')
            setattr(self, f'{mode}set', getattr(dataset, name)(mode=mode, **cfg[mode]))
            if mode == 'train':
                self.evalset = getattr(dataset, name)(mode='eval', **cfg['eval'])

        def load_dataloader(self, cfg):
            mode = cfg.pop('mode')
            setattr(self, f'{mode}loader', DataLoader(getattr(self, f'{mode}set'), **cfg[mode], generator=torch.Generator(device='cuda:1') ))
            if mode == 'train':
                self.evalloader = DataLoader(self.evalset, **cfg['eval'], generator=torch.Generator(device='cuda:1'))

        def load_metrics(self, cfg):
            self.metrics = metrics.Metrics(cfg)
        
        def load_lr_decay(self, cfg):
            self.lr_decay = partial(globals()['lr_decay'], self=self, **cfg)

        def load_clip_grad(self, cfg):
            self.clip_grad = partial(globals()['clip_grad'], self=self, **cfg)

        def load_sampling(self, cfg):
            self.sample_fn = partial(getattr(sampling, f"{cfg.pop('stype')}_sampling"), **cfg)

        def load_importance(self, cfg):
            self.imp_fn = partial(getattr(importance, f"{cfg.pop('stype')}_imp"), **cfg)
        
        def load_rendering(self, cfg):
            self.render_fn = partial(getattr(rendering, f"{cfg.pop('stype')}_rendering"), **cfg)

        def load_loss(self, cfg):
            self.loss_fn = partial(getattr(loss, cfg.pop('name')), **cfg)

        text_cfgs = ''
        # fname = reduce(lambda x1, x2 : f'{x1}.{x2}', os.path.basename(cfg['file']).split('.')[:-1])
        fname = os.path.basename(cfg['file']).split('.')[0]
        for key, value in cfg.items():
            if isinstance(value, dict):
                text_cfgs += f'{key}\n'
                for k, v in value.items():
                    text_cfgs += f'  {k:<15} : {v}\n'
                text_cfgs += '----------------------------------------------------------\n'
                if f'load_{key}' in locals().keys():
                    locals()[f'load_{key}'](self, copy.deepcopy(value))
                    continue

            else:
                if key.endswith('path'):
                    value = os.path.join(value, cfg['expname'], fname)
                    cfg[key] = value
                    os.makedirs(value, exist_ok=True)
                text_cfgs += f'{key:<15} : {value}\n'
            setattr(self, key, value)

        self.i_step = 1
        self.scores = {key : 0 if value else 1e2 for key, value in self.metrics.getDict().items()}
        
        if self.mode == 'train':
            self.log_file = open(os.path.join(self.log_path, 'logs.txt'), 'a' if self.resume else 'w')
            # with open(os.path.join(self.log_path, 'config.yaml'), 'w') as yf:
            #     yaml.safe_dump(cfg, yf, indent=0)

            if not self.resume:
                self.log_file.write(text_cfgs)
            
            print (text_cfgs)

    def Record(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        self.m_psnr.append(self.metrics['PSNR'](self.i_pred, self.i_target))
        self.m_loss.append(self.i_loss)
            
    # special funcs
    def Log(self):
        mdict = {k : sum(getattr(self, f'm_{k}')) / len(getattr(self, f'm_{k}')) for k in ['lr', 'loss', 'psnr']}
        mtext = reduce(lambda x1, x2 : f'{x1}  {x2}', [f'{k.upper()} : {v:.6f}' for k, v in mdict.items()])
        logs = f'[TRAIN] Iter: {self.i_step} {mtext}'
        self.log_file.write(f'{logs}\n')
        self.log_file.flush()
        self.pbar.set_description(logs)
        self.m_psnr, self.m_loss = [], []

    def Resume(self):
        ckpt = os.path.join(self.save_path, f'{self.resume_type}.pkl')

        # find the resume file
        try:
            print (f'Resuming from {ckpt}')
            params = torch.load(ckpt)
            self.i_step = params.pop('i_step') + 1
            self.scores = params.pop('scores')

            # reload state_dict
            for k, v in params.items():
                getattr(self, k).load_state_dict(v)

        except:
            print (f'Failing in resuming from {ckpt}')

            # no resume file && eval -> gg
            if self.mode == 'eval':
                print ('No resume file for eval')
                raise

    def Save(self, name='current', iflog=True):
        save_path = os.path.join(self.save_path, f'{name}.pkl')
        save_dict = {}

        # save parameters
        for item in filter(lambda x : hasattr(self, x), ['model', 'model_ft', 'optim', 'i_step', 'scores']):
            try:
                save_dict[item] = getattr(self, item).state_dict()

            except:
                save_dict[item] = getattr(self, item)

        torch.save(save_dict, save_path)
        if iflog:
            print(f'Saving checkpoint at {save_path}')

    def Save_test_map(self, maps, zs, angles, scales):
        axis = reduce(lambda x1, x2 : str(x1) + str(x2), [int(axis) for axis in self.axis])
        angle = self.angles[0] if len(self.angles) == 1 else f'{self.angles[0]}-{self.angles[1]}'
        scale = f'{self.scales[0]:.1f}x' if len(self.scales) == 1 else f'{self.scales[0]:.1f}x-{self.scales[1]:.1f}x'
        zpos = f'{self.zpos[0]:.2f}' if len(self.zpos) == 1 else f'{self.zpos[0]:.2f}-{self.zpos[1]:.2f}'
        result_dir = os.path.join(self.result_path, 'test', f'{zpos}_{axis}_{angle}_{scale}')
        os.makedirs(result_dir, exist_ok=True)
        d = len(str(len(maps))) + 1
        maps = np.uint8(maps * 255.)
        imglist = []
        import cv2
        if self.is_details:
            for idx, (mp, z, a, s) in enumerate(zip(maps, zs, angles, scales)):
                mp = cv2.cvtColor(mp, cv2.COLOR_GRAY2RGB)
                mp = cv2.putText(mp, f'pos={z:.2f}|angle={a}|scale={s:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                img = Image.fromarray(mp)
                imglist.append(img)
                img.save(os.path.join(result_dir, str(idx).zfill(d) + '.png'))

        else:
            for idx, mp in enumerate(maps):
                Image.fromarray(mp).save(os.path.join(result_dir, str(idx).zfill(d) + '.png'))

        if self.is_gif:
            gif_path = os.path.join(result_dir, 'result.gif')
            imglist[0].save(gif_path, save_all=True, append_images=imglist, loop=0, duration=100)

        if self.is_video:
            video_path = os.path.join(result_dir, 'result.mp4')
            os.system(f'ffmpeg -i {result_dir}/%0{d}d.png -pix_fmt yuv422p -vcodec libx264 -vsync 0 {video_path} -y')

    def Save_map(self, pds, gts=None):
        
        def save_map_unit(rets, tid, sufix, N, mp):
            savepath = os.path.join(result_path, f'{str(tid).zfill(len(str(N)))}_{sufix}.png')
            
            if not os.path.exists(savepath) or sufix != 'gt':
                Image.fromarray(mp).save(savepath)

        # names = self.evalset.get_names()
        result_path = os.path.join(self.result_path, 'eval')
        os.makedirs(result_path, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(pds), os.path.join(result_path, 'ours.nii.gz'))
        for sufix, maps in {'ours' : pds, 'gt' : gts}.items():
            if maps is not None:
                maps = np.uint8(maps * 255.)
                maps = [maps] if len(maps) == 1 else [[m] for m in maps]
                mlt_process(None, [sufix, len(maps)], maps, save_map_unit, self.workers)

    def Update_lr(self):
        self.i_lr = self.lr_decay(self.i_step)
        self.m_lr.append(self.i_lr.item())

    def Update_grad(self):
        self.clip_grad()

    def Update_score(self, scores):
        performs = []
        self.save_psnr = False
        for k, v in scores.items():
            if self.metrics.compare(k, self.scores[k], v):
                if k == 'psnr':
                    self.save_psnr = True
                self.scores[k] = v
                self.Save(k, iflog=False)
            performs.append(f"{k} : {self.scores[k]}")
        performs = '[BEST] ' + reduce(lambda x1, x2 : x1 + ' | ' + x2, performs)
        self.log_file.write(f'{performs}\n')
        print(performs)

    def Update(self, loss, rgb, gts):
        value = loss.item()
        self.m_psnr.append(self.metrics.psnr(rgb, gts))
        self.m_loss.append(value)
        self.Update_lr()
        
    def Render(self, coord_batch, depths, is_train=False, R=None):
        ans0 = self.sample_fn(coord_batch, depths, is_train=is_train, R=R)
        # print("ans0['pts'].shape:", ans0['pts'].shape)
        raw0 = self.model(ans0['pts'])
        out0 = self.render_fn(raw0, **ans0)
        ans = self.imp_fn(**ans0, **out0, is_train=is_train)
        raw = self.model_ft(ans['pts'])
        out = self.render_fn(raw, **ans)
        return out['rgb'], out0['rgb']
    
    def evaluation(self, pds):
        gts = self.evalset.getLabel()

        with torch.no_grad():
            scores = self.metrics.evaluation(pds, gts)
            scores_txt = reduce(lambda x1, x2 : x1 + ' | ' + x2, [f'{k.upper()} : {v}' for k, v in scores.items()])
            logs = f'[EVAL] {scores_txt}'
            if self.mode == 'train':
                self.log_file.write(f'{logs}\n')
                self.Update_score(scores)
                # self.record_image(pds, gts)

            if self.save_map:
                if (self.mode == 'train' and self.save_psnr) or self.mode != 'train':
                    self.Save_map(pds, gts)

        print(logs)