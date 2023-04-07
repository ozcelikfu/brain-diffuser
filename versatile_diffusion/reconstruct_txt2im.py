import os
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset
import argparse

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt

from skimage.transform import resize, downscale_local_mean

def regularize_image(x):
        BICUBIC = PIL.Image.Resampling.BICUBIC
        if isinstance(x, str):
            x = Image.open(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, PIL.Image.Image):
            x = x.resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            assert False, 'Unknown image type'

        assert (x.shape[1]==512) & (x.shape[2]==512), \
            'Wrong image size'
        return x

cfgm_name = 'vd_noema'
sampler = DDIMSampler_VD
pth = '/home/furkan/Versatile-Diffusion/pretrained/vd-four-flow-v1-0-fp16.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    

#net.model.cuda(1)
sampler = sampler(net)
#sampler.model.model.cuda(1)
#sampler.model.cuda(1)
net.clip.cuda(0)
net.autokl.cuda(0).half()
sampler.model.model.diffusion_model.device='cuda:1'
sampler.model.model.diffusion_model.half().cuda(1)

pred_clip = np.load('/home/furkan/Versatile-Diffusion/extractedfeatures/nsd/nsd_cliptext_predtest_sepembeds.npy')
pred_clip = torch.tensor(pred_clip).half().cuda(1)

n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5
xtype = 'image'
ctype = 'prompt'
h, w = 512,512
shape = [n_samples, 4, h//8, w//8]

u = None
if scale != 1.0:
    dummy = ''
    u = net.clip_encode_text(dummy)
    u = u.cuda(1).half()

torch.manual_seed(0)
idx = [35,70,13,40,54,78,97,102,451,461,570,774]
for i in idx: #range(len(pred_clip)):

    c = pred_clip[i:i+1]
    

    z, _ = sampler.sample(
                steps=ddim_steps,
                shape=shape,
                conditioning=c,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=u,
                xtype=xtype, ctype=ctype,
                eta=ddim_eta,
                verbose=False,)


    z = z.cuda(0)
    x = net.autokl_decode(z)
    #x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
    #x = [tvtrans.ToPILImage()(xi) for xi in x]
    
    im = Image.open('/home/furkan/NSD/nsddata_stimuli/test_images/test_image{}.png'.format(i))
    im = regularize_image(im)
    cin = im*2 - 1
    color_adj='None'
    color_adj_to = cin
    color_adj_flag = (color_adj!='none') and (color_adj!='None') and (color_adj is not None)
    color_adj_simple = (color_adj=='Simple') or color_adj=='simple'
    color_adj_keep_ratio = 0.5

    if color_adj_flag and (ctype=='vision'):
        x_adj = []
        for xi in x:
            color_adj_f = color_adjust(ref_from=(xi+1)/2, ref_to=color_adj_to)
            xi_adj = color_adj_f((xi+1)/2, keep=color_adj_keep_ratio, simple=color_adj_simple)
            x_adj.append(xi_adj)
        x = x_adj
        x = [tvtrans.ToPILImage()(xi) for xi in x]
    else:
        x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
        x = [tvtrans.ToPILImage()(xi) for xi in x]

    x[0].save('/home/furkan/Versatile-Diffusion/results/nsd_trials/nsd_cliptext_fmri/{}.png'.format(i))
  

