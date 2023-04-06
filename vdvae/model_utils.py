import torch
import numpy as np
#from mpi4py import MPI
import socket
import argparse
import os
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
#from apex.optimizers import FusedAdam as AdamW
from vae import VAE
from torch.nn.parallel.distributed import DistributedDataParallel
from train_helpers import restore_params

def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    setup_mpi(H)
    setup_save_dirs(H)
    logprint = logger(H.logdir)
    for i, k in enumerate(sorted(H)):
        logprint(type='hparam', key=k, value=H[k])
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)
    logprint('training model', H.desc, 'on', H.dataset)
    return H, logprint

def set_up_data(H):
    shift_loss = -127.5
    scale_loss = 1. / 127.5
    
    #trX, vaX, teX = imagenet64(H.data_root)
    H.image_size = 64
    H.image_channels = 3
    shift = -115.92961967
    scale = 1. / 69.37404
    
    #if H.test_eval:
    #    print('DOING TEST')
    #    eval_dataset = teX
    #else:
    #    eval_dataset = vaX

    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)
    
    #train_data = TensorDataset(torch.as_tensor(trX))
    #valid_data = TensorDataset(torch.as_tensor(eval_dataset))
    #untranspose = False

    def preprocess_func(x):
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        #untranspose = False
        #if untranspose:
        #    x[0] = x[0].permute(0, 2, 3, 1)
        inp = x.cuda(non_blocking=True).float()
        out = inp.clone()
        inp.add_(shift).mul_(scale)
        out.add_(shift_loss).mul_(scale_loss)
        return inp, out
    
    return H, preprocess_func

def load_vaes(H, logprint=None):

    ema_vae = VAE(H)
    if H.restore_ema_path:
        print(f'Restoring ema vae from {H.restore_ema_path}')
        restore_params(ema_vae, H.restore_ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size)
    else:
        ema_vae.load_state_dict(vae.state_dict())
    ema_vae.requires_grad_(False)
    ema_vae = ema_vae.cuda(H.local_rank)

    #vae = DistributedDataParallel(vae, device_ids=[H.local_rank], output_device=H.local_rank)

    #if len(list(vae.named_parameters())) != len(list(vae.parameters())):
    #    raise ValueError('Some params are not named. Please name all params.')
    #total_params = 0
    #for name, p in vae.named_parameters():
    #    total_params += np.prod(p.shape)

    #print(total_params=total_params, readable=f'{total_params:,}')
    return ema_vae