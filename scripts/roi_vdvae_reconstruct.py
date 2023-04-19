import sys
sys.path.append('vdvae')
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
from image_utils import *
from model_utils import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import pickle

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-bs", "--bs",help="Batch Size",default=30)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
batch_size=int(args.bs)

print('Libs imported')

H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
H = dotdict(H)

H, preprocess_fn = set_up_data(H)

print('Models is Loading')
ema_vae = load_vaes(H)

  
class batch_generator_external_images(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)


    def __getitem__(self,idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img,(64,64))
        img = torch.tensor(np.array(img)).float()
        #img = img/255
        #img = img*2 - 1
        return img

    def __len__(self):
        return  len(self.im)



image_path = 'data/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub)
test_images = batch_generator_external_images(data_path = image_path)
testloader = DataLoader(test_images,batch_size,shuffle=False)

test_latents = []
for i,x in enumerate(testloader):
  data_input, target = preprocess_fn(x)
  with torch.no_grad():
        print(i*batch_size)
        activations = ema_vae.encoder.forward(data_input)
        px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
        #recons = ema_vae.decoder.out_net.sample(px_z)
        batch_latent = []
        for i in range(31):
            batch_latent.append(stats[i]['z'].cpu().numpy().reshape(len(data_input),-1))
        test_latents.append(np.hstack(batch_latent))
        #stats_all.append(stats)
        #imshow(imgrid(recons, cols=batch_size,pad=20))
        #imshow(imgrid(test_images[i*batch_size : (i+1)*batch_size], cols=batch_size,pad=20))
test_latents = np.concatenate(test_latents)      

pred_latents = np.load('data/predicted_features/subj{:02d}/nsd_vdvae_nsdgeneral_roi_sub{}_31l_alpha50k.npy'.format(sub,sub))
ref_latent = stats

# Transfor latents from flattened representation to hierarchical
def latent_transformation(latents, ref):
  layer_dims = np.array([2**4,2**4,2**8,2**8,2**8,2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**14])
  transformed_latents = []
  for i in range(31):
    t_lat = latents[:,layer_dims[:i].sum():layer_dims[:i+1].sum()]
    #std_norm_test_latent = (t_lat - np.mean(t_lat,axis=0)) / np.std(t_lat,axis=0)
    #renorm_test_latent = std_norm_test_latent * np.std(kamitani_latents[i][num_test:].reshape(num_train,-1),axis=0) + np.mean(kamitani_latents[i][num_test:].reshape(num_train,-1),axis=0)
    c,h,w=ref[i]['z'].shape[1:]
    transformed_latents.append(t_lat.reshape(len(latents),c,h,w))
  return transformed_latents

idx = range(13)
input_latent = latent_transformation(pred_latents[idx],ref_latent)

  
def sample_from_hier_latents(latents,sample_ids):
  sample_ids = [id for id in sample_ids if id<len(latents[0])]
  layers_num=len(latents)
  sample_latents = []
  for i in range(layers_num):
    sample_latents.append(torch.tensor(latents[i][sample_ids]).float().cuda())
  return sample_latents

#samples = []

res_dir = 'results/vdvae/subj{:02d}/roi/'.format(sub)
if not os.path.exists(res_dir):
   os.makedirs(res_dir)

for i in range(1):
  print(i*batch_size)
  samp = sample_from_hier_latents(input_latent,range(i*batch_size,(i+1)*batch_size))
  px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
  sample_from_latent = ema_vae.decoder.out_net.sample(px_z)
  upsampled_images = []
  for j in range(len(sample_from_latent)):
      im = sample_from_latent[j]
      im = Image.fromarray(im)
      im = im.resize((512,512),resample=3)
      im.save('results/vdvae/subj{:02d}/roi/{}.png'.format(sub,i*batch_size+j))
      

