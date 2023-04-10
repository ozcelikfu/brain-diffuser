import sys
sys.path.append('versatile_diffusion')
import os
import PIL
from PIL import Image
import numpy as np

import torch
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import torchvision.transforms as T

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

cfgm_name = 'vd_noema'

pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.clip = net.clip.to(device)

class batch_generator_external_images(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)


    def __getitem__(self,idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img,(512,512))
        img = T.functional.to_tensor(img).float()
        #img = img/255
        img = img*2 - 1
        return img

    def __len__(self):
        return  len(self.im)
    
batch_size=1
image_path = 'data/processed_data/subj{:02d}/nsd_train_stim_sub{}.npy'.format(sub,sub)
train_images = batch_generator_external_images(data_path = image_path)

image_path = 'data/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub)
test_images = batch_generator_external_images(data_path = image_path)

trainloader = DataLoader(train_images,batch_size,shuffle=False)
testloader = DataLoader(test_images,batch_size,shuffle=False)

num_embed, num_features, num_test, num_train = 257, 768, len(test_images), len(train_images)

train_clip = np.zeros((num_train,num_embed,num_features))
test_clip = np.zeros((num_test,num_embed,num_features))

with torch.no_grad():
    for i,cin in enumerate(testloader):
        print(i)
        #ctemp = cin*2 - 1
        c = net.clip_encode_vision(cin)
        test_clip[i] = c[0].cpu().numpy()
    
    np.save('data/extracted_features/subj{:02d}/nsd_clipvision_test.npy'.format(sub),test_clip)
        
    for i,cin in enumerate(trainloader):
        print(i)
        #ctemp = cin*2 - 1
        c = net.clip_encode_vision(cin)
        train_clip[i] = c[0].cpu().numpy()
    np.save('data/extracted_features/subj{:02d}/nsd_clipvision_train.npy'.format(sub),train_clip)



    
