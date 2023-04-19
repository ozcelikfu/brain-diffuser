import numpy as np
import pickle

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

# Load ROI Masks

with open('data/regression_weights/subj{:02d}/vdvae_regression_weights.pkl'.format(sub),"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']

roi_dir = 'data/processed_data/subj{:02d}/roi'.format(sub)
num_rois = 13
roi_act=np.zeros((num_rois,reg_w.shape[1])).astype(np.float32)
roi_act[0] = np.load("{}/floc-faces.npy".format(roi_dir))
roi_act[1] = np.load("{}/floc-words.npy".format(roi_dir))
roi_act[2] = np.load("{}/floc-places.npy".format(roi_dir))
roi_act[3] = np.load("{}/floc-bodies.npy".format(roi_dir))
roi_act[4] = np.load("{}/V1.npy".format(roi_dir))
roi_act[5] = np.load("{}/V2.npy".format(roi_dir))
roi_act[6] = np.load("{}/V3.npy".format(roi_dir))
roi_act[7] = np.load("{}/V4.npy".format(roi_dir))
roi_act[8] = np.load("{}/ecc05.npy".format(roi_dir))
roi_act[9] = np.load("{}/ecc10.npy".format(roi_dir))
roi_act[10] = np.load("{}/ecc20.npy".format(roi_dir))
roi_act[11] = np.load("{}/ecc40.npy".format(roi_dir))
roi_act[12] = np.load("{}/ecc40p.npy".format(roi_dir))


roi_act[roi_act>0]=1
roi_act[roi_act<0]=0

# Generate VDVAE Features

nsd_features = np.load('data/extracted_features/subj{:02d}/nsd_vdvae_features_31l.npz'.format(sub))
train_latents = nsd_features['train_latents']

pred_vae = (roi_act @ reg_w.T) 
pred_vae = pred_vae / (np.linalg.norm(pred_vae,axis=1).reshape((num_rois,1)) + 1e-8)
pred_vae = pred_vae * 50 + reg_b

pred_vae = (pred_vae - np.mean(pred_vae,axis=0)) / np.std(pred_vae,axis=0)

pred_vae = pred_vae * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)
pred_vae = pred_vae / np.linalg.norm(pred_vae,axis=1).reshape((num_rois,1))
pred_vae = pred_vae * 80
np.save('data/predicted_features/subj{:02d}/nsd_vdvae_nsdgeneral_roi_sub{}_31l_alpha50k.npy'.format(sub,sub),pred_vae)

# Generate CLIP-Text Features

with open('data/regression_weights/subj{:02d}/cliptext_regression_weights.pkl'.format(sub),"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']

pred_clipt = np.zeros((num_rois,reg_w.shape[0],reg_w.shape[1])).astype(np.float32)
for j in range(reg_w.shape[0]):
    pred_clipt[:,j] = (roi_act @ reg_w[j].T) 
    
pred_clipt = pred_clipt / (np.linalg.norm(pred_clipt,axis=(1,2)).reshape((num_rois,1,1)) + 1e-8)
pred_clipt = pred_clipt * 9 + reg_b

np.save('data/predicted_features/subj{:02d}/nsd_cliptext_roi_nsdgeneral.npy'.format(sub),pred_clipt)

# Generate CLIP-Vision Features

with open('data/regression_weights/subj{:02d}/clipvision_regression_weights.pkl'.format(sub),"rb") as f:
    datadict = pickle.load(f)
    reg_w = datadict['weight']
    reg_b = datadict['bias']
    
pred_clipv = np.zeros((num_rois,reg_w.shape[0],reg_w.shape[1])).astype(np.float32)
for j in range(reg_w.shape[0]):
    pred_clipv[:,j] = (roi_act @ reg_w[j].T) 
    
pred_clipv = pred_clipv / (np.linalg.norm(pred_clipv,axis=(1,2)).reshape((num_rois,1,1)) + 1e-8)
pred_clipv = pred_clipv * 15 + reg_b

np.save('data/predicted_features/subj{:02d}/nsd_clipvision_roi_nsdgeneral.npy'.format(sub),pred_clipv)
