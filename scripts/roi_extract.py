import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]


roi_dir = 'data/nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub)
betas_dir = 'data/nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub)
res_roi_dir = 'data/processed_data/subj{:02d}/roi/'.format(sub)
if not os.path.exists(res_roi_dir):
   os.makedirs(res_roi_dir)

nsdgeneral_mask_filename = 'nsdgeneral.nii.gz'
nsdgeneral_mask = nib.load(roi_dir+nsdgeneral_mask_filename).get_fdata()
nsdgeneral_mask[nsdgeneral_mask<0] = 0
num_voxel = nsdgeneral_mask[nsdgeneral_mask>0].shape[0]
print(f'NSD General : {num_voxel}')


mask_files = [
              'floc-faces.nii.gz',
              'floc-words.nii.gz',
              'floc-places.nii.gz',
              'floc-bodies.nii.gz'
              ]


    
for mfile in mask_files:
    roi_mask = nib.load(roi_dir+mfile).get_fdata()
    np.save('data/processed_data/subj{:02d}/roi/{}.npy'.format(sub,mfile[:-7]), roi_mask[nsdgeneral_mask>0])
    

roi_mask = nib.load(roi_dir+mask_files[0]).get_fdata()
v1 = np.zeros_like(nsdgeneral_mask)
v2 = np.zeros_like(nsdgeneral_mask)
v3 = np.zeros_like(nsdgeneral_mask)
v4 = np.zeros_like(nsdgeneral_mask)

v1[roi_mask==1] = 1
v1[roi_mask==2] = 1
v2[roi_mask==3] = 1
v2[roi_mask==4] = 1
v3[roi_mask==5] = 1
v3[roi_mask==6] = 1
v4[roi_mask==7] = 1

np.save('data/processed_data/subj{:02d}/roi/V1.npy'.format(sub), v1[nsdgeneral_mask>0])
np.save('data/processed_data/subj{:02d}/roi/V2.npy'.format(sub), v2[nsdgeneral_mask>0])
np.save('data/processed_data/subj{:02d}/roi/V3.npy'.format(sub), v3[nsdgeneral_mask>0])
np.save('data/processed_data/subj{:02d}/roi/V4.npy'.format(sub), v4[nsdgeneral_mask>0])


roi_mask = nib.load(roi_dir+"prf-eccrois.nii.gz").get_fdata()
ecc05 = np.zeros_like(nsdgeneral_mask)
ecc10 = np.zeros_like(nsdgeneral_mask)
ecc20 = np.zeros_like(nsdgeneral_mask)
ecc40 = np.zeros_like(nsdgeneral_mask)
ecc40p = np.zeros_like(nsdgeneral_mask)

ecc05[roi_mask==1] = 1
ecc10[roi_mask==2] = 1
ecc20[roi_mask==3] = 1
ecc40[roi_mask==4] = 1
ecc40p[roi_mask==5] = 1

np.save('data/processed_data/subj{:02d}/roi/ecc05.npy'.format(sub), ecc05[nsdgeneral_mask>0])
np.save('data/processed_data/subj{:02d}/roi/ecc10.npy'.format(sub), ecc10[nsdgeneral_mask>0])
np.save('data/processed_data/subj{:02d}/roi/ecc20.npy'.format(sub), ecc20[nsdgeneral_mask>0])
np.save('data/processed_data/subj{:02d}/roi/ecc40.npy'.format(sub), ecc40[nsdgeneral_mask>0])
np.save('data/processed_data/subj{:02d}/roi/ecc40p.npy'.format(sub), ecc40p[nsdgeneral_mask>0])

