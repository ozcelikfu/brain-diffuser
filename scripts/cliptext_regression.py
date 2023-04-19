import sys
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
train_fmri = np.load(train_path)
test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
test_fmri = np.load(test_path)

## Preprocessing fMRI

train_fmri = train_fmri/300
test_fmri = test_fmri/300


norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

print(np.mean(train_fmri),np.std(train_fmri))
print(np.mean(test_fmri),np.std(test_fmri))

print(np.max(train_fmri),np.min(train_fmri))
print(np.max(test_fmri),np.min(test_fmri))

num_voxels, num_train, num_test = train_fmri.shape[1], len(train_fmri), len(test_fmri)


train_clip = np.load('data/extracted_features/subj{:02d}/nsd_cliptext_train.npy'.format(sub))
test_clip = np.load('data/extracted_features/subj{:02d}/nsd_cliptext_test.npy'.format(sub))

## Regression
num_samples,num_embed,num_dim = train_clip.shape

print("Training Regression")
reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float32)
reg_b = np.zeros((num_embed,num_dim)).astype(np.float32)
pred_clip = np.zeros_like(test_clip)
for i in range(num_embed):
    reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True)
    reg.fit(train_fmri, train_clip[:,i])
    reg_w[i] = reg.coef_
    reg_b[i] = reg.intercept_
    
    pred_test_latent = reg.predict(test_fmri)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    pred_clip[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)
    print(i,reg.score(test_fmri,test_clip[:,i]))

np.save('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral.npy'.format(sub),pred_clip)


datadict = {
    'weight' : reg_w,
    'bias' : reg_b,

}

with open('data/regression_weights/subj{:02d}/cliptext_regression_weights.pkl'.format(sub),"wb") as f:
  pickle.dump(datadict,f)
