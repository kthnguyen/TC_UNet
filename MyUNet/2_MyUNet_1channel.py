# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:33:52 2019

@author: z3439910
"""

from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np

from scipy.ndimage import gaussian_filter
import h5py
import os, sys

#os.chdir(os.path.dirname(os.path.abspath(__file__)))
from tf_unet import unet
from tf_unet import util
from tf_unet.image_util import BaseDataProvider
import matplotlib.pyplot as plt 
from matplotlib import colors

#%%
data_root = r"K:\THEJUDGEMENT\TRAINING_DATA_UNET\datasets\train"
output_path = r"./unet_trained_ufig_withdist"
training_iters = 20
epochs = 10
restore = False
layers = 3
features_root = 16


class DataProvider(BaseDataProvider):
    channels = 1
    n_class = 2
    a_min = -np.inf
    a_max = np.inf
    
    def __init__(self, train_path):
        self.file_idx = -1
        self.data_files = self._find_data_files(train_path)
        self.train_path = train_path
        
    def _find_data_files(self, train_path):
        folder_names = os.listdir(train_path)
        return folder_names
    
    def _cycle_file(self):
        self.file_idx = np.random.choice(len(self.data_files))

    def _next_data(self):
        self._cycle_file()
        image_name = self.data_files[self.file_idx]

        BT_dir = os.path.join(self.train_path,image_name,"image",image_name + "_bt.h5")
        BT_channel = np.array(h5py.File(BT_dir,'r')['bt'],dtype=np.int16)
#        DIST_dir = os.path.join(self.train_path,image_name,"dist",image_name + "_dist.h5")
#        DIST_channel = np.array(h5py.File(DIST_dir,'r')['dist'], dtype=np.int16)
#        image = np.dstack((BT_channel,DIST_channel))
#        
        image = BT_channel
        LABEL_dir = os.path.join(self.train_path,image_name,"label",image_name + "_label.h5")
        label = np.array(h5py.File(LABEL_dir,'r')['label'],dtype=np.bool)
        return image, label
#%%
data_provider = DataProvider(data_root)
data,label = data_provider(1)
weights = None

net = unet.Unet(channels=data_provider.channels,
                n_class=data_provider.n_class,
                layers=layers,
                features_root=features_root,
                cost_kwargs=dict(regularizer=0.001,
                                 class_weights=weights))
path = output_path

trainer = unet.Trainer(net, optimizer="adam", opt_kwargs=dict(beta1=0.91))

path_2 = trainer.train(data_provider, path, 
                       training_iters=training_iters,
                       epochs=epochs,
                       dropout=0.5,
                       display_step=2,
                       restore=restore)

#%%
test_data_root = r"K:\THEJUDGEMENT\TRAINING_DATA_UNET\datasets\test"
test_data_provider = DataProvider(test_data_root)
data,label = test_data_provider(1)
prediction = net.predict(path_2,data)
print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))))

#%
im_data = data[0,...,0]
plt.figure()
plt.imshow(im_data,cmap='Greys',origin='lower')

im_mask0 = prediction[0,...,0]
im_mask1 = prediction[0,...,1]
im0 = plt.figure()
plt.imshow(im_mask0,cmap='Greys',origin='lower')
im1 = plt.figure()
plt.imshow(im_mask1,cmap='Greys',origin='lower')

#%%
test_data_root = r"K:\THEJUDGEMENT\TRAINING_DATA_UNET\datasets\test"
test_data_provider = DataProvider(test_data_root)
data,label = test_data_provider(1)
prediction = net.predict(path_2,data)
print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))))


data_crop = util.crop_to_shape(data, prediction.shape)
im_data = data_crop[0,...,0]

label_crop = util.crop_to_shape(label, prediction.shape)
im_mask1 = label_crop[0,...,1]
label_mask = np.where(im_mask1>0,1,np.nan)

im_mask0 = prediction[0,...,0]
im_mask1 = prediction[0,...,1]
prediction_mask = np.where(im_mask1>0.5,1,np.nan)
plt.figure()
plt.subplot(211)
plt.imshow(im_data,cmap='Greys',origin='lower')
plt.imshow(label_mask,cmap=colors.ListedColormap(['yellow']),origin='lower',alpha=0.3)
plt.subplot(212)
plt.imshow(im_data,cmap='Greys',origin='lower')
plt.imshow(prediction_mask,cmap=colors.ListedColormap(['green']),origin='lower',alpha=0.3)
#%%
label_crop = util.crop_to_shape(label, prediction.shape)

im_data = data[0,...,0]
plt.figure()
plt.imshow(im_data,cmap='Greys',origin='lower')

im_mask0 = label_crop[0,...,0]
im_mask1 = label_crop[0,...,1]
im0 = plt.figure()
plt.imshow(im_mask0,cmap='Greys',origin='lower')
im1 = plt.figure()
plt.imshow(im_mask1,cmap='Greys',origin='lower')
#%%
a = np.sum(np.argmax(prediction, 3) == np.argmax(label_crop, 3))
