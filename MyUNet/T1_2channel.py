# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:36:05 2019

@author: z3439910
"""
train_path = r"K:\THEJUDGEMENT\TRAINING_DATA_UNET\datasets\train"

image_name = "2012147N30284_201205262230"
BT_dir = os.path.join(train_path,image_name,"image",image_name + "_bt.h5")
BT_channel = np.array(h5py.File(BT_dir,'r')['bt'],dtype=np.int16)
DIST_dir = os.path.join(train_path,image_name,"dist",image_name + "_dist.h5")
DIST_channel = np.array(h5py.File(DIST_dir,'r')['dist'], dtype=np.int16)
image = np.zeros((550,550,2))
image[...,0] = BT_channel
image[...,1] = DIST_channel
#        
#        image = BT_channel
LABEL_dir = os.path.join(train_path,image_name,"label",image_name + "_label.h5")
label = np.array(h5py.File(LABEL_dir,'r')['label'],dtype=np.bool)
