import cv2
import torch
import torch.nn as nn
from data import getDataset, get_color_transform, get_inverse_transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torch import optim
from model import Encoder, Segnet
import matplotlib.pyplot as plt
import numpy as np
from preprocess import get_mean_std
from labels import mylabels, mynames, name2label
import os
import pdb

def get_my_colors():
	max_index = max(list(mylabels.values()))
	color_arr = np.zeros((max_index+1,3))
	for i in range(max_index):
		name = mynames[i]
		color = np.array([0,0,0])
		if name == 'unknown':
		    color = np.array([0,0,0])
		else:
			color = name2label[name].color
			color = np.array(list(color)).reshape((1,3))
		color_arr[i,:] = color
		print('i: ', i, ' color: ', color_arr[i,:])
	return color_arr


def predict(img, modelpath):
	model = Segnet(7,3)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
	mean, std = get_mean_std('kitti')
	checkpoint = torch.load(modelpath)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
	print('epoch: ', epoch)
	#img = cv2.imread(imgpath)
	color_transform = get_color_transform('kitti')
	img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_NEAREST)
	img = img.transpose(2,0,1)
	print('img shape: ', img.shape)
	print('img: ', img)
	imgt = torch.from_numpy(img)
	imgt = color_transform(imgt)
	imgt = torch.unsqueeze(imgt, 0)
	print('imgt shape: ', imgt.shape)
	print('imgt: ', imgt)
	out = model.forward(imgt)
	print('out shape: ', out.shape)
	print('out: ', out)
	inds = torch.argmax(out, dim=1)
	print('inds: ', inds)
	inds = torch.squeeze(inds, 0)
	print('inds shape: ', inds.shape)
	print('np.unique(inds.flatten()): ', np.unique(inds.flatten()))
	color_arr = get_my_colors()
	outimg = np.ones((360,480,3))
	indices = np.indices((360,480))
	outimg[indices[0,:,:],indices[1,:,:],:] = color_arr[inds]
	outimg = outimg.astype('int8')
	cv2.imshow('outimg int: ', outimg)
	cv2.waitKey(0)
	breakpoint()
	print('color_arr[inds] shape: ', color_arr[inds].shape)
	print('color_arr[inds]: ', color_arr[inds])
	

datapath = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/datasets/data_semantics/training/image_2'
modelpath = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/ML code/Segnet/bestlosssegnetmodel.pt'

i = 0
for filename in os.listdir(datapath):
	if i > 0:
		break
	filepath = os.path.join(datapath, filename)
	img = cv2.imread(filepath)
	#print('img shape: ', img.shape)
	#cv2.imshow('img: ', img)
	#cv2.waitKey(0)
	predict(img, modelpath)
	i = i+1