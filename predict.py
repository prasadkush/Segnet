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

from labels import Label, id2myid, id2label, names2mynames

def getmyids():
    myids = np.zeros((34,))
    for i in range(34):
        name = id2label[i].name
        if name in list(mylabels.keys()):
            myids[i] = mylabels[name]
        elif name in list(names2mynames.keys()):
            namekey = names2mynames[name]
            myids[i] = mylabels[namekey] 
        else:
            myids[i] = 14
    return myids

def get_my_colors():
	max_index = max(list(mylabels.values()))
	print('max_index: ', max_index)
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


def predict(img, imgs, imginds, modelpath):
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
	imgs = cv2.resize(imgs, (480, 360), interpolation=cv2.INTER_NEAREST)
	img = img.transpose(2,0,1)
	print('img shape: ', img.shape)
	print('img: ', img)
	imgt = torch.from_numpy(img)
	imgt = color_transform(imgt)
	imgt = torch.unsqueeze(imgt, 0)
	#print('imgt shape: ', imgt.shape)
	#print('imgt: ', imgt)
	out = model.forward(imgt)
	#print('out shape: ', out.shape)
	#print('out: ', out)
	inds = torch.argmax(out, dim=1)
	#print('inds: ', inds)
	inds = torch.squeeze(inds, 0)
	#print('inds shape: ', inds.shape)
	#print('np.unique(inds.flatten()): ', np.unique(inds.flatten()))
	color_arr = get_my_colors()
	outimg = np.ones((360,480,3))
	indices = np.indices((360,480))
	outimg[indices[0,:,:],indices[1,:,:],:] = color_arr[inds]
	outimg2 = np.ones((360,480,3))
	outimg2[indices[0,:,:],indices[1,:,:],:] = color_arr[imginds]
	outimg = outimg.astype('uint8')
	outimg2 = outimg2.astype('uint8')
	images = np.concatenate((outimg, imgs), axis=1)
	print('inds: ', inds)
	print('np.unique(inds.flatten()): ', np.unique(inds.flatten()))
	print('np.unique(imginds.flatten()): ', np.unique(imginds.flatten()))
	cv2.imshow('outimg int: ', outimg)
	cv2.imshow('imgs: ', imgs)
	cv2.imshow('outimg2: ', outimg2)
	cv2.imshow('images: ', images)
	cv2.waitKey(0)
	breakpoint()
	print('color_arr[inds] shape: ', color_arr[inds].shape)
	print('color_arr[inds]: ', color_arr[inds])

def predict_single_image(img, imgs, imgorig, modelpath=None, model=None, imgdir='results', optimizer=None, epoch=None):
	if model == None:
		model = Segnet(7,3)	
	#if optimizer == None:
	#	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
	if modelpath != None:
		checkpoint = torch.load(modelpath)
		model.load_state_dict(checkpoint['model_state_dict'])
		epoch = checkpoint['epoch']
	#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	if epoch == None:
		epoch = 10
	model.eval()
	torch.no_grad()
	#print('epoch: ', epoch)
	#img = cv2.imread(imgpath)
	#color_transform = get_color_transform('kitti')
	#print('img shape: ', img.shape)
	#print('img: ', img)
	#print('imgt shape: ', imgt.shape)
	#print('imgt: ', imgt)
	out = model.forward(img)
	#print('out shape: ', out.shape)
	#print('out: ', out)
	inds = torch.argmax(out, dim=1)
	#print('inds: ', inds)
	#inds = torch.squeeze(inds, 0)
	#imgs = torch.squeeze(imgs, 0)
	#imgorig = torch.squeeze(imgorig, 0)
	#print('inds shape: ', inds.shape)
	#print('np.unique(inds.flatten()): ', np.unique(inds.flatten()))
	color_arr = get_my_colors()
	outimg = np.ones((img.shape[0], 360,480,3))
	indices = np.indices((img.shape[0], 360,480))
	#outimg[indices[0,:,:],indices[1,:,:],:] = color_arr[inds]
	outimg[indices[0,:,:,:], indices[1,:,:,:],indices[2,:,:,:],:] = color_arr[inds]
	outimg2 = np.ones((img.shape[0], 360,480,3))
	#outimg2[indices[0,:,:],indices[1,:,:],:] = color_arr[imgs]
	outimg2[indices[0,:,:,:], indices[1,:,:,:],indices[2,:,:,:],:] = color_arr[imgs]
	outimg = outimg.astype('uint8')
	outimg2 = outimg2.astype('uint8')
	#imgorig = torch.permute(imgorig, (1,2,0))
	imgorig = torch.permute(imgorig, (0,2,3,1))
	imgorig = 255*imgorig
	imgorig = imgorig.numpy().astype('uint8')
	#print('inds: ', inds[0,:,:])
	#print('np.unique(inds.flatten()): ', np.unique(inds[0,:,:].flatten()))
	#print('np.unique(imgs.flatten()): ', np.unique(imgs[0,:,:].flatten()))
	#cv2.imwrite('outimg.jpg', outimg)
	#cv2.imwrite('outimg2.jpg', outimg2)
	#cv2.imwrite('imorig.jpg', imgorig.numpy())
	mean = 0
	for i in range(img.shape[0]):
		cv2.imwrite(imgdir + '/e' + str(epoch) + '_outimg_' + str(i) + '.jpg', outimg[i,:,:,:])
		cv2.imwrite(imgdir + '/e' + str(epoch) + '_img_'  + str(i) + '.jpg', outimg2[i,:,:,:])
		cv2.imwrite(imgdir + '/e' + str(epoch) + '_imgorig_' + str(i)  + '.jpg', imgorig[i,:,:,:])
		mean += np.sum(np.equal(inds[i,:,:].numpy(), imgs[i,:,:].numpy()))/(inds.shape[1]*inds.shape[2])
		print('np.sum(np.equal(inds[i,:,:], imgs[i,:,:]))/(inds.shape[1]*inds.shape[2]): ', np.sum(np.equal(inds[i,:,:].numpy(), imgs[i,:,:].numpy()))/(inds.shape[1]*inds.shape[2]))
	print('mean/img.shape[0]: ', mean/img.shape[0])
	return mean/img.shape[0]
	#print('color_arr[inds] shape: ', color_arr[inds].shape)
	#print('color_arr[inds]: ', color_arr[inds])


if __name__ == 'main':
	datapath = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/datasets/data_semantics/training/image_2'
	modelpath = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/ML code/Segnet/bestlosssegnetmodel.pt'
	semrgbdatapath = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/datasets/data_semantics/training/semantic_rgb'
	semdatapath = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/datasets/data_semantics/training/semantic' 
	trainingdata = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/datasets/data_semantics/training'

	val_dataset = getDataset(trainingdata)
	loader = DataLoader(val_dataset, batch_size=4, shuffle=True, pin_memory=True)
	loaderiter = iter(loader)
	data = next(loaderiter)
	img = data['image']
	imgs = data['semantic']
	imgorig = data['original']

	predict_single_image(img, imgs, imgorig, modelpath)

'''
i = 0
for filename in os.listdir(datapath):
	if i > 0:
		break
	filepath = os.path.join(datapath, filename)
	semrgbfilepath = os.path.join(semrgbdatapath, filename)
	semfilepath = os.path.join(semdatapath, filename)
	img = cv2.imread(filepath)
	imgs = cv2.imread(semrgbfilepath)
	imginds = cv2.imread(semfilepath)
	imginds = cv2.resize(imginds, (480, 360), interpolation=cv2.INTER_NEAREST)
	imginds = imginds[:,:,0]
	print('imginds before: ', imginds)
	myids = getmyids()
	arr = np.arange(34)
	d = np.nonzero(imginds[:,:,np.newaxis] == arr)
	myids = getmyids()
	imginds[d[0],d[1]] = myids[d[2]]
	print('imginds after: ', imginds)
	#print('img shape: ', img.shape)
	#cv2.imshow('img: ', img)
	#cv2.waitKey(0)
	predict(img, imgs, imginds, modelpath)
	i = i+1
'''