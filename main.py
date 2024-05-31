import cv2
import numpy as np
from data import getDataset
from torch.utils.data import DataLoader, Dataset
from model import Encoder, Segnet
import torch
from train import train


from labels import mylabels, Label, id2myid, id2label

datapath = 'C:/Users/Kush/OneDrive/Desktop/CV-Ml/datasets/data_semantics/training'
dataset = getDataset(datapath)

model = Segnet(7,3)

train(dataset, model, resume_training=False, useWeights=True, resultsdir='results/layerwise/Decoder/layer234', pretrained_encoder=True)

#loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)

#loaderiter = iter(loader)

#data = next(loaderiter)

#print('data:', data)
#d = data['image'].numpy()
#ds = data['semantic'].numpy()
#print('d shape: ', d.shape)
#print('ds shape: ', ds.shape)
#for i in range(4):
#	print('np.unique(ds.flatten()) ', i, ': ', (np.unique(ds[i,:,:].flatten())))
#print('ds: ', ds)
#print('ds[0,0,0,:]: ', ds[0,0,0,:])
#print('ds[0,0,1,:]: ', ds[0,0,1,:])
#print('ds[0,0,2,:]: ', ds[0,0,2,:])
#print('ds[0,1,0,:]: ', ds[0,21,0,:])
#print('ds[0,0,0,:]: ', ds[1,200,120,:])
#print('ds[0,0,1,:]: ', ds[1,100,100,:])
#print('ds[0,0,2,:]: ', ds[0,200,200,:])
#print('ds[0,1,0,:]: ', ds[0,100,150,:])
#print('ds[0,:,:,0] == ds[0,:,:,1]: ', ds[0,:,:,0] == ds[0,:,:,1])

#print('np.unique(np.flatten(ds)): ', np.unique(ds[1,:,:,].flatten()))
#print('np.unique(np.flatten(ds)): ', np.unique(ds[2,:,:,:], axis=2))
#print('np.unique(np.flatten(ds)): ', np.unique(ds[3,:,:,:], axis=2))

#arr = np.unique(ds[1,:,:,].flatten())
#print('arr: ', arr)
#print('tpye(arr): ', type(arr))

#res = map(id2myid, arr)
#print('res: ', list(res))
#print('type(res): ', type(res))
#resarr = map(id2myid, ds[1,:,:])
#print('resarr: ', list(resarr))

#print('ds[0,0:50,0:50,0]): ', ds[0,0:50,0:50,0])
#print('ds[0,0:50,0:50,1]): ', ds[0,0:50,0:50,1])
#print('ds[0,0:50,0:50,2]): ', ds[0,0:50,0:50,2])

#print('ds0new[0,:,:,0] == ds0new[0,:,:,1]: ', ds[0,:,:,0] == ds[0,:,:,1])
#print('ds0new[0,:,:,2] == ds0new[0,:,:,1]: ', ds[0,:,:,2] == ds[0,:,:,1])



#imgsem = ds[0,:,:]


#print('imgsem[0:200, 0:200]: ', imgsem[0:200,0:200])
#print('np.unique(imgsem.flatten()): ', np.unique(imgsem.flatten()))
#d_ = np.nonzero(imgsem[:,:,np.newaxis] == arr)
#print('type(d_[0]): ', type(d_[0]))
#imgsem[d_[0],d_[1]] = myids[d_[2]]
#print('np.unique(imgsem.flatten()): ', np.unique(imgsem.flatten()))

#net = Segnet()
#x = net.forward(torch.from_numpy(d.transpose((0,3,1,2))))

#print('x: ', x)
#print('x shape: ', x.shape)
#print('net: ', net)
#cv2.imshow('ds0: ', ds[0,:,:,:])
#cv2.waitKey(0)