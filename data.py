from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch

import numpy as np  
import pandas as pd
from PIL import Image
from io import BytesIO
import random
from itertools import permutations
import os
import cv2

from labels import mylabels, Label, id2myid, id2label

def getmyids():
    myids = np.zeros((34,))
    for i in range(34):
        name = id2label[i].name
        if name in list(mylabels.keys()):
            myids[i] = mylabels[name]
        else:
            myids[i] = 15
    return myids

def load_zip_to_mem(zip_file, is_mono=True):
    """
    Function to load CLEVR-D data from the zip file.
    """
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    file_dict = {name.split('/')[1]: input_zip.read(name) for 
            name in input_zip.namelist() if '.png' in name}
    data = []
    for file_name in file_dict:
      #Only deal with right rgb images, all else via dict lookup
      if 'right' in file_name and 'CLEVR-D' not in file_name:
        rgb_right = file_dict[file_name]
        right_depth_name = file_name.replace('CLEVR','CLEVR-D')
        depth_right = file_dict[right_depth_name]
        if is_mono:
          data.append( (rgb_right, depth_right))
        else:
          rgb_left = file_dict[file_name.replace('right','left')]
          depth_left = file_dict[right_depth_name.replace('right','left')]
          data.append( (rgb_right,rgb_left, depth_right,depth_left))
    return data

def get_inverse_transforms():
    """
    Get inverse transforms to undo data normalization
    """
    inv_normalize_color = transforms.Normalize((-0.38399/0.32906, -0.39878/0.31968, -0.37933/0.3109),
    (1/0.32906, 1/0.31968, 1/0.3109)
    )
    #inv_normalize_depth = transforms.Normalize(
    #mean=[-0.480/0.295],
    #std=[1/0.295]
    #)

    return inv_normalize_color

def get_tensor_to_image_transforms():
    """
    Get transforms to go from Pytorch Tensors to PIL images that can be displayed
    """
    tensor_to_image = transforms.ToPILImage()
    inv_normalize_color, inv_normalize_depth = get_inverse_transforms()
    return (transforms.Compose([inv_normalize_color,tensor_to_image]),
            transforms.Compose([inv_normalize_depth,tensor_to_image]))

#mean:  [0.38399986 0.39878138 0.3793309 ]
#std:  [0.32906724 0.31968708 0.31093021]

def get_color_transform(dataset='kitti'):
    if dataset == 'kitti':
        color_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.38399, 0.39878, 0.37933), (0.32906, 0.31968, 0.3109)),
        ])
        return color_transform

class getDataset(Dataset):
    """
    The Dataset class 

    Arguments:
        data (int): list of tuples with data from the zip files
        is_mono (boolen): whether to return monocular or stereo data
        start_idx (int): start of index to use in data list  
        end_idx (int): end of i
    """
    def __init__(self, datapath, pct=1.0, train_val_split=1.0, dataset='kitti'):
        self.start_idx = 0
        self.color_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.38399, 0.39878, 0.37933), (0.32906, 0.31968, 0.3109)),
        ])
        self.ConvertImageDtype = transforms.ConvertImageDtype(torch.float)
        self.Normalize = transforms.Normalize((0.38399, 0.39878, 0.37933), (0.32906, 0.31968, 0.3109))
        self.data = []
        self.data_orig = []
        self.images = []
        myids = getmyids()
        imgdir = os.path.join(datapath, 'image_2')
        for filename in os.listdir(imgdir):
            filepath = os.path.join(imgdir, filename)
            img = cv2.imread(filepath)
            imgnew = cv2.resize(img, (480, 360), interpolation=cv2.INTER_NEAREST)
            imgnew = imgnew.transpose(2,0,1)
            t = torch.from_numpy(imgnew)
            t = self.ConvertImageDtype(t)
            self.data_orig.append(t)
            t = self.Normalize(t)
            self.data.append(t)
        self.end_idx = int(pct * train_val_split * len(self.data))
        self.images = self.data[0:self.end_idx]
        self.semantic_images = []
        i = 0
        self.semantic_data = []
        self.samples = []
        imgdir = os.path.join(datapath, 'semantic')
        myids = 0
        if dataset == 'kitti':
            myids = getmyids()
        print('myids: ', myids)
        for filename in os.listdir(imgdir):
            i = i + 1
            filepath = os.path.join(imgdir, filename)
            img = cv2.imread(filepath)
            imgnew = cv2.resize(img, (480, 360), interpolation=cv2.INTER_NEAREST)
            imgsem = imgnew[:,:,0]
            arr = np.arange(34)
            d = np.nonzero(imgsem[:,:,np.newaxis] == arr)
            imgsem[d[0],d[1]] = myids[d[2]]
            t = torch.from_numpy(imgsem)
            t = t.long()
            self.semantic_data.append(t)
            sample = {'image': self.images[i-1], 'semantic': self.semantic_data[i-1], 'original': self.data_orig[i-1]}
            self.samples.append(sample)
            if i == self.end_idx:
                break
        print('self.end_idx: ', self.end_idx)




    def __getitem__(self, idx):
        return self.samples[idx] # TODO 

    def __len__(self):
        return len(self.samples) # TODO 

def get_data_loaders(path,  
                    batch_size=1, 
                    train_val_split=1.0, 
                    pct_dataset=1.0):
    """
    The function to return the Pytorch Dataloader class to iterate through
    the dataset. 

    Arguments:
        is_mono (boolen): whether to return monocular or stereo data
        batch_size (int): batch size for both training and testing 
        train_test_split (float): ratio of data from training to testing
        pct_dataset (float): percent of dataset to use 
    """
    training_dataset = SegnetDataset(path, pct, train_val_split) # TODO 
    #testing_dataset = DepthDatasetMemory(data, is_mono, test_start_idx, test_end_idx) # TODO 

    #return (DataLoader(training_dataset, batch_size, shuffle=True, pin_memory=True),
    #        DataLoader(testing_dataset, batch_size, shuffle=False, pin_memory=True))

    return DataLoader(training_dataset, batch_size, shuffle=True, pin_memory=True)


