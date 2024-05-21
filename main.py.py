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

train(dataset, model)

