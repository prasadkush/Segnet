import cv2
import torch
import torch.nn as nn
from torch import optim
from data import getDataset, get_inverse_transforms
from torch.utils.data import DataLoader, Dataset
from model import Encoder, Segnet
import matplotlib.pyplot as plt
import numpy as np
from config import datapath
import pickle
import pdb

def save_model(loss, path, epoch, model, optimizer):
    EPOCH = epoch
    PATH_ = path
    LOSS_ = loss

    torch.save({
        'epoch': EPOCH, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS_}, PATH_)

def compute_loss(dataset):
    batch_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size,shuffle=True)
    loaderiter = iter(data_loader)
    data = next(loaderiter)


def weights_init(m):
    classname = m.__class__.__name__
    print('initializing weights, classname: ', classname)
    if isinstance(m, nn.Conv2d):
        #torch.nn.init.normal_(m.weight, mean=0.2, std=1)
        #torch.nn.init.uniform_(m.weight)  
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        #torch.nn.init.normal_(m.weight, mean=0.2, std=1)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        #torch.nn.init.uniform_(m.weight)
        #torch.nn.init.zeros_(m.bias)
    


def train(dataset, model, epochs=10, batch_size=4, shuffle=True, testdataset=False, val_dataset=None):

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # TODO initialize this to be a Cross Entropy Classification loss.
    criterion = nn.CrossEntropyLoss()

    if val_dataset == None:
        val_dataset = getDataset(datapath)

    lr_initial = 0.001
    lr_new = 0.001

    model.apply(weights_init)
    print('after model.apply')
    optimizer = optim.Adam(model.parameters(), lr=lr_initial, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    print('model: ', model)
    loss = 0
    best_loss = 100000
    total_loss = 0
    training_loss_list = []

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}  | Requires_grad: {param.requires_grad} \n")

    #loaderiter = iter(loader)
    for e in range(epochs):
        print('epoch: ', e)
        if (e+1) % 2 == 0:
            lr_new = lr_initial*(1-e/epochs)
            for g in optimizer.param_groups:
                g['lr'] = lr_new
        total_loss = 0

        for i, data in enumerate(loader):

            print('i: ', i)
            d = data['image']
            ds = data['semantic']
            dimg = data['original']
            optimizer.zero_grad()
            x = model.forward(d)
            #dimg = torch.permute(dimg, (0,2,3,1))
            #dimg = dimg.numpy()
            #print('dimg shape: ', dimg.shape)
            #cv2.imshow('dimg[0,:,:,:]: ', dimg[0,:,:,:])
            #cv2.waitKey(0)
            #inv_trans = get_inverse_transforms()
            #dorig = inv_trans(d)
            #dorig = torch.permute(dorig, (0,2,3,1))
            #dorig = dorig.numpy()
            #print('dorig shape: ', dorig.shape)
            #cv2.imshow('dorig[0,:,:,:]: ', dorig[0,:,:,:])
            #cv2.waitKey(0)
            print('x shape: ', x.shape)
            #print('x: ', x)
            x_logits = torch.logit(x,eps=1e-7) 
            #print('x_logits.shape')
            output = criterion(x_logits,ds)
            print('output: ', output)

            #breakpoint()
            output.backward(retain_graph=False)
            optimizer.step()
            loss = output.item()
            total_loss = total_loss + batch_size*loss


        training_loss = total_loss/len(dataset)
        training_loss_list.append(training_loss)
        epoch_list = range(len(training_loss_list))
        if training_loss < best_loss:
            path = 'bestlosssegnetmodel.pt'
            save_model(training_loss, path, e, model, optimizer)
            best_loss = training_loss

        if best_loss != training_loss:
            path = 'latestsegnetmodel.pt'
            save_model(training_loss, path, e, model, optimizer)

        w1_en = model.Encoder.layer1[0].weight
        w2_en = model.Encoder.layer2[0].weight
        w4_en = model.Encoder.layer4[0].weight
        w5_en = model.Encoder.layer5[0].weight
        w14_decoder = model.Decoder.layer14[0].weight
        w15_decoder = model.Decoder.layer15[0].weight
        w18_decoder = model.Decoder.layer18[0].weight
        w19_decoder = model.Decoder.layer19[0].weight
        tensor_dict = {'w1_en' : w1_en, 'w2_en' : w2_en, 'w4_en' : w4_en, 'w5_en' : w5_en, 'w14_decoder' : w14_decoder, 'w15_decoder' : w15_decoder,
            'w18_decoder' : w18_decoder, 'w19_decoder' : w19_decoder}
        torch.save(tensor_dict, 'weights.pt')


        with open('results.pkl', 'wb') as f:
            pickle.dump(training_loss_list, f)
        plt.plot(epoch_list, training_loss_list)
        plt.xlabel('epochs')
        plt.ylabel('training loss')
        # giving a title to my graph
        plt.title('training loss vs epochs')
        plt.show()
