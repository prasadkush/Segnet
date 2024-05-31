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
from predict import predict_single_image
from Exceptions import ModelPathrequiredError
from torchvision.models import vgg16_bn

weight_dict = {}

def save_model(loss, path, epoch, model, optimizer, lr_schedule, lr_milestones):
    EPOCH = epoch
    PATH_ = path
    LOSS_ = loss

    torch.save({
        'epoch': EPOCH, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS_, 'lr_schedule': lr_schedule, 'lr_milestones': lr_milestones}, PATH_)

def compute_loss(dataset):
    batch_size = len(dataset)
    data_loader = DataLoader(dataset, batch_size,shuffle=True)
    loaderiter = iter(data_loader)
    data = next(loaderiter)


def weights_init(m):
    #print('m: ', m)
    print('m.__class__: ', m.__class__)
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
    


def train(dataset, model, epochs=35, batch_size=4, shuffle=True, testdataset=False, val_dataset=None, modelpath=None, bestmodelpath=None, resume_training=False, useWeights=False, resultsdir=None, pretrained_encoder=True,
    layer_wise_training=True):

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # TODO initialize this to be a Cross Entropy Classification loss.
    if useWeights == True:
        weights = dataset.getWeights()
        weights = weights.to(torch.float)
        print('weights: ', weights)
        criterion = nn.CrossEntropyLoss(weight=weights)
        #criterion = nn.NLLLoss(weight=weights,reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.NLLLoss(reduction='none')


    if val_dataset == None:
        val_dataset = getDataset(datapath)

    lr_initial = 0.001
    lr_new = 0.001

    if resume_training == False and pretrained_encoder == False:
        model.apply(weights_init)
    elif resume_training == False and pretrained_encoder == True:
        model.initialize_Encoder()
        model.Decoder.apply(weights_init)
        model.ClassifyBlock.apply(weights_init)
    print('after model.apply')
    #optimizer = optim.Adam(model.parameters(), lr=lr_initial, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    optimizer = optim.SGD(model.parameters(), lr=lr_initial,  momentum=0.9)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    batch_size_val = 4
    loader_val = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True)
    loaderiter_val = iter(loader_val)
    num_val = 0

    start_epoch = 0
    loss = 0
    best_loss = 100000
    total_loss = 0
    training_loss_list = []
    mean_list = []

    if resume_training == True and modelpath != None:
        checkpoint = torch.load(modelpath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        if bestmodelpath != None:
            checkpoint = torch.load(bestmodelpath)
            best_loss = checkpoint['loss']
        else:
            best_loss = checkpoint['loss']
        start_epoch = epoch + 1

        with open('results/trial1/training_loss_list.pkl', 'rb') as f:
            training_loss_list = pickle.load(f)
        with open('results/trial1/mean_list.pkl', 'rb') as f:
            mean_list = pickle.load(f)
    elif resume_training == True and modelpath == None:
        raise ModelPathrequiredError("Provide Model path if resume_training is set to True")


    layers = ['Decoder.layer2', 'Decoder.layer3', 'Decoder.layer4']
    if layer_wise_training == True:
        for name, param in model.named_parameters():
            ind = name.index('.')
            ind2 = name.find('.', ind + 1)
            layername = name[0:ind2]
            if layername not in layers and param.requires_grad == True:
                param.requires_grad = False
            print('name: ', name, ' layername: ', layername, ' requires_grad: ', param.requires_grad)
        #print('Decoder parameters: ')
        #for name, param in model.Decoder.named_parameters():
        #    ind = name.index('.')
        #    layername = name[0:ind]
        #    if layername not in layers and param.requires_grad == True:
        #            param.requires_grad = False
        #print('name: ', name, ' layername: ', layername, ' requires_grad: ', param.requires_grad)

    #print('model: ', model)
#    for name, param in model.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}  | Requires_grad: {param.requires_grad} \n")
#         print(f"Layer: {name} | Size: {param.size()} \n")

    print('resume_training: ', resume_training)
    print('start_epoch: ', start_epoch)
    print('best_loss: ', best_loss)

    epochs = 24
    #lr_schedule = [lr_initial, lr_initial/2, lr_initial/5, lr_initial/10]
    #lr_milestones = [10, 20, 30, epochs]
    lr_schedule = [lr_initial, lr_initial/10]
    lr_milestones = [12, epochs]
    #loaderiter = iter(loader)
    ind_sch = 0
    for e in range(start_epoch, epochs):
        print('epoch: ', e)
        '''
        if e >= 10 and e < 20:
            #lr_new = lr_initial*(1-e/epochs)
            lr_new = lr_initial/2
            for g in optimizer.param_groups:
                g['lr'] = lr_new
        elif e >= 20 and e < 30:
            lr_new = lr_initial/5
            for g in optimizer.param_groups:
                g['lr'] = lr_new
        elif e >= 30:
            lr_new = lr_initial/10
            for g in optimizer.param_groups:
                g['lr'] = lr_new
        '''
        if e in lr_milestones:
            ind_sch += 1
            lr_new = lr_schedule[ind_sch]
            for g in optimizer.param_groups:
                g['lr'] = lr_new

        total_loss = 0
        model.train()
        for i, data in enumerate(loader):

            print('epoch: ', e, ' i: ', i)
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
            #output = criterion(x,ds)
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
            #path = 'results/trial0/bestlosssegnetmodelnew.pt'
            path = resultsdir + '/bestlosssegnetmodelnew.pt'
            save_model(training_loss, path, e, model, optimizer, lr_schedule, lr_milestones)
            best_loss = training_loss

        if best_loss != training_loss:
            #path = 'results/trial0/latestsegnetmodelnew.pt'
            path = resultsdir + '/latestsegnetmodelnew.pt'
            save_model(training_loss, path, e, model, optimizer, lr_schedule, lr_milestones)

        with open(resultsdir + '/training_loss_list.pkl', 'wb') as f:
            pickle.dump(training_loss_list, f)
        
        '''
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
        torch.save(tensor_dict, resultsdir + '/weights_e' + str(e) + '.pt')
        '''
        tensor_dict = {}
        if layer_wise_training == True:
            for name, param in model.named_parameters():
                ind = name.index('.')
                ind2 = name.find('.', ind + 1)
                layername = name[0:ind2]
            if layername in layers and param.requires_grad == True:
                tensor_dict[name] = param
        torch.save(tensor_dict, resultsdir + '/params_e' + str(e) + '.pt')

        if num_val >= 200:
            data_val = next(cycle(loaderiter_val))
        else:
            data_val = next(loaderiter_val)
        num_val = num_val + batch_size_val
        img_val = data_val['image']
        imgs_val = data_val['semantic']
        imgorig_val = data_val['original']
        with torch.no_grad():
            mean = predict_single_image(img_val, imgs_val, imgorig_val, model=model, imgdir=resultsdir + '/imgs', epoch=e)
        print('mean: ', mean)
        mean_list.append(mean)
        with open(resultsdir + '/mean_list.pkl', 'wb') as f:
            pickle.dump(mean_list, f)
        #plt.plot(epoch_list, training_loss_list)
        #plt.xlabel('epochs')
        #plt.ylabel('training loss')
        # giving a title to my graph
        #plt.title('training loss vs epochs')
        #plt.show()
