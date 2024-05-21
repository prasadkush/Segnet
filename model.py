import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  
import numpy as np
from Exceptions import OutofIndexError

class Encoder(nn.Module):

    def __init__(self, kernel_size=3, padding=1, encoder_pretrained=False):
        super(Encoder, self).__init__() 
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer3 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer6 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer10 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer14 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer15 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer16 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer17 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer18 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        '''
        torch.nn.init.normal_(self.layer1.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer2.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer4.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer5.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer7.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer8.weight, mean=0, std=1)  
        torch.nn.init.normal_(self.layer9.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer11.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer12.weight, mean=0, std=1)    
        torch.nn.init.normal_(self.layer13.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer15.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer16.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer17.weight, mean=0, std=1) 
        '''

    def forward(self, x):
        
        indsout = []
        outsize = []
        #print('x shape: ', x.shape)
        #print('Encoder breakpoint 1: ')
        #breakpoint()
        out = self.layer1(x)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer2(out)
        #print('out shape: ', out.shape)
        layershape = out.detach().numpy().shape
        outsize.append(out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out, inds = self.layer3(out)
        #inds = np.unravel_index(inds.numpy(), (layershape[2], layershape[3]))
        indsout.append(inds)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer4(out)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer5(out)
        layershape = out.detach().numpy().shape
        #print('out shape: ', out.shape)
        outsize.append(out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out, inds = self.layer6(out)
        #inds = np.unravel_index(inds.numpy(), (layershape[2], layershape[3]))
        indsout.append(inds)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer7(out)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer8(out)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer9(out)
        layershape = out.detach().numpy().shape
        #print('out shape: ', out.shape)
        outsize.append(out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out, inds = self.layer10(out)
        #inds = np.unravel_index(inds.numpy(), (layershape[2], layershape[3]))
        indsout.append(inds)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer11(out)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer12(out)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer13(out)
        #print('out shape: ', out.shape)
        layershape = out.detach().numpy().shape
        outsize.append(out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out, inds = self.layer14(out)
        #inds = np.unravel_index(inds.numpy(), (layershape[2], layershape[3]))
        indsout.append(inds)
        #print('out shape: ', out.shape)        
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer15(out)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer16(out)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer17(out)
        layershape = out.detach().numpy().shape
        #print('out shape: ', out.shape)
        outsize.append(out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out, inds = self.layer18(out)
        #inds = np.unravel_index(inds.numpy(), (layershape[2], layershape[3]))
        #print('Encoder breakpoint: ')
        #breakpoint()
        indsout.append(inds)
        #print('out shape: ', out.shape)
        return indsout, outsize, out


class Decoder(nn.Module):

    def __init__(self, kernel_size=3, padding=1, encoder_pretrained=False):
        super(Decoder, self).__init__()
        self.layer1 = nn.MaxUnpool2d(kernel_size = 2, stride = 2) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer5 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer9 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer13 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer15 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer16 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer17 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer19 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU())
        ''' 
        torch.nn.init.normal_(self.layer1.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer2.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer3.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer5.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer6.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer7.weight, mean=0, std=1)  
        torch.nn.init.normal_(self.layer9.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer10.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer11.weight, mean=0, std=1)    
        torch.nn.init.normal_(self.layer13.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer14.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer16.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layer17.weight, mean=0, std=1) 
        '''

    def forward(self, x, indices, outsize):
        l = len(indices)
        lout = len(outsize)
        l = l - 1
        #print('len indices: ', len(indices))
        #print('len outsize: ', lout)
        #print('type(indices[0]): ', type(indices[0]))
        #print('x shape: ', x.shape)
        out = 0
        if l >= 0:
            out = self.layer1(x, indices[l], output_size=outsize[l])
            l = l - 1
            #print('out shape: ', out.shape)
            #print('Decoder breakpoint: ')
            #breakpoint()
        else:
            raise OutofIndexError("Maxpool indices storage does not contain enough indices")  
        out = self.layer2(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer3(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer4(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        if l >= 0: 
            out = self.layer5(out, indices[l], output_size=outsize[l])
            l = l - 1
            #print('out shape: ', out.shape)
            #print('Decoder breakpoint: ')
            #breakpoint()
        else:
            raise OutofIndexError("Maxpool indices storage does not contain enough indices")
        out = self.layer6(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer7(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer8(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        if l >= 0:
            out = self.layer9(out, indices[l], output_size=outsize[l])
            l = l - 1
            #print('out shape: ', out.shape)
            #print('Decoder breakpoint: ')
            #breakpoint()
        else:
            raise OutofIndexError("Maxpool indices storage does not contain enough indices")
        out = self.layer10(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer11(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer12(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        if l >= 0:
            out = self.layer13(out, indices[l], output_size=outsize[l])
            l = l - 1
            #print('out shape: ', out.shape)
            #print('Decoder breakpoint: ')
            #breakpoint()
        else:
            raise OutofIndexError("Maxpool indices storage does not contain enough indices")
        out = self.layer14(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer15(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        if l >= 0:
            out = self.layer16(out)
            #print('out shape: ', out.shape)
            #print('Decoder breakpoint: ')
            #breakpoint()
            out = self.layer17(out, indices[l], output_size=outsize[l])
            #print('out shape: ', out.shape)
            #print('Decoder breakpoint: ')
            #breakpoint()
            l = l - 1
        else:
            raise OutofIndexError("Maxpool indices storage does not contain enough indices") 
        out = self.layer18(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer19(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()      
        return out        


class ClassifyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers1, hidden_layers2):
        super(ClassifyBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_channels, hidden_layers1), nn.ReLU(), 
        nn.Linear(hidden_layers1, hidden_layers2), nn.ReLU(), nn.Linear(hidden_layers2, out_channels))
        self.layerprob = nn.Softmax(dim=1)
        '''
        torch.nn.init.normal_(self.layer.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layerprob.weight, mean=0, std=1)
        '''

    def forward(self, x):
        #print('ClassifyBlock: ')
        #y = torch.clone(x)
        #print('x shape: ', x.shape)
        x = torch.permute(x, (0,2,3,1))
        #print('x shape: ', x.shape)
        out = self.layer(x)   
        #print('out shape: ', out.shape)
        #print('breakpoint 1:' )
        #breakpoint()
        out = torch.permute(out, (0,3,1,2))
        #print('out shape: ', out.shape)
        #print('breakpoint 2:' )
        #breakpoint()
        out = self.layerprob(out)
        #print('out shape: ', out.shape)
        #print('breakpoint 3:' )
        #breakpoint()
        return out

class Segnet(nn.Module):
    def __init__(self, kernel_size=7, padding=3, out_channels=16, encoder_pretrained=False):
        super(Segnet, self).__init__() 
        self.Encoder = Encoder(kernel_size, padding)
        self.Decoder = Decoder(kernel_size, padding)
        self.ClassifyBlock = ClassifyBlock(64,out_channels,200,200)

    def forward(self, x):
        #print('breakpoint before encoder: ')
        #breakpoint()
        indsout, outsize, out = self.Encoder(x)
        #print('breakpoint before decoder: ')
        #breakpoint()
        out = self.Decoder(out, indsout, outsize)
        #print('breakpoint before ClassifyBlock: ')
        #breakpoint()
        out = self.ClassifyBlock(out)
        #print('breakpoint after ClassifyBlock: ')
        #breakpoint()
        return out
