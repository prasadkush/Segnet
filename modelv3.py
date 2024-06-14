import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  
import numpy as np
from Exceptions import OutofIndexError
from torchvision.models import vgg16_bn
from torchvision.models import VGG16_BN_Weights


class Encoder(nn.Module):

    def __init__(self, kernel_size=3, padding=1, encoder_pretrained=False):
        super(Encoder, self).__init__() 
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(64),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(64),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer3 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer6 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer10 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer14 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer15 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer16 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer17 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
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
        outmap = {}
        #print('x shape: ', x.shape)
        #print('Encoder breakpoint 1: ')
        #breakpoint()
        out = self.layer1(x)
        #print('out shape: ', out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out = self.layer2(out)
        #print('out shape: ', out.shape)
        #layershape = out.detach().numpy().shape
        outsize.append(out.shape)
        #outmap.append(out)
        outmap['layer2'] = out
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
        #layershape = out.detach().numpy().shape
        #print('out shape: ', out.shape)
        outsize.append(out.shape)
        #outmap.append(out)
        outmap['layer5'] = out
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
        #layershape = out.detach().numpy().shape
        #print('out shape: ', out.shape)
        outsize.append(out.shape)
        #outmap.append(out)
        outmap['layer9'] = out
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
        #layershape = out.detach().numpy().shape
        outsize.append(out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out, inds = self.layer14(out)
        #outmaps.append(out)
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
        #layershape = out.detach().numpy().shape
        #print('out shape: ', out.shape)
        outsize.append(out.shape)
        #print('Encoder breakpoint: ')
        #breakpoint()
        out, inds = self.layer18(out)
        #inds = np.unravel_index(inds.numpy(), (layershape[2], layershape[3]))
        #print('Encoder breakpoint: ')
        #breakpoint()
        indsout.append(inds)
        #outmaps.append(out)
        #print('out shape: ', out.shape)
        return indsout, outsize, out, outmap


class Decoder(nn.Module):

    def __init__(self, kernel_size=3, padding=1, encoder_pretrained=False):
        super(Decoder, self).__init__()
        self.layer1 = nn.MaxUnpool2d(kernel_size = 2, stride = 2) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer5 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer9 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding), nn.Dropout(p=0.30),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer12 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding), nn.Dropout(p=0.30),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer13 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding), nn.Dropout(p=0.30),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer15 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=padding), nn.Dropout(p=0.30),
            nn.BatchNorm2d(128),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer16 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0), nn.Dropout(p=0.30),
            nn.BatchNorm2d(64),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer17 = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=padding), nn.Dropout(p=0.30),
            nn.BatchNorm2d(64),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer19 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernel_size, stride=1, padding=padding), 
            nn.BatchNorm2d(64),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.upsampleLayer = nn.Upsample(scale_factor=2,mode='bilinear')
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

    def forward(self, x, indices, outsize, outmap):
        l = len(indices)
        lout = len(outsize)
        l = l - 1
        #print('len indices: ', len(indices))
        #print('len outsize: ', lout)
        #print('type(indices[0]): ', type(indices[0]))
        #print('x shape: ', x.shape)
        out = 0
        if l >= 0:
            #out1up = self.upsampleLayer(x)
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
            #out5up = self.upsampleLayer(out)
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
            out9up = self.upsampleLayer(out)
            #print('out9up shape: ', out9up.shape)
            out = self.layer9(out, indices[l], output_size=outsize[l])
            l = l - 1
            #print('out9 shape: ', out.shape)
            #print('Decoder breakpoint: ')
            #breakpoint()
        else:
            raise OutofIndexError("Maxpool indices storage does not contain enough indices")
        out = self.layer10(out) + outmap['layer9']
        #print('outmap[layer9] shape: ', outmap['layer9'].shape)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer11(out)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer12(out) + out9up
        #print('out12 shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        if l >= 0:
            out13up = self.upsampleLayer(out)
            #print('out13up shape: ', out13up.shape)
            out = self.layer13(out, indices[l], output_size=outsize[l])
            l = l - 1
            #print('out13 shape: ', out.shape)
            #print('Decoder breakpoint: ')
            #breakpoint()
        else:
            raise OutofIndexError("Maxpool indices storage does not contain enough indices")
        out = self.layer14(out) + outmap['layer5']
        #print('outmap[layer5] shape: ', outmap['layer5'].shape)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer15(out) + out13up
        #print('out15 shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        if l >= 0:
            out = self.layer16(out)
            #print('out shape: ', out.shape)
            #print('Decoder breakpoint: ')
            #breakpoint()
            out17up = self.upsampleLayer(out)
            #print('out17up shape: ', out17up.shape)
            out = self.layer17(out, indices[l], output_size=outsize[l])
            #print('out17 shape: ', out.shape)
            #print('Decoder breakpoint: ')
            #breakpoint()
            l = l - 1
        else:
            raise OutofIndexError("Maxpool indices storage does not contain enough indices") 
        out = self.layer18(out) + outmap['layer2']
        #print('outmap[layer2] shape: ', outmap['layer2'].shape)
        #print('out shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()
        out = self.layer19(out) + out17up
        #print('out19 shape: ', out.shape)
        #print('Decoder breakpoint: ')
        #breakpoint()      
        return out        


class ClassifyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers1, hidden_layers2):
        super(ClassifyBlock, self).__init__()
        #self.layer = nn.Sequential(nn.Linear(in_channels, hidden_layers1), nn.ReLU(), 
        #nn.Linear(hidden_layers1, hidden_layers2), nn.ReLU(), nn.Linear(hidden_layers2, out_channels))
        #self.layer = nn.Sequential(nn.Linear(in_channels, hidden_layers1), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None), 
        #nn.Linear(hidden_layers1, hidden_layers2), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None), nn.Linear(hidden_layers2, out_channels))
        self.layer = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)
        self.layerprob = nn.Softmax(dim=1)
        '''
        torch.nn.init.normal_(self.layer.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layerprob.weight, mean=0, std=1)
        '''

    def forward(self, x):
        #print('ClassifyBlock: ')
        #y = torch.clone(x)
        #print('x shape: ', x.shape)
        #x = torch.permute(x, (0,2,3,1))
        #print('x shape: ', x.shape)
        out = self.layer(x)   
        #print('out shape: ', out.shape)
        #print('breakpoint 1:' )
        #breakpoint()
        #out = torch.permute(out, (0,3,1,2))
        #print('out shape: ', out.shape)
        #print('breakpoint 2:' )
        #breakpoint()
        out = self.layerprob(out)
        #print('out shape: ', out.shape)
        #print('out[0,:,0,10]: ', out[0,:,0:2,10])
        #print('torch.sum(out[0,:,0,10]): ', torch.sum(out[0,:,0,10]))
        #print('breakpoint 3:' )
        #breakpoint()
        return out

class Segnet(nn.Module):
    def __init__(self, kernel_size=7, padding=3, out_channels=16, encoder_pretrained=False):
        super(Segnet, self).__init__() 
        self.Encoder = Encoder(kernel_size, padding)
        self.Decoder = Decoder(kernel_size, padding)
        self.ClassifyBlock = ClassifyBlock(64,out_channels,200,200)
        self.upsamplefromvgg = torch.nn.Upsample(size=(7,7), mode='bilinear')

    def forward(self, x):
        #print('breakpoint before encoder: ')
        #breakpoint()
        indsout, outsize, out, outmap = self.Encoder(x)
        #print('breakpoint before decoder: ')
        #breakpoint()
        out = self.Decoder(out, indsout, outsize, outmap)
        #print('breakpoint before ClassifyBlock: ')
        #breakpoint()
        out = self.ClassifyBlock(out)
        #print('breakpoint after ClassifyBlock: ')
        #breakpoint()
        return out

    def initialize_Encoder(self):
        model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        self.Encoder.layer1[0].weight.data.copy_(self.upsamplefromvgg(model.features[0].weight.data))
        self.Encoder.layer1[0].bias.data.copy_(model.features[0].bias.data)
        self.Encoder.layer2[0].weight.data.copy_(self.upsamplefromvgg(model.features[3].weight.data))
        self.Encoder.layer2[0].bias.data.copy_(model.features[3].bias.data)
        self.Encoder.layer4[0].weight.data.copy_(self.upsamplefromvgg(model.features[7].weight.data))
        self.Encoder.layer4[0].bias.data.copy_(model.features[7].bias.data)
        self.Encoder.layer5[0].weight.data.copy_(self.upsamplefromvgg(model.features[10].weight.data))
        self.Encoder.layer5[0].bias.data.copy_(model.features[10].bias.data)
        self.Encoder.layer7[0].weight.data.copy_(self.upsamplefromvgg(model.features[14].weight.data[0::2,:,:,:]))
        self.Encoder.layer7[0].bias.data.copy_(model.features[14].bias.data[0::2])
        self.Encoder.layer8[0].weight.data.copy_(self.upsamplefromvgg(model.features[17].weight.data[0::2,0::2,:,:]))
        self.Encoder.layer8[0].bias.data.copy_(model.features[17].bias.data[0::2])
        self.Encoder.layer9[0].weight.data.copy_(self.upsamplefromvgg(model.features[20].weight.data[0::2,0::2,:,:]))
        self.Encoder.layer9[0].bias.data.copy_(model.features[20].bias.data[0::2])
        self.Encoder.layer11[0].weight.data.copy_(self.upsamplefromvgg(model.features[24].weight.data[0::4,0::2,:,:]))
        self.Encoder.layer11[0].bias.data.copy_(model.features[24].bias.data[0::4])
        self.Encoder.layer12[0].weight.data.copy_(self.upsamplefromvgg(model.features[27].weight.data[0::4,0::4,:,:]))
        self.Encoder.layer12[0].bias.data.copy_(model.features[27].bias.data[0::4])
        self.Encoder.layer13[0].weight.data.copy_(self.upsamplefromvgg(model.features[30].weight.data[0::4,0::4,:,:]))
        self.Encoder.layer13[0].bias.data.copy_(model.features[30].bias.data[0::4])
        self.Encoder.layer15[0].weight.data.copy_(self.upsamplefromvgg(model.features[34].weight.data[0::4,0::4,:,:]))
        self.Encoder.layer15[0].bias.data.copy_(model.features[34].bias.data[0::4])
        self.Encoder.layer16[0].weight.data.copy_(self.upsamplefromvgg(model.features[37].weight.data[0::4,0::4,:,:]))
        self.Encoder.layer16[0].bias.data.copy_(model.features[37].bias.data[0::4])
        self.Encoder.layer17[0].weight.data.copy_(self.upsamplefromvgg(model.features[40].weight.data[0::4,0::4,:,:]))
        self.Encoder.layer17[0].bias.data.copy_(model.features[40].bias.data[0::4])


weight_dict = {'Encoder.layer1.0.weight': 'features.0.weight', 'Encoder.layer1.0.bias': 'features.0.bias', 'Encoder.layer2.0.weight': 'features.3.weight', 'Encoder.layer2.0.bias': 'features.3.bias',
'Encoder.layer4.0.weight' : 'features.7.weight', 'Encoder.layer4.0.bias' : 'features.7.bias', 'Encoder.layer5.0.weight' : 'features.10.weight', 'Encoder.layer5.0.bias' : 'features.10.bias', 
'Encoder.layer7.0.weight' : 'features.14.weight', 'Encoder.layer7.0.bias' : 'features.14.bias', 'Encoder.layer8.0.weight' : 'features.17.weight', 'Encoder.layer8.0.bias' : 'features.17.bias',
'Encoder.layer9.0.weight' : 'features.20.weight', 'Encoder.layer9.0.bias' : 'features.20.bias', 'Encoder.layer11.0.weight' : 'features.24.weight', 'Encoder.layer11.0.bias' : 'features.24.bias',
'Encoder.layer12.0.weight' : 'features.27.weight', 'Encoder.layer12.0.bias' : 'features.27.bias', 'Encoder.layer13.0.weight' : 'features.30.weight', 'Encoder.layer13.0.bias' : 'features.30.bias',
'Encoder.layer15.0.weight' : 'features.34.weight', 'Encoder.layer15.0.bias' : 'features.34.bias', 'Encoder.layer16.0.weight' : 'features.34.weight', 'Encoder.layer16.0.bias' : 'features.34.bias',
'Encoder.layer17.0.weight' : 'features.37.weight', 'Encoder.layer17.0.bias' : 'features.37.bias'}
