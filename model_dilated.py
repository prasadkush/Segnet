import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  
import numpy as np
from Exceptions import OutofIndexError
from torchvision.models import vgg16_bn
from torchvision.models import VGG16_BN_Weights


class DilationModule(nn.Module):
    def __init__(self, inputfeatures, module_type=1, output7=48, output3=16, output5=16, kernel1_size=7, padding1=3, kernel2_size=3, padding2=2, dilation2=1, kernel3_size=5, padding3=4, dilation3=1):
        super(DilationModule, self).__init__()
        self.layer1a = nn.Sequential(
            nn.Conv2d(inputfeatures, output7, kernel_size=kernel1_size, stride=1, padding=padding1),
            nn.BatchNorm2d(output7), nn.Dropout(p=0.30),
           #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer1b = nn.Sequential(
            nn.Conv2d(inputfeatures, output3, kernel_size=kernel2_size, stride=1, padding=padding2, dilation=dilation2),
            nn.BatchNorm2d(output3), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer1c = nn.Sequential(
            nn.Conv2d(inputfeatures, output5, kernel_size=kernel3_size, stride=1, padding=padding3, dilation=dilation3),
            nn.BatchNorm2d(output5), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer2a = nn.Sequential(
            nn.Conv2d(output7, output7, kernel_size=kernel1_size, stride=1, padding=padding1),
            nn.BatchNorm2d(output7), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) 
        self.layer2b = nn.Sequential(
            nn.Conv2d(output3, output3, kernel_size=kernel2_size, stride=1, padding=padding2, dilation=dilation2),
            nn.BatchNorm2d(output3), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer2c = nn.Sequential(
            nn.Conv2d(output5, output5, kernel_size=kernel3_size, stride=1, padding=padding3, dilation=dilation3),
            nn.BatchNorm2d(output5), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer3a = nn.Sequential(
            nn.Conv2d(output7, output7, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output7), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer3b = nn.Sequential(
            nn.Conv2d(output3, output3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output3), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer3c = nn.Sequential(
            nn.Conv2d(output5, output5, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output5), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.module_type = module_type

    def forward(self, x):
        #print ('DilationModule: ')
        #print('x shape: ', x.shape)
        out1a = self.layer1a(x)
        #print('1a shape: ', out1a.shape)
        out1b = self.layer1b(x)
        #print('1b shape: ', out1b.shape)
        out1c = self.layer1c(x)
        #print('1c shape: ', out1c.shape)
        if self.module_type == 1:
            out2a = self.layer2a(out1a)
            out2b = self.layer2b(out1b)
            out2c = self.layer2c(out1c)
        else:
            out2a = self.layer3a(out1a)
            out2b = self.layer2b(out1b)
            out2c = self.layer2c(out1c)
        #print('2a shape: ', out2a.shape)
        #print('2b shape: ', out2b.shape)
        #print('2c shape: ', out2c.shape)
        out3a = self.layer3a(out2a)
        #print('3a shape: ', out3a.shape)
        out3b = self.layer3b(out2b)
        #print('3b shape: ', out3b.shape)
        out3c = self.layer3c(out2c)
        #print('3c shape: ', out3c.shape)
        out = torch.concat((out3a, out3b, out3c), dim=1)
        return out


class ConvLayer(nn.Module):
    def __init__(self, inputfeatures, outputinter, kernel_size=7, stride=1, padding=3,  output=64, layertype=1):
        super(ConvLayer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer2 = nn.Sequential(
            nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(output),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer4 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=False)
        self.layertype = layertype

    def forward(self, x):
        #print('ConvLayer: ')
        #print('x shape: ', x.shape)
        out1 = self.layer1(x)
        #print('out1 shape: ', out1.shape)
        if self.layertype == 1:
            out2 = self.layer3(out1)
            #print('out2 shape: ', out2.shape)
            out3 = self.layer4(out2)
            #print('out3 shape: ', out3.shape)
            return out3
        else:
            out2 = self.layer2(out1)
            #print('out2 shape: ', out2.shape)
            out3 = self.layer3(out2)
            #print('out3 shape: ', out3.shape)
            out4 = self.layer4(out3)
            #print('out4 shape: ', out4.shape)
            return out4


class ClassifyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ClassifyBlock, self).__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.layerprob = nn.Softmax(dim=1)
        '''
        torch.nn.init.normal_(self.layer.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layerprob.weight, mean=0, std=1)
        '''

    def forward(self, x):
        #print('ClassifyBlock: ')
        #print('x shape: ', x.shape)
        out = self.layer(x)   
        #print('out shape: ', out.shape)
        #print('breakpoint 1:' )
        #breakpoint()
        #out = torch.permute(out, (0,3,1,2))
        #print('out shape: ', out.shape)
        out = self.layerprob(out)
        #print('out shape: ', out.shape)
        #print('out[0,:,0,10]: ', out[0,:,0:2,10])
        #print('torch.sum(out[0,:,0,10]): ', torch.sum(out[0,:,0,10]))
        return out

class SegmentationDilated(nn.Module):
    def __init__(self, kernel1_size=7, kernel2_size=3, kernel3_size=5, padding=3, out_channels=12):
        super(SegmentationDilated, self).__init__()
        self.layer1 = ConvLayer(3, 64, kernel_size=kernel1_size, output=64, layertype=1)
        self.layer2 = ConvLayer(64, 64, kernel_size=kernel1_size, output=128, layertype=1)
        self.layer3 = DilationModule(128, module_type=1, output7=64, output3=32, output5=32, kernel1_size=kernel1_size, padding1=3, kernel2_size=3, padding2=2, dilation2=2, kernel3_size=5, padding3=4, dilation3=2)
        self.layer4 = DilationModule(128, module_type=1, output7=64, output3=32, output5=32, kernel1_size=kernel1_size, padding1=3, kernel2_size=3, padding2=4, dilation2=4, kernel3_size=5, padding3=8, dilation3=4)
        self.layer5 = DilationModule(128, module_type=2, output7=64, output3=32, output5=32, kernel1_size=kernel1_size, padding1=3, kernel2_size=3, padding2=6, dilation2=6, kernel3_size=5, padding3=12, dilation3=6)
        self.upsampleLayer = nn.Upsample(scale_factor=4,mode='bilinear')
        self.layer6 = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), 
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None),
            nn.Conv2d(128, 128, 1, stride=1, padding=0), nn.Dropout(p=0.30), nn.BatchNorm2d(128), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) 
        self.layer7 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), 
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None),
            nn.Conv2d(64, 64, 1, stride=1, padding=0), nn.Dropout(p=0.30), nn.BatchNorm2d(64), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        #self.layer7 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3),
        #    nn.BatchNorm2d(128),
            #nn.ReLU())
        #    nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        #self.layer7 = nn.Sequential(
        #    nn.Conv2d(128, 64, kernel_size=7, stride=1, padding=3),
        #    nn.BatchNorm2d(64),
            #nn.ReLU())
        #    nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.ClassifyBlock = ClassifyBlock(64, out_channels)

    def forward(self, x):
        out1 = self.layer1(x)
        out = self.layer2(out1)
        out2 = self.layer3(out)
        out = self.layer4(out2)
        out = self.layer5(out) + out2
        #print('after layer5 out shape: ', out.shape)
        out = self.layer6(out) + torch.concat((out1, out1), dim=1)
        #out = self.layer6(out)
        #print('after layer6 out shape: ', out.shape)
        out = self.layer7(out) + self.upsampleLayer(out2[:,::2,:,:])
        #out = self.layer7(out) 
        #print('after layer7 out shape: ', out.shape)
        out = self.ClassifyBlock(out)
        return out



weight_dict = {'Encoder.layer1.0.weight': 'features.0.weight', 'Encoder.layer1.0.bias': 'features.0.bias', 'Encoder.layer2.0.weight': 'features.3.weight', 'Encoder.layer2.0.bias': 'features.3.bias',
'Encoder.layer4.0.weight' : 'features.7.weight', 'Encoder.layer4.0.bias' : 'features.7.bias', 'Encoder.layer5.0.weight' : 'features.10.weight', 'Encoder.layer5.0.bias' : 'features.10.bias', 
'Encoder.layer7.0.weight' : 'features.14.weight', 'Encoder.layer7.0.bias' : 'features.14.bias', 'Encoder.layer8.0.weight' : 'features.17.weight', 'Encoder.layer8.0.bias' : 'features.17.bias',
'Encoder.layer9.0.weight' : 'features.20.weight', 'Encoder.layer9.0.bias' : 'features.20.bias', 'Encoder.layer11.0.weight' : 'features.24.weight', 'Encoder.layer11.0.bias' : 'features.24.bias',
'Encoder.layer12.0.weight' : 'features.27.weight', 'Encoder.layer12.0.bias' : 'features.27.bias', 'Encoder.layer13.0.weight' : 'features.30.weight', 'Encoder.layer13.0.bias' : 'features.30.bias',
'Encoder.layer15.0.weight' : 'features.34.weight', 'Encoder.layer15.0.bias' : 'features.34.bias', 'Encoder.layer16.0.weight' : 'features.34.weight', 'Encoder.layer16.0.bias' : 'features.34.bias',
'Encoder.layer17.0.weight' : 'features.37.weight', 'Encoder.layer17.0.bias' : 'features.37.bias'}
