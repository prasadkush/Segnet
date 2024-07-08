import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  
import numpy as np
from Exceptions import OutofIndexError
from torchvision.models import vgg16_bn
from torchvision.models import VGG16_BN_Weights
from time import time


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
        if output5 != 0:
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
        if output5 != 0:
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
        if output5 != 0:
            self.layer3c = nn.Sequential(
            nn.Conv2d(output5, output5, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output5), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.module_type = module_type
        self.kernel5 = output5 != 0

    def forward(self, x):
        #print ('DilationModule: ')
        #print('x shape: ', x.shape)
        out1a = self.layer1a(x)
        #print('1a shape: ', out1a.shape)
        out1b = self.layer1b(x)
        #print('1b shape: ', out1b.shape)
        if self.kernel5:
            out1c = self.layer1c(x)
        #print('1c shape: ', out1c.shape)
        if self.module_type == 1:
            out2a = self.layer2a(out1a)
            out2b = self.layer2b(out1b)
            if self.kernel5:
                out2c = self.layer2c(out1c)
        else:
            out2a = self.layer3a(out1a)
            out2b = self.layer3b(out1b)
            if self.kernel5:
                out2c = self.layer2c(out1c)
        #print('2a shape: ', out2a.shape)
        #print('2b shape: ', out2b.shape)
        #print('2c shape: ', out2c.shape)
        out3a = self.layer3a(out2a)
        #print('3a shape: ', out3a.shape)
        out3b = self.layer3b(out2b)
        #print('3b shape: ', out3b.shape)
        if self.kernel5:
            out3c = self.layer3c(out2c)
            #print('3c shape: ', out3c.shape)
            out = torch.concat((out3a, out3b, out3c), dim=1)
        else:
            out = torch.concat((out3a, out3b), dim=1)
        return out

class DilationModule2(nn.Module):
    def __init__(self, inputfeatures, outputfeatures, output1=64, output2=64, output3=64, output4=64, kernel1_size=7, padding1=3, kernel2_size=3, padding2=2, dilation2=2, kernel3_size=3, padding3=4, dilation3=4, kernel4_size=3, padding4=6, dilation4=6, layer1features=False, layer1featuressize=64, layer1outputfeatures=64):
        super(DilationModule2, self).__init__()
        self.layer1a = nn.Sequential(
            nn.Conv2d(inputfeatures, output1, kernel_size=kernel1_size, stride=1, padding=padding1),
            nn.BatchNorm2d(output1), nn.Dropout(p=0.30),
           #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer1b = nn.Sequential(
            nn.Conv2d(inputfeatures, output2, kernel_size=kernel2_size, stride=1, padding=padding2, dilation=dilation2),
            nn.BatchNorm2d(output2), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer1c = nn.Sequential(
            nn.Conv2d(inputfeatures, output3, kernel_size=kernel3_size, stride=1, padding=padding3, dilation=dilation3),
            nn.BatchNorm2d(output3), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer1d = nn.Sequential(
            nn.Conv2d(inputfeatures, output4, kernel_size=kernel4_size, stride=1, padding=padding4, dilation=dilation4),
            nn.BatchNorm2d(output4), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) 
        self.layer2a = nn.Sequential(
            nn.Conv2d(output1, output1, kernel_size=kernel1_size, stride=1, padding=padding1),
            nn.BatchNorm2d(output1), nn.Dropout(p=0.30),
           #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer2b = nn.Sequential(
            nn.Conv2d(output2, output2, kernel_size=kernel2_size, stride=1, padding=padding2, dilation=dilation2),
            nn.BatchNorm2d(output2), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer2c = nn.Sequential(
            nn.Conv2d(output3, output3, kernel_size=kernel3_size, stride=1, padding=padding3, dilation=dilation3),
            nn.BatchNorm2d(output3), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer2d = nn.Sequential(
            nn.Conv2d(output4, output4, kernel_size=kernel4_size, stride=1, padding=padding4, dilation=dilation4),
            nn.BatchNorm2d(output4), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer3a = nn.Sequential(
            nn.Conv2d(output1, output1, kernel_size=kernel1_size, stride=1, padding=padding1),
            nn.BatchNorm2d(output1), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer3b = nn.Sequential(
            nn.Conv2d(output2, output2, kernel_size=kernel2_size, stride=1, padding=padding2, dilation=dilation2),
            nn.BatchNorm2d(output2), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer3c = nn.Sequential(
            nn.Conv2d(output3, output3, kernel_size=kernel3_size, stride=1, padding=padding3, dilation=dilation3),
            nn.BatchNorm2d(output3), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer3d = nn.Sequential(
            nn.Conv2d(output4, output4, kernel_size=kernel4_size, stride=1, padding=padding4, dilation=dilation4),
            nn.BatchNorm2d(output4), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer4 = nn.Sequential(
            nn.Conv2d(output1+output2+output3+output4, outputfeatures, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outputfeatures), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) 
        self.layer1features = layer1features
        if self.layer1features == True:
            self.layer5 = nn.Sequential(
            nn.Conv2d(layer1featuressize, layer1outputfeatures, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) 


    def forward(self, x):
        #print ('DilationModule2: ')
        #print('x shape: ', x.shape)
        outa = self.layer1a(x)
        #print('1a shape: ', outa.shape)
        outb = self.layer1b(x)
        #print('1b shape: ', outb.shape)
        outc = self.layer1c(x)
        #print('1c shape: ', outc.shape)
        outd = self.layer1d(x)
        #print('1d shape: ', outd.shape)
        outa = self.layer2a(outa)
        #print('2a shape: ', outa.shape)
        outb = self.layer2b(outb)
        #print('2b shape: ', outb.shape)
        outc = self.layer2c(outc)
        #print('2c shape: ', outc.shape)
        outd = self.layer2d(outd)
        #print('2d shape: ', outd.shape)
        outa = self.layer3a(outa)
        #print('3a shape: ', outa.shape)
        outb = self.layer3b(outb)
        #print('3b shape: ', outb.shape)
        outc = self.layer3c(outc)
        #print('3c shape: ', outc.shape)
        outd = self.layer3d(outd)
        #print('3d shape: ', outd.shape)
        out = torch.concat((outa, outb, outc, outd), dim=1)
        out = self.layer4(out)
        #print('4 shape: ', out.shape)
        return out

class ConvLayer(nn.Module):
    def __init__(self, inputfeatures, outputfeatures, kernel_size=7, stride=1, padding=3, dilation=1, outputinter=64, layertype=1, droupout=False):
        super(ConvLayer, self).__init__()
        if droupout == False:
            self.layer1 = nn.Sequential(
            nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer2 = nn.Sequential(
            nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, outputfeatures, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputfeatures),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        else: 
            self.layer1 = nn.Sequential(
            nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer2 = nn.Sequential(
            nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, outputfeatures, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputfeatures), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))

        self.layer4 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer4b = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=False)
        self.layertype = layertype

    def forward(self, x):
        #print('ConvLayer: ')
        #print('x shape: ', x.shape)
        out = self.layer1(x)
        #print('out1 shape: ', out1.shape)
        if self.layertype == 1:
            out = self.layer3(out)
            #print('out2 shape: ', out2.shape)
            out, inds = self.layer4(out)
            #print('out3 shape: ', out3.shape)
            return out, inds
        elif self.layertype == 2:
            out = self.layer2(out)
            #print('out2 shape: ', out2.shape)
            out = self.layer3(out)
            #print('out3 shape: ', out3.shape)
            out, inds = self.layer4(out)
            #print('out4 shape: ', out4.shape)
            return out, inds
        elif self.layertype == 3:
            out = self.layer3(out)
            return out
        elif self.layertype == 4:
            out = self.layer3(out)
            #print('out2 shape: ', out2.shape)
            out = self.layer4b(out)
            #print('out3 shape: ', out3.shape)
            return out


class ClassifyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, intermediate=32):
        super(ClassifyBlock, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels, intermediate, kernel_size=1, stride=1, padding=0), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layera = nn.Conv2d(intermediate, out_channels, kernel_size=1, stride=1, padding=0)
        self.layerprob = nn.Softmax(dim=1)
        '''
        torch.nn.init.normal_(self.layer.weight, mean=0, std=1)
        torch.nn.init.normal_(self.layerprob.weight, mean=0, std=1)
        '''

    def forward(self, x):
        #print('ClassifyBlock: ')
        #print('x shape: ', x.shape)
        out = self.layer(x)   
        out = self.layera(out)
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
        # layertype = 1 --> 2 conv layers, layertype = 2 --> 3 conv layers, layertype = 3 --> 2 conv layers without unpooling
        self.layer1 = ConvLayer(3, 64, kernel_size=4, stride=1, padding=3, dilation=2, outputinter=64, layertype=4)
        self.layer2 = ConvLayer(64, 128, kernel_size=5, padding=2, outputinter=64, layertype=1)
        self.layer3 = ConvLayer(128, 128, kernel_size=kernel1_size, padding=3, outputinter=128, layertype=1)
        self.layer4 = DilationModule2(128, 256, output1=64, output2=64, output3=64, output4=64, kernel1_size=7, padding1=3, kernel2_size=3, padding2=2, dilation2=2, kernel3_size=3, padding3=4, dilation3=4, kernel4_size=3, padding4=6, dilation4=6)
        self.layer5 = DilationModule2(256, 256, output1=64, output2=64, output3=64, output4=64, kernel1_size=7, padding1=3, kernel2_size=3, padding2=2, dilation2=2, kernel3_size=3, padding3=4, dilation3=4, kernel4_size=3, padding4=6, dilation4=6)
        self.layer6 = DilationModule2(256, 256, output1=64, output2=64, output3=64, output4=64, kernel1_size=7, padding1=3, kernel2_size=3, padding2=2, dilation2=2, kernel3_size=3, padding3=4, dilation3=4, kernel4_size=3, padding4=6, dilation4=6)
        self.layer7 = DilationModule2(256, 256, output1=64, output2=64, output3=64, output4=64, kernel1_size=7, padding1=3, kernel2_size=3, padding2=2, dilation2=2, kernel3_size=3, padding3=4, dilation3=4, kernel4_size=3, padding4=6, dilation4=6)

        self.layer8 = nn.Sequential(nn.Conv2d(256, 128, 1, stride=1, padding=0), 
            nn.BatchNorm2d(128), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) 
        self.upsampleLayer = nn.Upsample(scale_factor=2,mode='bilinear')
        self.poollayer = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.unpoollayer = nn.MaxUnpool2d(kernel_size = 2, stride = 2)
        #self.layer = nn.Sequential(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(128), 
        #    nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None),
        #    nn.Conv2d(128, 128, 1, stride=1, padding=0), nn.Dropout(p=0.30), nn.BatchNorm2d(128), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) 
        #self.layer7 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), 
        #    nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None),
        #    nn.Conv2d(64, 64, 1, stride=1, padding=0), nn.Dropout(p=0.30), nn.BatchNorm2d(64), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        #self.layer7 = nn.Sequential(
        #    nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        #    nn.BatchNorm2d(128),
           #nn.ReLU())
        #    nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.layer9 = ConvLayer(128, 128, kernel_size=3, stride=1, padding=1,  outputinter=128, layertype=3, droupout=True)
        self.layer10 = ConvLayer(128, 64, kernel_size=3, stride=1, padding=1,  outputinter=64, layertype=3, droupout=True)
        #self.layer11 = ConvLayer(64, 64, kernel_size=3, stride=1, padding=1,  outputinter=64, layertype=3, droupout=True)
        #self.layer11b = ConvLayer(64, 64, kernel_size=1, stride=1, padding=0,  outputinter=64, layertype=3, droupout=True)
        self.layer11b = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64), nn.Dropout(p=0.30), 
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.ClassifyBlock = ClassifyBlock(64, out_channels)

    def forward(self, x):
        outsize0 = x.shape
        #start_time = time()
        out1 = self.layer1(x)
        #end_time = time()
        #print('time layer 1: ', (end_time - start_time)/8)
        #print('after layer1 out shape: ', out1.shape)
        outsize1 = out1.shape
        #start_time = time()
        out2, inds2 = self.layer2(out1)
        #end_time = time()
        #print('time layer 2: ', (end_time - start_time)/8)
        #print('after layer2 out shape: ', out2.shape)
        outsize2 = out2.shape
        #start_time = time()
        out3, inds3 = self.layer3(out2)
        #end_time = time()
        outsize3 = out3.shape
        #print('time layer 3: ', (end_time - start_time)/8)
        #print('after layer3 out shape: ', out3.shape)
        #start_time = time()
        out = self.layer4(out3)
        #end_time = time()
        #print('time layer 4: ', (end_time - start_time)/8)
        #print('after layer4 out shape: ', out.shape)
        #start_time = time()
        out = self.layer5(out)
        #end_time = time()
        #print('time layer 5: ', (end_time - start_time)/8)
        #print('after layer5 out shape: ', out.shape)
        #start_time = time()
        out = self.layer6(out)
        #end_time = time()
        #print('time layer 6: ', (end_time - start_time)/8)       
        #print('after layer6 out shape: ', out.shape) 
        #start_time = time()
        out = self.layer7(out)
        #end_time = time()
        #print('time layer 7: ', (end_time - start_time)/8)       
        #print('after layer7 out shape: ', out.shape) 
        #start_time = time()
        out = self.layer8(out) + out3
        #end_time = time()
        #print('time layer 8: ', (end_time - start_time)/8)       
        #print('after layer8 out shape: ', out.shape)
        #start_time = time()
        out = self.unpoollayer(out, inds3, output_size=outsize2)
        #end_time = time()
        #print('time unpool layer layer: ', (end_time - start_time)/8)
        #print('after unpool layer, out shape: ', out.shape)
        # 2 conv layers with 5 x 5 filters
        #start_time = time()
        out = self.layer9(out) + out2
        #end_time = time()
        #print('time layer 9: ', (end_time - start_time)/8)       
        #print('after layer9 out shape: ', out.shape)
        #end_time = time()
        #print('time layer 8: ', (end_time - start_time)/8)
        #start_time = time()
        out = self.unpoollayer(out, inds2, output_size=outsize1) 
        #end_time = time()
        #print('time unpool layer: ', (end_time - start_time)/8)
        #print('after unpool layer, out shape: ', out.shape)
        #out = self.layer6(out)
        #print('after layer6 out shape: ', out.shape)
        
        #start_time = time()
        out = self.layer10(out) + out1
        #end_time = time()
        #print('time layer 10: ', (end_time - start_time)/8)       
        #print('after layer10 out shape: ', out.shape) 
        #start_time = time()
        out = self.upsampleLayer(out)
        #end_time = time()
        #print('time upsampleLayer: ', (end_time - start_time)/8)       
        #print('after upsampleLayer: ', out.shape) 
        '''
        start_time = time()
        out = self.unpoollayer(out, inds1, output_size=outsize0)
        end_time = time()
        print('time unpool layer: ', (end_time - start_time)/8)
        print('after unpool layer, out shape: ', out.shape)
        start_time = time()
        out = self.layer11b(out)
        #out = self.upsampleLayer(out)
        end_time = time()
        print('time layer 11: ', (end_time - start_time)/8) 
        print('after layer11 out shape: ', out.shape)
        '''
        #start_time = time()
        out = self.ClassifyBlock(out)
        #end_time = time()
        #print('time layer ClassifyBlock: ', (end_time - start_time)/8) 
        #print('after ClassifyBlock out shape: ', out.shape)
        return out



weight_dict = {'Encoder.layer1.0.weight': 'features.0.weight', 'Encoder.layer1.0.bias': 'features.0.bias', 'Encoder.layer2.0.weight': 'features.3.weight', 'Encoder.layer2.0.bias': 'features.3.bias',
'Encoder.layer4.0.weight' : 'features.7.weight', 'Encoder.layer4.0.bias' : 'features.7.bias', 'Encoder.layer5.0.weight' : 'features.10.weight', 'Encoder.layer5.0.bias' : 'features.10.bias', 
'Encoder.layer7.0.weight' : 'features.14.weight', 'Encoder.layer7.0.bias' : 'features.14.bias', 'Encoder.layer8.0.weight' : 'features.17.weight', 'Encoder.layer8.0.bias' : 'features.17.bias',
'Encoder.layer9.0.weight' : 'features.20.weight', 'Encoder.layer9.0.bias' : 'features.20.bias', 'Encoder.layer11.0.weight' : 'features.24.weight', 'Encoder.layer11.0.bias' : 'features.24.bias',
'Encoder.layer12.0.weight' : 'features.27.weight', 'Encoder.layer12.0.bias' : 'features.27.bias', 'Encoder.layer13.0.weight' : 'features.30.weight', 'Encoder.layer13.0.bias' : 'features.30.bias',
'Encoder.layer15.0.weight' : 'features.34.weight', 'Encoder.layer15.0.bias' : 'features.34.bias', 'Encoder.layer16.0.weight' : 'features.34.weight', 'Encoder.layer16.0.bias' : 'features.34.bias',
'Encoder.layer17.0.weight' : 'features.37.weight', 'Encoder.layer17.0.bias' : 'features.37.bias'}
