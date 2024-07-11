import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  
import numpy as np
from Exceptions import OutofIndexError
from torchvision.models import vgg16_bn
from torchvision.models import VGG16_BN_Weights
from time import time



class PatchEmbeddingLayer(nn.Module):
    def __init__(self, patch_size, in_channels, embedding_dims):
        super(PatchEmbeddingLayer, self).__init__()
        self.ConvLayer = nn.Conv2d(in_channels, embedding_dims, kernel_size=patch_size, stride=patch_size)
        self.flattenLayer = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        out = self.ConvLayer(x)
        out = self.flattenLayer(out)
        out = out.permute((0,2,1))
        return out

class MLPblock(nn.Module):
    def __init__(self, embedding_dims, hidden_dims):
        super(MLPblock, self).__init__()
        self.embedding_dims = embedding_dims
        self.hidden_dims = hidden_dims
        self.layerNorm = nn.LayerNorm(embedding_dims)
        self.mlplayer = nn.Sequential(nn.Linear(in_features = embedding_dims, out_features = hidden_dims), nn.GELU(),
            nn.Dropout(p=0.30), nn.Linear(in_features = hidden_dims, out_features = embedding_dims), nn.Dropout(p=0.30))

    def forward(self, x):
        out = self.layerNorm(x)
        out = self.mlplayer(out) + out
        return out


class MultiheadSelfAttentionblock(nn.Module):
    def __init__(self, embedding_dims, num_heads):
        super(MultiheadSelfAttentionblock, self).__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.layernorm = nn.LayerNorm(embedding_dims)
        self.multiheadattention = nn.MultiheadAttention(embedding_dims, num_heads, batch_first=True)

    def forward(self, x):
        out = self.layernorm(x)
        out, _ = self.multiheadattention(out, out, out, need_weights=False)
        #print('out shape after multiheadattention: ', out.shape)
        out = out + x
        return out

class Transformerblock(nn.Module):
    def __init__(self, embedding_dims, hidden_dims, num_heads):
        super(Transformerblock, self).__init__()
        self.msablock = MultiheadSelfAttentionblock(embedding_dims, num_heads)
        self.mlpblock = MLPblock(embedding_dims, hidden_dims)

    def forward(self, x):
        out = self.msablock(x)
        out = self.mlpblock(out)
        return out

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


class ConvLayer(nn.Module):
    def __init__(self, inputfeatures, outputinter, kernel_size=7, stride=1, padding=3, dilation=1, output=64, layertype=1, droupout=False):
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
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output),
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
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))

        self.layer4 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer5 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=False)
        self.layertype = layertype

    def forward(self, x):
        #print('ConvLayer: ')
        #print('x shape: ', x.shape)
        out1 = self.layer1(x)
        #print('out1 shape: ', out1.shape)
        if self.layertype == 1:
            out1 = self.layer3(out1)
            #print('out2 shape: ', out2.shape)
            out1, inds = self.layer4(out1)
            #print('out3 shape: ', out3.shape)
            return out1, inds
        elif self.layertype == 2:
            out1 = self.layer2(out1)
            #print('out2 shape: ', out2.shape)
            out1 = self.layer3(out1)
            #print('out3 shape: ', out3.shape)
            out1, inds = self.layer4(out1)
            #print('out4 shape: ', out4.shape)
            return out1, inds
        elif self.layertype == 3:
            out1 = self.layer3(out1)
            return out1
        elif self.layertype == 4:
            out1 = self.layer3(out1)
            #print('out2 shape: ', out2.shape)
            out1 = self.layer5(out1)
            #print('out3 shape: ', out3.shape)
            return out1


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
        self.layer1 = ConvLayer(3, 64, kernel_size=4, padding=3, dilation=2, output=64, layertype=4)
        self.layer2 = ConvLayer(64, 64, kernel_size=7, padding=3, output=128, layertype=1)
        self.layer3 = ConvLayer(128, 128, kernel_size=kernel1_size, output=256, layertype=1)
        self.layer4 = DilationModule(256, module_type=1, output7=128, output3=128, output5=0, kernel1_size=kernel1_size, padding1=3, kernel2_size=3, padding2=2, dilation2=2, kernel3_size=5, padding3=4, dilation3=2)
        self.layer5 = DilationModule(256, module_type=1, output7=128, output3=128, output5=0, kernel1_size=kernel1_size, padding1=3, kernel2_size=3, padding2=4, dilation2=4, kernel3_size=5, padding3=8, dilation3=4)
        self.layer6 = DilationModule(256, module_type=1, output7=128, output3=128, output5=0, kernel1_size=kernel1_size, padding1=3, kernel2_size=3, padding2=6, dilation2=6, kernel3_size=5, padding3=12, dilation3=6)
        self.layer7 = DilationModule(256, module_type=1, output7=128, output3=128, output5=0, kernel1_size=kernel1_size, padding1=3, kernel2_size=3, padding2=8, dilation2=8, kernel3_size=5, padding3=12, dilation3=8)
        self.embeddinglayer = PatchEmbeddingLayer(3, 256, 512)
        self.layertransformer = Transformerblock(512, 1024, 16)
        self.layertransformer2 = Transformerblock(512, 1024, 16)
        self.layertransup = nn.Sequential(nn.Conv2d(512, 256, 1, stride=1, padding=0), 
            nn.BatchNorm2d(256), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) 
        self.layer8 = nn.Sequential(nn.Conv2d(256, 256, 1, stride=1, padding=0), 
            nn.BatchNorm2d(256), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) 
        self.upsampleLayer = nn.Upsample(scale_factor=2,mode='bilinear')
        self.upsampleLayer2 = nn.Upsample(scale_factor=3,mode='bilinear')
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
        self.layer9 = ConvLayer(256, 256, kernel_size=3, stride=1, padding=1,  output=128, layertype=3, droupout=True)
        self.layer10 = ConvLayer(128, 128, kernel_size=3, stride=1, padding=1,  output=64, layertype=3, droupout=True)
        self.layer11 = ConvLayer(64, 64, kernel_size=3, stride=1, padding=1,  output=64, layertype=3, droupout=True)
        #self.layerEA1 = nn.Sequential(nn.Conv2d(256, 256, 1, stride=1, padding=0), 
        #    nn.BatchNorm2d(256), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        #self.layerEA2 = nn.Sequential(nn.Conv2d(128, 128, 1, stride=1, padding=0), 
        #    nn.BatchNorm2d(128), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        #self.layerEA3 = nn.Sequential(nn.Conv2d(64, 64, 1, stride=1, padding=0), 
        #    nn.BatchNorm2d(64), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        self.ClassifyBlock = ClassifyBlock(64, out_channels)

    def forward(self, x):
        outsize0 = x.shape
        #start_time = time()
        out1 = self.layer1(x)
        #end_time = time()
        #print('time layer 1: ', (end_time - start_time)/out1.shape[0])
        #print('after layer1 out shape: ', out1.shape)
        outsize1 = out1.shape
        #start_time = time()
        out2, inds2 = self.layer2(out1)
        #end_time = time()
        #print('time layer 2: ', (end_time - start_time)/out2.shape[0])
        #print('after layer2 out shape: ', out2.shape) 
        outsize2 = out2.shape
        #start_time = time()
        out3, inds3 = self.layer3(out2)
        #end_time = time()
        outsize3 = out3.shape
        #print('time layer 3: ', (end_time - start_time)/out3.shape[0])
        #print('after layer3 out shape: ', out3.shape)
        #start_time = time()
        out = self.layer4(out3) + out3
        #end_time = time()
        #print('time layer 4: ', (end_time - start_time)/out.shape[0])
        #print('after layer4 out shape: ', out.shape) 
        #start_time = time()
        out = self.layer5(out) + out
        #end_time = time()
        #print('time layer 5: ', (end_time - start_time)/out.shape[0])
        #print('after layer5 out shape: ', out.shape) 
        #start_time = time()
        out = self.layer6(out) + out
        #end_time = time()
        #print('time layer 6: ', (end_time - start_time)/out.shape[0])       
        #print('after layer6 out shape: ', out.shape) 
        #start_time = time()
        out7 = self.layer7(out) + out
        #end_time = time()
        #print('time layer 7: ', (end_time - start_time)/out7.shape[0])       
        #print('after layer7 out shape: ', out7.shape)
        #print('out shape befor PatchEmbeddingLayer: ', out.shape)
        #start_time = time()
        outembed = self.embeddinglayer(out7)
        #end_time = time()
        #print('time embeddinglayer: ', (end_time - start_time)/outembed.shape[0])
        #print('out shape before transformer: ', outembed.shape)
        #start_time = time()
        out = self.layertransformer(outembed)
        #end_time = time()
        #print('time transformer layer: ', (end_time - start_time)/out.shape[0])
        #print('out shape: ', out.shape)
        #start_time = time()
        out = self.layertransformer2(out + outembed)
        #end_time = time()
        #print('time transformer layer: ', (end_time - start_time)/out.shape[0])
        #print('out shape: ', out.shape)
        #out = self.layertransformer(out)
        #print('out shape: ', out.shape)
        out = out.permute((0,2,1))
        #print('out shape: ', out.shape)
        out = out.unflatten(2, (15,20))
        #print('out shape: ', out.shape)
        #start_time = time()
        out = self.layertransup(out)
        #end_time = time()
        #print('time layer transup: ', (end_time - start_time)/out.shape[0])
        #print('out shape: ', out.shape)
        out = self.upsampleLayer2(out)
        #print('out shape: ', out.shape)
        #out = self.layertransup(out)
        #print('out shape: ', out.shape)
        out8 = self.layer8(out + out7) + out3
        #end_time = time()
        #print('time layer 8: ', (end_time - start_time)/out8.shape[0]) 
        #print('after layer8 out shape: ', out8.shape)
        #print('outsize3: ', outsize3)
        #start_time = time()
        out = self.unpoollayer(out8, inds3, output_size=outsize2) 
        #end_time = time()
        #print('time unpool layer layer: ', (end_time - start_time)/out.shape[0])
        #print('after unpool layer out shape: ', out.shape)
        # 2 conv layers with 5 x 5 filters
        #start_time = time()
        #out = self.layer8(out)
        #out = self.layer8(out)    
        out9 = self.layer9(out) + out2 + self.upsampleLayer(out8[:,0::2,:,:])
        #end_time = time()
        #print('time layer 9: ', (end_time - start_time)/out9.shape[0])
        #print('after layer9 out shape: ', out9.shape)
        #start_time = time()
        out = self.unpoollayer(out9, inds2, output_size=outsize1) 
        #end_time = time()
        #print('time unpool layer: ', (end_time - start_time)/out.shape[0])
        #print('after unpool layer out shape: ', out.shape)
        #out = self.layer6(out)
        #print('after layer6 out shape: ', out.shape)
        #start_time = time()
        #out = self.layer9(out)
        out = self.layer10(out) + out1 + self.upsampleLayer(out9[:,0::2,:,:])
        #end_time = time()
        #print('time layer 10: ', (end_time - start_time)/out.shape[0]) 
        #print('after layer10 out shape: ', out.shape)
        #start_time = time()
        #out = self.unpoollayer(out, inds1, output_size=outsize0)
        out = self.layer11(out)
        #end_time = time()
        #print('time layer 11: ', (end_time - start_time)/out.shape[0]) 
        #print('after layer11 out shape: ', out.shape)
        #start_time = time()
        out = self.upsampleLayer(out)
        #end_time = time()
        #print('time unpool layer: ', (end_time - start_time)/8)
        #print('time upsample layer: ', (end_time - start_time)/out.shape[0])
        #print('after upsampleLayer out shape: ', out.shape)
        #start_time = time()
        #end_time = time()
        #print('time layer 10: ', (end_time - start_time)/8) 
        #print('after layer7 out shape: ', out.shape)
        #start_time = time()
        out = self.ClassifyBlock(out)
        #end_time = time()
        #print('time ClassifyBlock layer: ', (end_time - start_time)/out.shape[0])
        #print('after ClassifyBlock out shape: ', out.shape)
        return out



weight_dict = {'Encoder.layer1.0.weight': 'features.0.weight', 'Encoder.layer1.0.bias': 'features.0.bias', 'Encoder.layer2.0.weight': 'features.3.weight', 'Encoder.layer2.0.bias': 'features.3.bias',
'Encoder.layer4.0.weight' : 'features.7.weight', 'Encoder.layer4.0.bias' : 'features.7.bias', 'Encoder.layer5.0.weight' : 'features.10.weight', 'Encoder.layer5.0.bias' : 'features.10.bias', 
'Encoder.layer7.0.weight' : 'features.14.weight', 'Encoder.layer7.0.bias' : 'features.14.bias', 'Encoder.layer8.0.weight' : 'features.17.weight', 'Encoder.layer8.0.bias' : 'features.17.bias',
'Encoder.layer9.0.weight' : 'features.20.weight', 'Encoder.layer9.0.bias' : 'features.20.bias', 'Encoder.layer11.0.weight' : 'features.24.weight', 'Encoder.layer11.0.bias' : 'features.24.bias',
'Encoder.layer12.0.weight' : 'features.27.weight', 'Encoder.layer12.0.bias' : 'features.27.bias', 'Encoder.layer13.0.weight' : 'features.30.weight', 'Encoder.layer13.0.bias' : 'features.30.bias',
'Encoder.layer15.0.weight' : 'features.34.weight', 'Encoder.layer15.0.bias' : 'features.34.bias', 'Encoder.layer16.0.weight' : 'features.34.weight', 'Encoder.layer16.0.bias' : 'features.34.bias',
'Encoder.layer17.0.weight' : 'features.37.weight', 'Encoder.layer17.0.bias' : 'features.37.bias'}
