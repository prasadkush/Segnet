# Segmentation Using Dilated Convolutions
This is a semantic segmentation model containing 3 conv blocks (each containing 2 conv layers followed batchnorm and PRelu), 4 blocks with normal convolutions and dilated convolutions in paralled followed by a 1x1 conv layer and a decoder. Number of parameters in the model are 13.386 M. The decoder uses transfer of max-pooling indices as in [1] and feature maps from encoder layers. The framework used is PyTorch. 

References:

[1] Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." IEEE transactions on pattern analysis and machine intelligence 39.12 (2017): 2481-2495. <br/> <br/>
[2]  Chen LC, Papandreou G, Kokkinos I, Murphy K, Yuille AL. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence. 2017 Apr 27;40(4):834-48. <br/> <br/>
[3] Chen LC, Papandreou G, Kokkinos I, Murphy K, Yuille AL. Semantic image segmentation with deep convolutional nets and fully connected crfs. arXiv preprint arXiv:1412.7062. 2014 Dec 22. <br/> <br/>

## Instructions for training:

run python main.py

## Features of model in model_dilated4.py

1. The encoder consists of the following blocks:
    - Blocks 1: 2 4 x 4 conv layers, dilation=2 with batchnorm and PRelu followed by maxpooling layer
    - Blocks 2: 2 7 x 7 conv layers (input ch = 64, out ch = 128) with batchnorm and PRelu followed by maxpooling layer
    - Block 3: 2 7 x 7 conv layers (input ch = 128, out ch = 128) with batchnorm and PRelu followed by maxpooling layer
    - Blocks 4 - 7: 2 7 x 7 conv layers (input ch = 128, out ch = 128) followed by 1 x 1 conv layer in parallel with 2 3 x 3 conv layers (input ch = 128, out ch = 128, dilation 2 for block 4, 4 for block 5, 6 for block 6, 8 for block 7) folllowed by 1 x 1 conv layer and concatenation of the 2 parallel streams. Each conv layer in blocks 4 - 7 is followed by batchnorm, dropout and PRelu. Each block in 4-7 receives a residual connection from the previous layer.
    - Block 8: 1 1 x 1 conv layer with batchnorm and PRelu.
3. The decoder consists adds the output from the first 3 encoder blocks before unpooling, makes use of indices obtained from first encoder blocks 2 and 3. There are 2 unpooling layers, each followed by 2 3 x 3 conv layers with batchnorm, dropout and PRelu. The outputs from encoder blocks 1-3 are added to the output of these conv blocks along with upsampled output of the layer before unpooling. We don't unpool the last layer, we just upsample it.  
4. The classify block consists of a 1 x 1 conv layer followed by a Softmax activation.

## Results on CamVid dataset 
The model in model_dilated4.py was trained on CamVid semantic segmentation dataset obtained from [this github link](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid), consisting of 367 training images and 101 validation images. The training was done for 45 epochs. The training was done from scratch without use of any pretrained weights.

|  | Result |
| --- | --- |
| pixel accuracy on validation dataset| 91.080 % |
| mean IoU on validation dataset | 58.839 % |

Below are results on some images of the CamVid validation dataset. leftmost is the segmented output, middle is the original image of 360 x 480 resolution and rightmost is the segmentation overlayed on original RGB image.

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/13_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/17_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/19_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/20_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/29_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/2_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/43_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/64_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/66_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/6_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/76_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/79_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/85_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/8_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/91_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated2/CamVid%20Val%20Result%20Images/99_overlayimg_.jpg)
