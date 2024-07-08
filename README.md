# Segmentation Using Dilated Convolutions
The segmentation model uses of dilated comvolutions ([2], [3]), transfer of max-pooling indices as in [1], skip connections from encoder to decoder and use of normal convolutions in parallel with dilated convolutions. The framework used is PyTorch. 

1. Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." IEEE transactions on pattern analysis and machine intelligence 39.12 (2017): 2481-2495.
2.  Chen LC, Papandreou G, Kokkinos I, Murphy K, Yuille AL. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence. 2017 Apr 27;40(4):834-48.
3. Chen LC, Papandreou G, Kokkinos I, Murphy K, Yuille AL. Semantic image segmentation with deep convolutional nets and fully connected crfs. arXiv preprint arXiv:1412.7062. 2014 Dec 22.

## Instructions for training:

run python main.py

## Features of model in model_dilated2.py

1. The encoder consists of the following blocks:
    - Blocks 1 - 3: 2 7 x 7 conv layers with batchnorm and PRelu followed by maxpooling layer
    - Blocks 4 - 6: 2 7 x 7 conv layers followed by 1 x 1 conv layer in parallel 2 3 x 3 conv layers (dilation 2 for block 4, 4 for block 5 and 6 for block 7) folllowed by 1 x 1 conv layer and concatenation of the 2 parallel streams. Each conv layer in Blocks 4 - 6 is followed by batchnorm, dropout and PRelu.
    - Block 7: (in channels: 192, out channels: 128, 1 1 x 1 conv layer with batchnorm and PRelu.
2. The decoder consists adds the output from the first 3 encoder blocks before unpooling by making use of indices obtained from first 3 encoder blocks. Each unpooling layer is followed by 2 5 x 5 conv layers with batchnorm, dropout and PRelu. 
3. The classify block consists of a 1 x 1 conv layer followed by a Softmax activation.

## Results on CamVid dataset 
The model in model_dilated2.py was trained on CamVid semantic segmentation dataset obtained from [this github link](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid), consisting of 367 training images and 101 validation images. The training was done for 35 epochs. The training was done from scratch without use of any pretrained weights.

|  | Result |
| --- | --- |
| pixel accuracy on validation dataset| 89.709 % |
| mean IoU on validation dataset | 53.348 % |

Below are results on some images of the CamVid validation dataset. leftmost is the segmented output, middle is the original image of 360 x 480 resolution and rightmost is the segmentation overlayed on original RGB image.

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/18_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/26_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/37_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/47_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/50_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/63_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/6_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/72_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/76_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/87_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/98_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/95_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/92_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/81_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/28_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/30_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/39_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/47_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/56_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/68_overlayimg_.jpg)
