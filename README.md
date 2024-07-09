# Segmentation Using Dilated Convolutions
This branch contains code for training a model containing only 8.292 Million parameters designed for segmenting images of road scenes with 12 classes. The segmentation model ses dilated comvolutions ([2], [3]), transfer of max-pooling indices as in [1], skip connections from encoder to decoder and normal convolutions in parallel with dilated convolutions. The framework used is PyTorch. 

References:

[1] Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." IEEE transactions on pattern analysis and machine intelligence 39.12 (2017): 2481-2495. <br/> <br/>
[2]  Chen LC, Papandreou G, Kokkinos I, Murphy K, Yuille AL. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence. 2017 Apr 27;40(4):834-48. <br/> <br/>
[3] Chen LC, Papandreou G, Kokkinos I, Murphy K, Yuille AL. Semantic image segmentation with deep convolutional nets and fully connected crfs. arXiv preprint arXiv:1412.7062. 2014 Dec 22. <br/> <br/>

## Instructions for training:

run python main.py

## Features of model in model_dilated2.py

1. The encoder consists of the following blocks:
    - Blocks 1 - 3: 2 7 x 7 conv layers with batchnorm and PRelu followed by maxpooling layer
    - Blocks 4 - 6: 2 7 x 7 conv layers followed by 1 x 1 conv layer in parallel with 2 3 x 3 conv layers (dilation 2 for block 4, 4 for block 5 and 6 for block 6) folllowed by 1 x 1 conv layer and concatenation of the 2 parallel streams. Each conv layer in blocks 4 - 6 is followed by batchnorm, dropout and PRelu.
    - Block 7: 1 1 x 1 conv layer with batchnorm and PRelu.
2. The decoder consists adds the output from the first 3 encoder blocks before unpooling by making use of indices obtained from first 3 encoder blocks. Each unpooling layer is followed by 2 5 x 5 conv layers with batchnorm, dropout and PRelu. 
3. The classify block consists of a 1 x 1 conv layer followed by a Softmax activation.

## Results on CamVid dataset 

The model in model_dilated2.py was trained on CamVid semantic segmentation dataset obtained from [this github link](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid), consisting of 367 training images and 101 validation images. The training was done for 60 epochs and the model having best mean iou on validation data was selected. The training was done from scratch without use of any pretrained weights.
<br/><br/>


|  | Result |
| --- | --- |
| pixel accuracy on validation dataset| 89.769 % |
| mean IoU on validation dataset | 55.791 % |

<br/>
Below are results on some images of the CamVid validation dataset. leftmost is the segmented output, middle is the original image of 360 x 480 resolution and rightmost is the segmentation overlayed on original RGB image.

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/26_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/30_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/30_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/38_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/3_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/42_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/43_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/47_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/4_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/52_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/65_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/73_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/75_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/76_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/87_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/90_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/SegmentationDilated/CamVid%20Val%20Result%20Images/96_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/47_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/56_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/68_overlayimg_.jpg)
