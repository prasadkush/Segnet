# Segnet
Code for segmentation network based on Segnet. The framework is PyTorch. 

[1] Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." IEEE transactions on pattern analysis and machine intelligence 39.12 (2017): 2481-2495.

## Instructions for training:

run python main.py

## Features of model in modelv3.py

1. VGG-16 type encoder and decoder layers.
2. Transfer of max-pooling indices to decoder for unpooling.
3. Skip connections between certain layers in decoder.
3. Skip connections between certain layers of encoder and decoder. 
5. Use of PRelu as activation.

## Results on CamVid dataset 
The model in modelv3.py was trained on CamVid semantic segmentation dataset obtained from [this github link](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid), consisting of 367 training images and 101 validation images. The training was done for 35 epochs. The model consisted of skip connections between certain different scale feature maps in the decoder, and between certain encoder and decoder layers, besides a basic Segnet architecture. The training was done from scratch without use of any pretrained weights.

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

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/87_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/81_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/28_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/30_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/39_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/47_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/56_overlayimg_.jpg)

![Result on CamVid validation set image](https://github.com/prasadkush/Segnet/blob/CamVid/CamVid%20Val%20Result%20Images/68_overlayimg_.jpg)
