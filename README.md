# Segnet
Code for segmentation network based on Segnet. The framework is PyTorch. 

[1] Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." IEEE transactions on pattern analysis and machine intelligence 39.12 (2017): 2481-2495.

## Instructions for training:

run python main.py

## Results on Kitti test dataset 
The model in modelv2.py was trained on kitti semantic segmentation dataset obtained from [kitti website](https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015), consisting of 200 training images. The training was done for 46 epochs. The model consisted of skip connections between certain different scale feature maps in the decoder, besides a basic Segnet architecture.

Below are the segmented output images from the network when fed with test images from the kitti datasset. leftmost is the segmented output, middle is the original image resized to 360 x 480 resolution and rightmost is the segmentation overlayed on resized original RGB image.

![Result on kitti test image](https://github.com/prasadkush/Segnet/blob/main/images/117_overlayimg2_.jpg)

![Result on kitti test image](https://github.com/prasadkush/Segnet/blob/main/images/126_overlayimg2_.jpg)

![Result on kitti test image](https://github.com/prasadkush/Segnet/blob/main/images/133_overlayimg2_.jpg)

![Result on kitti test image](https://github.com/prasadkush/Segnet/blob/main/images/134_overlayimg2_.jpg)

![Result on kitti test image](https://github.com/prasadkush/Segnet/blob/main/images/16_overlayimg2_.jpg)

![Result on kitti test image](https://github.com/prasadkush/Segnet/blob/main/images/17_overlayimg2_.jpg)

![Result on kitti test image](https://github.com/prasadkush/Segnet/blob/main/images/198_overlayimg2_.jpg)

![Result on kitti test image](https://github.com/prasadkush/Segnet/blob/main/images/34_overlayimg2_.jpg)

![Result on kitti test image](https://github.com/prasadkush/Segnet/blob/main/images/46_overlayimg2_.jpg)

![Result on kitti test image](https://github.com/prasadkush/Segnet/blob/main/images/47_overlayimg2_.jpg)
