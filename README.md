# Trre Crown Classification

## Description

The aim of this project is to apply Deep Learning techniques, to classify tree crown species from UAV images.
The architectures under investigation are comprised of Convolutional Neural Networks, Transformer and Hybrid CNN-Transformer models. The dataset is comprised of UAV RGB and NDVI images depicting tree crowns of tree species from Wytham forest in Oxford. 

Moreover, conducting an investigation regarding Deep Learning Multimodal data fusion.

![alt text](https://github.com/gvsam7/TreeCrownClass/blob/main/Images/TreeCrownClasses.jpg)

*Data:*  RGB and NDVI UAV tree crown species from Wytham Forest.
Stratification method was used to split the data to train/validate: 80% (out of which train: 80% and
validation: 20%), and test: 20% data.

*Data texture bias:* Research techniques that take advantage texture bias in UAV images.

*Architectures:* 4, and 5 CNN, ResNet18, ResNet50, ResNet101, ResNet152, VGG11, VGG13, VGG16, VGG19, AlexNet.

*Images:* Size: 64x64, 128x128, 256x256, and 512x512 pixel images. 

*Test Procedure:* 5 runs for each architecture for each of the compressed data. Then plot the Interquartile range.

*Plots:* Average GPU usage per architecture, Interquartile, F1 Score heatmap for each class, Confusion Matrix, PCA and t-SNE plots, and most confident incorrect predictions.

*Data augmentations:* Geometric Transformations, Cutout, Mixup, and CutMix, Pooling (Global Pool, Mix Pool, Gated Mix Pool).

