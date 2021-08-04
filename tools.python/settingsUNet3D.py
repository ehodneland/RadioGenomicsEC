#!/usr/bin/env python3

# Innstallation CUDA
# https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e

# Dimension of images (NB must be a tuple with (.) brackets)
dim3D = (192, 192, 48)

# Window size
#ws = (92, 92, 12)
ws = (128, 128, 16)
#ws = (148, 148, 20)
ws = (192, 192, 32)

# Spacing between window sampling, 5
spacing = 8

# Number of samples to predict in one batch
predstep = 3

# Depth of network
depth = 4

# Model
model = 'UNet1'

# Learning rate
lr = 0.00001

# Pool size
poolsize = (2, 2, 2)

# Base filters
#n_base_filters = 16
n_base_filters = 32

# Batch normalization
batch_normalization = False
#batch_normalization = True

# Activation function
activation_name = "sigmoid"
#activation_name = "relu"

# Deconvolution
deconvolution = True

# Number of epochs
epochs = 1500

# Augmentation?
augment = True
#augment = False

# Augmentation settings
rotation_range = 0.10
zoom_range = 0.15
width_shift_range = 0.00
height_shift_range = 0.00
shear_range = 0.00
horizontal_flip = True

# Batch size: 25
batch_size = 5

# Number of samples from each image
ncent = 0

# Number of random samples
nrand = 8

# Validation/train split
valsplit = 0.00
