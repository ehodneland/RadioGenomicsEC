#!/usr/bin/env python3
# Inspired from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#

# https://stackoverflow.com/questions/4130355/python-matplotlib-framework-under-macosx
import argparse
import os
from settingsUNet3D import *
import nibabel as nib
from showim import show3D_1, show3D_2
import time
import config
import windowselector
import pathfun
import numpy as np
import sys
# Standard UNet model
sys.path.append('3DUnetCNN-master')
from loadimUNet3D import loadim
from unet3d.model import unet_model_3d
from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient

tic = time.time()
parser = argparse.ArgumentParser(description='Prediction of new data sets')
parser.add_argument('-network', action='store', nargs=1)
parser.add_argument('-gpunum', action='store', nargs=1)
parser.add_argument('-parfile', action='store', nargs=1)
parser.add_argument('-listdata', action='store', nargs='+')
parser.add_argument('-listpredicted', action='store', nargs='+')
args = parser.parse_args()

# The network path
networkfile = args.network[0]

# The network to load
print('Network file: ' + networkfile)

# Name of network
networkdir, e, networkbase, ext = pathfun.get(networkfile)
e, networkname = os.path.split(networkdir)

# GPU number to use for training
gpunum = args.gpunum[0]

if gpunum == 'CPU':
    gpunum=''

# Import setup file
import importlib

# List of data sets to predict on
listdata = args.listdata

# Number of sequenes
nseq = len(listdata)

# Import parameter file
try:
    parfile = args.parfile[0]
except:
    parfile = os.path.join('tools', config.parfile)

# Import parameter file
print('Import parameter file ' + parfile)
dirparfile, n, baseparfile, e = pathfun.get(parfile)
sys.path.append(dirparfile)
spec = importlib.util.spec_from_file_location("report", parfile)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
if hasattr(module, '__all__'):
    all_names = module.__all__
else:
    all_names = [name for name in dir(module) if not name.startswith('_')]
globals().update({name: getattr(module, name) for name in all_names})

# Is there a list of output data provided?
islistpred = True

# List of output data
listpredicted = args.listpredicted[0]

# File to save the path of the predicted data sets
print('Loading list of predicted data sets ' + listpredicted)
pathpredicted = np.loadtxt(listpredicted, dtype='str')
try:
    ndata = len(pathpredicted)
except:
    ndata = 1

# Load path to data to predict
pathdata = np.empty([ndata, nseq], dtype=object)
i = 0
for s in listdata:
    print('Loading lists of data files ' + s)
    pathdata[:, i] = np.loadtxt(s, dtype='str')
    i = i + 1

# To print entire array to screen
np.set_printoptions(threshold=1e6)

if len(gpunum) == 0:
    cmd = '/device:CPU:0'
else:
    cmd = '/device:GPU:' + str(gpunum)

import tensorflow as tf
from keras.models import load_model
with tf.device(cmd):

    # Define the model
    inputshape = (nseq, ws[0], ws[1], ws[2])
    if model == 'UNet1':
        model = unet_model_3d(inputshape,
                              pool_size=poolsize,
                              n_labels=1,
                              initial_learning_rate=lr,
                              deconvolution=deconvolution,
                              depth=depth,
                              n_base_filters=n_base_filters,
                              include_label_wise_dice_coefficients=False,
                              metrics=dice_coefficient,
                              batch_normalization=batch_normalization,
                              activation_name=activation_name)
    elif model == 'UNet2':
        model = isensee2017_model(inputshape,
                                  depth=depth,
                                  n_labels=1,
                                  initial_learning_rate=lr)

    # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    msg = "Loading model " + networkfile
    print(msg)
    model.load_weights(networkfile)
    model.summary()

    # List of the predicted to load later
    listpred = np.empty((ndata,), dtype='object')

    # Loop over each subject and predict
    for i in np.arange(0, ndata):

        pathsave = pathpredicted[i]

        # Load data
        im4D, nii3D, h, swapax, swapdims = loadim(pathdata[i, :], swap=True)
        h = h[0, :]
        header = nii3D.header
        dim = im4D.shape[1:]

        # Crop different window sizes
        ncrop, imcrop, startcrop, stopcrop, ww = windowselector.crop(im4D, ws, nseq, True, spacing)
        print('Number of crops: ' + str(ncrop))

        # Predict batch-wise
        start = 0
        step = np.minimum(ncrop, predstep)
        predicted = np.zeros((ncrop, ws[0], ws[1], ws[2]), dtype='float32')
        while 1:
            stop = np.minimum(start + step, ncrop)
            imh = imcrop[start:stop, :, :, :, :]
            v = model.predict(imh)
            v = np.squeeze(v)
            predicted[start:stop, :, :, :] = v
            if stop == ncrop:
                break
            start = stop

        # Put the results back  into the image matrix
        predim = np.zeros(dim, dtype='float32')

        c = 0
        for j in np.arange(ncrop):
            start = startcrop[j, :]
            stop = stopcrop[j, :]
            v = predicted[j, :, :, :]
            predim[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = predim[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] + v

        # Take the average
        predim = predim / ww

        # Swap back again to origin image
        if len(swapax) > 0:
            print('Swapping axes ' + str(swapax[1]) + ' with ' + str(swapax[0]))
            predim = np.swapaxes(predim, swapax[1], swapax[0])
            h = np.swapaxes(h, swapax[1], swapax[0])
            print('Shape of data after swapping: ' + str(predim.shape))
            print('Voxelsize of data after swapping: ' + str(h))

        print('Saving ' + pathsave)

        # Set as float32
        nii3D.set_data_dtype(np.float32)
        niisave = nib.Nifti1Image(predim, nii3D.affine, header=nii3D.header)
        nib.save(niisave, pathsave)

        toc = time.time()
        print('Time elapsed: ' + str(toc-tic) + 's')

        i = i + 1
