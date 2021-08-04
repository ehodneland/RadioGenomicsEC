#!/usr/bin/env python3

import matplotlib  
matplotlib.use('TkAgg')   
import numpy as np
import os
import sys
import nibabel as nib
import argparse
from loadimUNet3D import loadim
from showim import show3D_1, show3D_2
import swapdim
import pathfun
import config

os.environ['KERAS_BACKEND']='tensorflow'
sys.path.append('3DUnetCNN-master')

import tensorflow as tf
#from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

# Standard UNet model
from unet3d.model import unet_model_3d
from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient

parser = argparse.ArgumentParser(description='Label training data')
#parser.add_argument('values', metavar='N', nargs='+')
parser.add_argument('-networkdir', action='store', nargs=1)
parser.add_argument('-gpunum', action='store', nargs=1)
parser.add_argument('-ws', action='store', nargs=3)
parser.add_argument('-setupfile', action='store', nargs=1)
parser.add_argument('-parfile', action='store', nargs=1)
parser.add_argument('-networkfile0', action='store', nargs=1)
args = parser.parse_args()

# The network path
networkdir = args.networkdir[0]

# The network to save
networkfile = os.path.join(networkdir, config.networkbase + '.h5')
# Name of network for making list of predicted files
e, f, networkname, ext0 = pathfun.get(networkdir)
if not os.path.exists(networkdir):
    os.mkdir(networkdir)

# GPU number to use for training
gpunum = args.gpunum[0]

# Parameter file
try:
    parfile = args.parfile[0]
except:
    parfile = os.path.join('tools', config.parfile)

# Initial network file given?
try:
    networkfile0 = args.networkfile0[0]
    networkdir0, e, networkbase0, ext0 = pathfun.get(networkfile0)
    msg = 'Pre-existing weights given, remember to assign a correct PARFILE'
    existweights = True
except:
    msg = 'Pre-existing weights not given'
    existweights = False
print(msg)

# Import setup file of data sets
setupfile = args.setupfile[0]
dirsetup, n, basesetup, e = pathfun.get(setupfile)
sys.path.append(dirsetup)
print('Import setupfile ' + setupfile)
setup = __import__(basesetup)

# Import parameter file
print('Import parameter file ' + parfile)
dirparfile, n, baseparfile, e = pathfun.get(parfile)
sys.path.append(dirparfile)
import importlib
spec = importlib.util.spec_from_file_location("report", parfile)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
if hasattr(module, '__all__'):
    all_names = module.__all__
else:
    all_names = [name for name in dir(module) if not name.startswith('_')]
globals().update({name: getattr(module, name) for name in all_names})

# List of data sets to train
listdatatrain = setup.listdatatrain

# List of subjects to train
listsubjtrain = setup.listsubjtrain[0]

# List of masks to train
listmasktrain = setup.listmasktrain[0]

# List of subjects to predict
listsubjpredict = setup.listsubjpredict[0]

# Lists of data to predict
listdatapredict = setup.listdatapredict

# Sequence names
seqname = setup.seqname

# Number of sequenes
nseq = len(listdatatrain)

# To print entire array to screen
np.set_printoptions(threshold=1e6)

# Load list of masks
pathtrainmask = np.loadtxt(listmasktrain, dtype='str')

# Load list of subjects
subjstr = np.loadtxt(listsubjtrain, dtype='str')

# Number of subjects
ndata = len(pathtrainmask)

# Subjects to predict on
subjstrpredict = np.loadtxt(listsubjpredict, dtype='str')
ndatapredict = len(subjstrpredict)

# Copy the lists used into a separate folder
cmd = 'cp ' + listmasktrain + ' ' + networkdir
print(cmd)
os.system(cmd)

# Copy setupfile into separate folder
cmd = 'cp ' + setupfile + ' ' + networkdir
print(cmd)
os.system(cmd)

cmd = 'cp ' + listsubjtrain + ' ' + networkdir
print(cmd)
os.system(cmd)

cmd = 'cp ' + listsubjpredict + ' ' + networkdir
print(cmd)
os.system(cmd)

# Copy settings file to have the settings for later use
cmd = 'cp ' + parfile + ' ' + networkdir
print(cmd)
os.system(cmd)

for pathname in listdatatrain:
    cmd = 'cp ' + pathname + ' ' + networkdir
    print(cmd)
    os.system(cmd)
for pathname in listdatapredict:
    cmd = 'cp ' + pathname + ' ' + networkdir
    print(cmd)
    os.system(cmd)

# Load list of training data into a multidimensiontal np char array
pathtraindata = np.empty([ndata, nseq], dtype=object)
i = 0
for s in listdatatrain:
    pathtraindata[:, i] = np.loadtxt(s, dtype='str')
    i = i + 1

# Make list of predicted files using this network
listpredicted = np.empty([ndatapredict, 1], dtype='object')
for idx, s in enumerate(subjstrpredict):
    s2 = str(s).zfill(3)
    b = os.path.join(config.datadir, s2, config.analysis, 'PREDICTED-NETWORK-' + networkname + '.nii.gz')
    listpredicted[idx] = b

# Make list of predicted files
pathpredicted = os.path.join(networkdir, config.listpredicted)
print('Saving list of predicted data: ' + pathpredicted)
np.savetxt(pathpredicted, listpredicted, fmt='%s')

# Number of subjects
print('Number of subjects: ' + str(ndata))

# Number of channels
print('Number of channels: ' + str(nseq))

# Image dimensions
dim4D = (nseq, dim3D[0], dim3D[1], dim3D[2])

# Number of spatial dimensions
ndim = len(dim3D)

# Number of total samples
nsamp = nrand + ncent

# Trainingdata
X = np.zeros((ndata*nsamp, nseq, ws[0], ws[1], ws[2]), dtype='float32')
Y = np.zeros((ndata*nsamp, ws[0], ws[1], ws[2]), dtype='float32')

######################################################################
# Load the data and do augmentation
######################################################################

crand = 0
samplec = 0
ndataload = 0
i = 0
#i = 58
for s in subjstr:
    #s = '260'
    #subj = 260
    subj = s.astype('int')
    print("New subject " + s)

    # Array containing whether the data sets exist
    # exist = np.zeros(nseq+1, dtype=bool)

    # Load image data
    im4D, nii3D, h, swapax, swapdims = \
        loadim(pathtraindata[i, :], swap=True)
    h = h[0, :]

    # Load training mask
    msg = "Reading training mask " + pathtrainmask[i]
    print(msg)
    nii = nib.load(pathtrainmask[i])
    maskload = nii.get_data().astype('float')
    dim = maskload.shape
    header = nii.header
    hmask = header.get_zooms()

    print("Shape of training mask " + str(maskload.shape))
    print('Voxelsize of training mask: ' + str(hmask))

    # Swap dimensions?
    mask, h, swapax, swapdims = swapdim.swapdim(maskload, h)

    # Find the best window placement so the tumor is in the middle of the window
    # OR crop a random placement elsewhere to train the network on background
    maskcrop = np.empty([nsamp, ws[0], ws[1], ws[2]], dtype='float32')
    maskcrop[:] = np.nan
    im4Dcrop = np.empty([nsamp, nseq, ws[0], ws[1], ws[2]], dtype='float32')
    im4Dcrop[:] = np.nan

    # The valid data
    valid = np.ones(nsamp, dtype='bool')
    #   The range of valid coordinates for a random crop placement
    validc = np.array([[ws[0]/2, dim[0]-ws[0]/2],
                       [ws[1]/2, dim[1]-ws[1]/2],
                       [ws[2]/2, dim[2]-ws[2]/2]],
                       dtype='int').round()

    crand = 0
    ccent = 0
    for k in np.arange(nsamp):
        print('Applying cut to sampling ' + str(k))

        # k == 0 is the central placement around the mask
        if k < ncent:
            print('Central mask placement')
            placement = 'central'
        else:
            # Random placement
            print('Random mask placement')
            placement = 'random'

        while 1:
            if placement == 'central':
                # Find a window placement with the tumor in the middle
                c = np.where(mask > 0)
            elif placement == 'random':
                # Make random coordinates
                c = np.zeros(3, dtype='int')
                for ii in np.arange(3):
                    c[ii] = np.random.random_integers(validc[ii][0], validc[ii][1])

            start = np.zeros(3, dtype='int')
            stop = np.zeros(3, dtype='int')
            for ii in np.arange(3):
                meanc = np.around(np.mean(c[ii]))
                start[ii] = meanc - np.round(ws[ii]/2)
                stop[ii] = start[ii] + ws[ii]

            maskh = mask[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            imh = im4D[:, start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            equalcropdim = np.array_equal(maskh.shape, ws)
            validcrop = False
            if equalcropdim:
                validcrop = True
                print('Valid crop x: ' + str(start[0]) + ' to ' + str(stop[0]))
                print('Valid crop y: ' + str(start[1]) + ' to ' + str(stop[1]))
                print('Valid crop z: ' + str(start[2]) + ' to ' + str(stop[2]))
            # try:
            #     # Crop training data
            #     maskh = mask[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            #     imh = im4D[:, start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            #     print(maskh.shape)
            #     a
            #     validcrop = True
            # except:
            #     print('Could not crop due to outside image'.upper())
            #     validcrop = False
            #     break

            # show3D_2(imh[0, :, :, :], maskh)
            if validcrop:
                break

        if placement == 'central':
            ccent = ccent + 1
        elif placement == 'random':
            crand = crand + 1

        #if validcrop:
        maskcrop[k, :, :, :] = maskh
        im4Dcrop[k, :, :, :, :] = imh
        #else:
        #    # Dont take this crop for any of the reasons above
        #    valid[k] = False


            #show3D_2(imh[0, :, :, :], maskh)

        print('Number of central samples: ' + str(ccent))
        print('Number or random samples: ' + str(crand))

    # Pick out the valid data
    maskcrop = maskcrop[valid, :, :, :]
    im4Dcrop = im4Dcrop[valid, :, :, :, :]

    # Add to array
    X[samplec:samplec+nsamp, :, :, :, :] = im4Dcrop
    Y[samplec:samplec+nsamp, :, :, :] = maskcrop

    # Update global counter
    samplec = samplec + nsamp

    i = i + 1

# Restrict to valid data
X = X[0:samplec+1, :, :, :, :]
Y = Y[0:samplec+1, :, :, :]

msg = "Shape of feature data: " + str(X.shape)
print(msg)
msg = "Shape of class data: " + str(Y.shape)
print(msg)

######################################################################
# Train a deep learning network
######################################################################

tf.debugging.set_log_device_placement(True)
cmd = '/device:GPU:' + str(gpunum)
print(cmd)
with tf.device(cmd):

    # From https://github.com/dhuy228/augmented-volumetric-image-generator
    from sklearn.model_selection import train_test_split
    seed = 42
    d = Y.shape
    Y = np.reshape(Y, (d[0], 1, d[1], d[2], d[3]))
    if valsplit > 0:
        X_train_c, X_validation_c, Y_train_c, Y_validation_c = train_test_split(X, Y, test_size=valsplit, random_state=seed)
    else:
        X_train_c = X.copy()
        Y_train_c = Y.copy()

    #Y_train_c = np.reshape(Y_train_c, (d[0], 1, d[1], d[2], d[3]))
    #Y_validation_c = np.reshape(Y_validation_c, (d[0], 1, d[1], d[2], d[3]))

    from augmented import generator
    if augment:
        image_aug = generator.customImageDataGenerator(
            rotation_range=rotation_range,
            zoom_range=zoom_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            horizontal_flip=horizontal_flip,
            data_format='channels_first')
        mask_aug = generator.customImageDataGenerator(
            rotation_range=rotation_range,
            zoom_range=zoom_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            horizontal_flip=horizontal_flip)
    else:
        image_aug = generator.customImageDataGenerator(data_format='channels_first')
        mask_aug = generator.customImageDataGenerator()

    X_train_datagen = image_aug.flow(X_train_c, batch_size=batch_size, seed=seed)  # set equal seed number
    Y_train_datagen = mask_aug.flow(Y_train_c, batch_size=batch_size, seed=seed)  # set equal seed number
    train_generator = zip(X_train_datagen, Y_train_datagen)
    print('Validation split: ' + str(valsplit))
    if valsplit > 0:
        X_validation_datagen = image_aug.flow(X_validation_c, batch_size=batch_size, seed=seed)  # set equal seed number
        Y_validation_datagen = mask_aug.flow(Y_validation_c, batch_size=batch_size, seed=seed)  # set equal seed number
        validation_generator = zip(X_validation_datagen, Y_validation_datagen)

    # # Uncomment to see augmentations
    # import matplotlib.pyplot as plt
    # it = image_aug.flow(X_train_c, batch_size=1, seed=seed)
    # itmask = image_aug.flow(Y_train_c, batch_size=1, seed=seed)
    # for i in range(100):
    #     # generate batch of images
    #     batch = it.next()
    #     batchmask = itmask.next()
    #     image = batch[0].astype('float32')
    #     image = np.squeeze(image)
    #     mask = batchmask[0].astype('float32')
    #     mask = np.squeeze(mask)
    #     show3D_2(image, mask)
    # # Define the model
    inputshape = (nseq, ws[0], ws[1], ws[2])
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

    # Print summary
    model.summary()
    #config = model.get_config()
    #print(config)
    #print(config["initial_learning_rate"])

    # Compile
    model.compile(optimizer=RMSprop(lr=lr), loss=[dice_coefficient_loss], metrics=[dice_coefficient])
    # model.compile(optimizer=RMSprop(lr=0.0001), loss=dice_loss, metrics=[dice_coeff])
    # Binary crossentropy is not working, why not?
    # model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=[dice_coeff])

    # Load pre-existing model weights?
    if existweights:
        msg = "Loading model " + networkfile0
        print(msg)
        model.load_weights(networkfile0)
        #from keras.models import load_model
        #model = load_model(networkfile0)

    # Save network to disc for each epoch
    checkpointer = ModelCheckpoint(filepath=networkfile, verbose=1)

    # Early stopping?
    from keras.callbacks import EarlyStopping
    usualCallback = EarlyStopping()
    fitCallback = EarlyStopping(monitor='loss', min_delta=0, patience=30)

    # Fit model
    if valsplit > 0:
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=len(Y_train_c) // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(Y_validation_c) // batch_size,
            callbacks=[checkpointer, fitCallback]
        )
    else:
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=len(Y_train_c) // batch_size,
            epochs=epochs,
            callbacks=[checkpointer, fitCallback]
        )


    #history = model.fit(X, Y, batch_size=batch_size, epochs=epochs, callbacks=[checkpointer, fitCallback], validation_split=valsplit)
    msg = "Saving model " + networkfile
    print(msg)
    model.save(networkfile)

    # Save training loss
    pathsave = os.path.join(networkdir, config.trainloss)
    print('Saving train loss: ' + pathsave)
    np.savetxt(pathsave, history.history['loss'], fmt='%s')

    # Save training cost function
    pathsave = os.path.join(networkdir, config.traincost)
    print('Saving train cost: ' + pathsave)
    np.savetxt(pathsave, history.history['dice_coefficient'], fmt='%s')

    if valsplit > 0:
        # Save validation loss
        pathsave = os.path.join(networkdir, config.valloss)
        print('Saving validation loss: ' + pathsave)
        np.savetxt(pathsave, history.history['val_loss'], fmt='%s')

        # Save validation cost function
        pathsave = os.path.join(networkdir, config.valcost)
        print('Saving validation cost: ' + pathsave)
        np.savetxt(pathsave, history.history['val_dice_coefficient'], fmt='%s')


