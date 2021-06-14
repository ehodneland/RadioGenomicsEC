#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import swapdim
import resizeimage
from scale import sc
from showim import show3D_1, show3D_2
#from settingsCNN import *
#from scale import sc
from copy import deepcopy
import os.path
import imresize

# Number of timepoints to use in the 4D dataset
ntime = 160

import re

# Remember that there are three features for each 4D dataset
def loadim(pathdata, swap):

    # Number of sequenes to load
    nseq = len(pathdata)

    # Voxel size
    h = np.zeros((nseq, 3), dtype='float32')

    # Load images
    jj = 0
    for j in np.arange(nseq):

        # Read data as image
        pathload = pathdata[j]
        # Remove whitespaces
        pathload = re.sub(r"\s+", "", pathload)
        msg = "Reading image data " + pathload
        print(msg)
        nii = nib.load(pathload)        
        im = nii.get_data().astype('float32')
        dim = im.shape
        header = nii.header
        H = np.array(header.get_zooms(), dtype='float32')
        H = H[0:3]
        print('Shape of loaded data: ' + str(dim))
        print('Voxelsize of loaded data: ' + str(H))

        # Make empty arrays
        if j == 0:
            dim4D = (nseq, dim[0], dim[1], dim[2])
            im4D = np.empty(dim4D, dtype='float32')

        # Swap dimensions and resize
        swapax = list()
        swapdims = list()
        if swap:
            im, H, swapax, swapdims = swapdim.swapdim(im, H)

        #if resize:
            #im, H = resizeimage.resize(im, H, dim3D, False)

        # Assign 3D array to 4D array
        im4D[jj, :, :, :] = im
        h[jj, :] = H.copy()
        jj = jj + 1

    return im4D, nii, h, swapax, swapdims
