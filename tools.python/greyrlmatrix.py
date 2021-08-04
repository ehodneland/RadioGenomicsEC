import numpy as np

from scipy.ndimage import rotate
from showim import show3D_1

# To print entire array to screen
np.set_printoptions(threshold=1e6)

# NB: The function is working column wise, so maximum run length is the number of rows
def greyrlmatrix(im, uval, maxrl, theta):

    dim = im.shape
    # Number of rotations
    ntheta = len(theta)

    # Number of unique grey values
    nuval = len(uval)

    # Initialization of glrlm matrix
    glrlm = np.zeros((nuval, maxrl, ntheta), dtype='int')
    thi = 0
    for th in theta:
        # Rotate image
        imh = rotate(im, th*180/np.pi, mode='constant', order=0, reshape=False)
        for j in np.arange(dim[1]):
            counter = 1
            valold = im[0, j]
            for i in np.arange(1, dim[0]):
                val = imh[i, j]
                if val == valold:
                    # We are repeating the old grayscale value
                    counter = counter + 1
                else:
                    # We are hitting a new grayscale value
                    glrlm[valold, counter-1] = glrlm[valold, counter-1] + 1
                    # Reset counter
                    counter = 1
                valold = val
            # Finalizing one column
            glrlm[valold, counter - 1, thi] = glrlm[valold, counter - 1, thi] + 1
        thi = thi + 1
    return glrlm


