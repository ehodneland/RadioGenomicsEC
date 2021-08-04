import numpy as np

from scipy.ndimage import rotate
from showim import show3D_1

# To print entire array to screen
np.set_printoptions(threshold=1e6)


def greyrlprops(mat, option):

    if option == 'SRE':
        val = computeSRE(mat)
    elif option == 'LRE':
        val = computeLRE(mat)
    elif option == 'LGRE':
        val = computeLGRE(mat)
    elif option == 'HGRE':
        val = computeHGRE(mat)
    else:
        print('Wrong option to matprops')
        val = ''

    return val


def computeSRE(mat):

    dim = mat.shape
    v = 0
    for i in np.arange(dim[0]):
        for j in np.arange(dim[1]):
            j2 = (j+1)*(j+1)
            val = mat[i, j]/j2
            v = v + val
    return v


def computeLRE(mat):

    dim = mat.shape
    v = 0
    for i in np.arange(dim[0]):
        for j in np.arange(dim[1]):
            j2 = (j+1)*(j+1)
            val = mat[i, j]*j2
            v = v + val
    return v


def computeLGRE(mat):

    dim = mat.shape
    v = 0
    for i in np.arange(dim[0]):
        i2 = (i+1)*(i+1)
        for j in np.arange(dim[1]):
            val = mat[i, j]/i2
            v = v + val
    return v


def computeHGRE(mat):

    dim = mat.shape
    v = 0
    for i in np.arange(dim[0]):
        i2 = (i+1)*(i+1)
        for j in np.arange(dim[1]):
            val = mat[i, j]*i2
            v = v + val
    return v

