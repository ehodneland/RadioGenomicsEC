
import swapdim
#import imresize
import numpy as np
from scipy.interpolate import interpn, RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
import gengrid
from showim import show3D_1, show3D_2


def resize(im, h, dim3D, ismask):

    dim = im.shape

    ntime = 1
    if len(dim) == 4:
        ntime = dim[3]

    dim = dim[0:3]

    interpmethod = 'linear'
    if ismask:
        interpmethod = 'nearest'

    # Make an image with the correct dimensions and put it in there
    if dim == dim3D:
        # Do nothing, everything is fine with the dimensions
        im = im.copy()
    elif ntime == 1:
        im, h = imresize3D(im, h, dim3D, interpmethod)
    elif ntime > 1:
        im, h = imresize4D(im, h, dim3D, interpmethod)

    print('Shape of data after imresize: ' + str(im.shape))
    print('Voxelsize of data after imresize: ' + str(h))

    return im, h


def imresize3D(im0, h0, dim, interpmethod):

    # Image size
    dim0 = im0.shape[0:3]

    # Create a regular grid
    x0, y0, z0, x0lin, y0lin, z0lin = gengrid.centered3D(dim0, h0)

    # Require the same FOV
    fov0 = dim0 * h0

    # Interpolation function
    f = RegularGridInterpolator((x0lin, y0lin, z0lin), im0, method=interpmethod, bounds_error=False, fill_value=0)

    # New voxelsize
    h = fov0 / dim

    # Create a regular grid where we want the new values
    x, y, z, xlin, ylin, zlin = gengrid.centered3D(dim, h)

    # Interpolating points
    x = x.flatten()
    x0min = x0lin.min()
    x0max = x0lin.max()
    x[x < x0min] = x0min
    x[x > x0max] = x0max

    y = y.flatten()
    y0min = y0lin.min()
    y0max = y0lin.max()
    y[y < y0min] = y0min
    y[y > y0max] = y0max

    z = z.flatten()
    z0min = z0lin.min()
    z0max = z0lin.max()
    z[z < z0min] = z0min
    z[z > z0max] = z0max

    # Interpolate
    im = f((x, y, z)).reshape(dim)
    #show3D_2(im0[:,:,61], im[:,:,28])

    return im, h


# Repeated 3D interpolations in 4D
def imresize4D(im0, h0, dim, interpmethod):

    dim0 = im0.shape
    ntime = dim0[3]

    im = np.zeros((dim[0], dim[1], dim[2], ntime), dtype='float32')
    for i in np.arange(ntime):
        # Reshape into 3D array
        imh, h = imresize3D(im0[:, :, :, i], h0, dim, interpmethod)
        imh = imh.squeeze()
        im[:, :, :, i] = imh

    return im, h


def imresize2D(im0, h0, dim):

    h0 = h0[0:2]
    dim = dim[0:2]

    # Image size
    dim0 = im0.shape
    dim0 = dim0[0:2]

    # Create a regular grid
    x0, y0, x0lin, y0lin = gengrid.centered2D(dim0, h0)

    # Require the same FOV
    fov0 = dim0 * h0

    # New voxelsize
    h = fov0 / dim

    # Create a regular grid where we want the new values
    x, y, xlin, ylin = gengrid.centered2D(dim, h)

    # Interpolation
    f = RectBivariateSpline(x0lin, y0lin, im0)
    x = x.flatten()
    y = y.flatten()

    # Must revert the order of x and y, otherwise its rotated
    im = f.ev(y, x).reshape(dim[0], dim[1])

    return im



