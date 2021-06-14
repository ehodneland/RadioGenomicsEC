import numpy as np
from showim import show3D_1, show3D_2


def swapdim(im, h):

    dim = im.shape[0:3]
    h = np.array(h, dtype='float32')

    ax = list()
    if dim[2] > dim[1]:
        ax = [1, 2]

    if dim[2] > dim[0]:
        ax = [0, 2]

    if len(ax) > 0:
        # Flip the image axes
        im = np.swapaxes(im, ax[0], ax[1])

        # Flip the voxelsize also
        h[ax] = h[np.flip(ax)]

        print('Swapping axes ' + str(ax[0]) + ' with ' + str(ax[1]))

    dim = im.shape[0:3]

    # New shape after swapping axes?
    print('Shape of data after swapping: ' + str(im.shape))
    print('Voxelsize of data after swapping: ' + str(h))

    return im, h, ax, dim
