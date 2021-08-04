import numpy as np


def crop(im, ws, nfeat, collect, spacing):


    dim = im.shape
    ndim = np.ndim(im)
    spacing = spacing * np.ones(3, dtype='int')
    # Work with 3D matrix size
    if ndim == 4:
        dim = dim[1:]

    # Its 3D, not 4D data, we need to add another dimension
    if ndim == 3:
        im = np.reshape(im, (1, dim[0], dim[1], dim[2]))

    # Count number of crops
    imcrop = []
    startcrop = []
    stopcrop = []
    # Window weight for normalization of predictions
    ww = []
    ncrop, imcrop, startcrop, stopcrop, ww = runningindex(im, imcrop, startcrop, stopcrop, ws, False, spacing, ww)

    if not(collect):
        # Only count
        return ncrop, imcrop, startcrop, stopcrop, ww

    # Collect image data and crops
    imcrop = np.zeros([ncrop, nfeat, ws[0], ws[1], ws[2]], dtype='float32')
    startcrop = np.zeros([ncrop, 3], dtype='int')
    stopcrop = np.zeros([ncrop, 3], dtype='int')
    ww = np.zeros([dim[0], dim[1], dim[2]], dtype='float32')
    ncrop, imcrop, startcrop, stopcrop, ww = runningindex(im, imcrop, startcrop, stopcrop, ws, True, spacing, ww)

    return ncrop, imcrop, startcrop, stopcrop, ww


def runningindex(im, imcrop, startcrop, stopcrop, ws, collect, spacing, ww):

    dim = im.shape
    dim = dim[1:]

    # Start in upper left corner
    p0 = np.array([0, 0, 0], dtype='int')

    # Loop over each spatial direction
    ncrop = int(0)

    while True:
        start = p0
        stop = p0 + ws

        if stop[0] > dim[0]:
            p0[0] = 0
            p0[1] = p0[1] + spacing[1]
            continue

        if stop[1] > dim[1]:
            p0[0] = 0
            p0[1] = 0
            p0[2] = p0[2] + spacing[2]
            continue
        if stop[2] > dim[2]:
            break

        if collect:
            # Add new image to array
            imh = im[:, start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            imcrop[ncrop, :, :, :, :] = imh
            startcrop[ncrop, :] = start
            stopcrop[ncrop, :] = stop
            ww[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] = \
                ww[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]] + np.ones(ws, dtype='float32')

        # Count number of crops
        ncrop = ncrop + 1

        # Add counter
        p0[0] = p0[0] + spacing[0]

    if collect:
        # Ensure no zeros
        ww[ww < 1] = 1

    return ncrop, imcrop, startcrop, stopcrop, ww
