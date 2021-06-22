import numpy as np
from showim import show3D_1, show3D_2


def bwsize(faser):

    n = faser.max()
    iterval = np.arange(1, n+1, 1)
    vol = np.zeros(n, dtype='int')
    for i in iterval:
        reghere = faser == i
        vol[i-1] = reghere.sum()

    return vol, iterval
