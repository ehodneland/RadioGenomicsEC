import numpy as np
from showim import show3D_1, show3D_2

# Grid orientation:
# x = increasing rows
# y = increasing columns


def centered2D(dim, h):

    # Generating centred coordinates of the mesh
    x1 = np.linspace(0, h[0]*(dim[0]-1), dim[0])
    x1 = x1 - np.mean(x1)
    x2 = np.linspace(0, h[1]*(dim[1]-1), dim[1])
    x2 = x2 - np.mean(x2)
    y, x = np.meshgrid(x2, x1)

    return x, y, x1, x2


def centered3D(dim, h):

    # Generating centred coordinates of the mesh
    x1 = np.linspace(0, h[0]*(dim[0]-1), dim[0])
    x1 = x1 - np.mean(x1)
    x2 = np.linspace(0, h[1]*(dim[1]-1), dim[1])
    x2 = x2 - np.mean(x2)
    x3 = np.linspace(0, h[2]*(dim[2]-1), dim[2])
    x3 = x3 - np.mean(x3)
    y, x, z = np.meshgrid(x2, x1, x3)

    return x, y, z, x1, x2, x3