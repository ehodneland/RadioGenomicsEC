from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np



def show3D_1(im):
    # figure axis setup 
    fig, ax = plt.subplots(1)
    fig.subplots_adjust(bottom=0.15)
    minval = np.amin(im)
    maxval = np.amax(im)
    
    d = np.shape(im)
    dim = np.array(d).astype('int')
    ndim = len(dim)
    
    # display initial image    
    if ndim > 2:
        idx = np.floor(dim[2]/2).astype('int')
        im_h = ax.imshow(im[:,:,idx], cmap='gray', vmin=minval, vmax=maxval)        
        # update the figure with a change on the slider 
        def update_depth(val):
            slider_depth.val = int(round(slider_depth.val))
            idx = slider_depth.val        
            im_h.set_data(im[:, :, idx])        

        # setup a slider axis and the Slider
        ax_depth = plt.axes([0.23, 0.02, 0.56, 0.04])
        slider_depth = Slider(ax_depth, 'depth', 0, im.shape[2]-1, valinit=idx)
    
        slider_depth.on_changed(update_depth)
    else:
        im_h = ax.imshow(im, cmap='gray', vmin=minval, vmax=maxval)        
    
    
    plt.show()

def show3D_2(im,im2):
    # figure axis setup 
    fig, ax = plt.subplots(1,2)
    fig.subplots_adjust(bottom=0.15)

    minval = np.amin(im)
    maxval = np.amax(im)
    minval2 = np.amin(im2)
    maxval2 = np.amax(im2)

    d = np.shape(im)
    dim = np.array(d).astype('int')
    ndim = len(dim)
    
    if ndim > 2:
        # display initial image 
        idx = np.floor(dim[2]/2).astype('int')
        im_h = ax[0].imshow(im[:, :, idx], cmap='gray', interpolation='nearest', vmin=minval, vmax=maxval)
        im2_h = ax[1].imshow(im2[:, :, idx], cmap='gray', interpolation='nearest', vmin=minval2, vmax=maxval2)
    
        # update the figure with a change on the slider 
        def update_depth(val):
            slider_depth.val = int(round(slider_depth.val))
            idx = slider_depth.val
            im_h.set_data(im[:, :, idx])
            im2_h.set_data(im2[:, :, idx])

        # setup a slider axis and the Slider
        ax_depth = plt.axes([0.23, 0.02, 0.56, 0.04])
        slider_depth = Slider(ax_depth, 'depth', 0, im.shape[2]-1, valinit=idx)
    
        slider_depth.on_changed(update_depth)
    else:
        im_h = ax[0].imshow(im, cmap='gray', interpolation='nearest', vmin=minval, vmax=maxval)
        im2_h = ax[1].imshow(im2, cmap='gray', interpolation='nearest', vmin=minval2, vmax=maxval2)

    plt.show()

