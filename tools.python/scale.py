#!/usr/bin/env python3

import numpy as np

def sc(img,low,high):
        
    # Take to zero           
    img = img - np.amin(img)
        
    # Take to 0,1
    img = img/np.amax(img)
    
    # Scale to difference
    diff = high - low
    img = img*diff

    # Lift up
    img = img + low
    
    return img


