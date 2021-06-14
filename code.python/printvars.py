import numpy as np
import os
import config
import nibabel as nib
from loadimUNet3D import loadim
from scipy.stats import itemfreq, kurtosis, skew, entropy
from datastruct import Struct
#from showim import show3D_1, show3D_2
import scale
import re
import pandas as pd
import argparse
from greyrlmatrix import *
from greyrlprops import *
import pathfun
import str2bool
from imresize import imresize4D
from imresize import imresize3D
import bwsize
from skimage import measure

parser = argparse.ArgumentParser(description='Label training data')
parser.add_argument('-listdata', action='store', nargs='+')
parser.add_argument('-maskfield', action='store', nargs='+')
parser.add_argument('-imfield', action='store', nargs='*')
parser.add_argument('-outpath', action='store', nargs=1)

parser.add_argument('-useslice', action='store', nargs=1)
#parser.add_argument('-resize', action='store', nargs=1, default=False)
parser.add_argument('-largestonly', action='store', nargs=1, default=False)
args = parser.parse_args()

# -----------------------------------------
# Mandatory arguments
# -----------------------------------------

# CSV file to file with image paths
listdata = args.listdata[0]

# Mask and image field
maskfield = args.maskfield[0]
imfield = args.imfield

# Export file
outpath = args.outpath[0]


# -----------------------------------------
# Optional arguments
# -----------------------------------------

# Number of slices, either 'all' or 'one'
try:
    useslice = args.useslice[0]
except:
    useslice = 'one'

# Largest object only?
try:
    largestOnly = str2bool.str2bool(args.largestonly[0])
except:
    largestOnly = True

# Prediction threshold
predth = 0.5

# Resize images?
resize = False

# Create output dir if not exists
outdir, e, base, ext = pathfun.get(outpath)
if not os.path.exists(outdir):
    os.mkdir(outdir)

# Dimension of reference image to be used
dimref = (192, 192, 48)

# ------------------------------------------------
# Read data file
# ------------------------------------------------
df = pd.read_csv(listdata)
subj = df.subj
pathmask = df[maskfield]

# Number of subjects
nsubj = len(subj)

# Number of sequences
nseq = len(imfield)

# List of image data into a multidimensiontal np array
pathdata = np.empty([nsubj, nseq], dtype=object)
for idx, s in enumerate(imfield):
    pathdata[:, idx] = df[s]

# The imagevar properties
imagevar = Struct()
imagevar.stats = Struct()
imagevar.stats.fn = np.array(['vol', 'metastasis', 'meanint', 'mean15perc', 'normsurfvolratio', 'clustindex', 'clustersize',
                              'stdint', 'kurtosis', 'skewness', 'entropy', 'gaborvar'], dtype=object)

imagevar.stats.n = imagevar.stats.fn.size
imagevar.stats.varlen = nseq*np.ones(imagevar.stats.n, dtype='int')
imagevar.stats.varlen[imagevar.stats.fn == 'vol'] = 1
imagevar.stats.varlen[imagevar.stats.fn == 'normsurfvolratio'] = 1

# Gabor filter properties
imagevar.gabor = Struct()
imagevar.gabor.fn = np.array(['gaborvar'], dtype=object)
imagevar.gabor.n = imagevar.gabor.fn.size
imagevar.gabor.varlen = nseq*np.ones(imagevar.gabor.n, dtype='int')
# Orientations in radians
imagevar.gabor.theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]
# Standard deviations
imagevar.gabor.sigma = [1, 3]
# Frequencies
imagevar.gabor.frequency = [0.10, 0.25]

# GLCM properties
imagevar.glcm = Struct()
imagevar.glcm.fn = np.array(['contrast', 'homogeneity', 'energy', 'correlation'], dtype=object)
#imagevar.glcm.fn = np.empty(0, dtype=object)
imagevar.glcm.n = imagevar.glcm.fn.size
imagevar.glcm.varlen = nseq*np.ones(imagevar.glcm.n, dtype='int')
# Jump intervals
imagevar.glcm.d = [1, 3, 5]
# Orientations in radians
imagevar.glcm.theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]
# Number of intervals in GLCM
imagevar.glcm.m = 24

# GLRLM properties
imagevar.glrlm = Struct()
# Short run emphasis, long run emphasis, low grey level run emphasis, high grey level run emphasis
imagevar.glrlm.fn = np.array(['SRE', 'LRE', 'LGRE', 'HGRE'], dtype=object)
imagevar.glrlm.n = imagevar.glrlm.fn.size
imagevar.glrlm.varlen = nseq*np.ones(imagevar.glrlm.n, dtype='int')
# Number of intervals in GLRLM
imagevar.glrlm.m = 8
# Orientation in radians
imagevar.glrlm.theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Genetic data
geneticvar = Struct()

# compute Gabor features
def compute_gabor_feats(image, kernels):
    from scipy import ndimage as ndi

    feats = np.zeros((len(kernels), 1), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        #show3D_2(image, filtered)
        #feats[k, 0] = filtered.mean()
        feats[k] = filtered.var()
    return feats

# Get image variables
def getImageVar(imagevar, pathdata, pathmask, useslice, resize, dimref):

    from skimage.feature import greycomatrix, greycoprops
    import swapdim
    from sklearn.cluster import KMeans
    from scipy import ndimage
    from skimage import measure
    from skimage import morphology
    from bwsize import bwsize
    from skimage.filters import gabor_kernel

    # Initialize empty image data arrays
    keyi = 0
    for key in imagevar.fn:
        imagevar[key] = np.empty((imagevar.nsubj, imagevar.varlen[keyi]), dtype='float32')
        imagevar[key][:] = np.nan
        keyi = keyi + 1

    # Load image data and compute image features within the ROIs
    si = 0

    v = imagevar.subj[si:]
    for s in v:

        print("New subject " + str(s))

        # Load data
        swap = True
        # resize = True
        im4D, nii3D, h, swapax, swapdims = loadim(pathdata[si, :], swap=swap)
        h = h[0, :]
        dim4D = im4D.shape

        flag = False
        if resize:
            if not dim4D[1:] == dimref:
                im4D, h = imresize4D(im4D, h, dimref, 'linear')
                dim4D = im4D.shape
                flag = True

        # Load mask
        filepath = pathmask[si]
        filepath = re.sub(r"\s+", "", filepath)

        print('Loading ' + filepath)
        nii = nib.load(filepath)
        header = nii.header
        predim = nii.get_fdata()
        predim = np.array(predim, dtype='float')

        if flag:
            predim, aaa = imresize3D(predim, h, dimref, 'nearest')

        # Ensure boolean
        predim = predim > predth

        # Swap dimensions?
        if swap:
            predim, h, swapax, swapdims = swapdim.swapdim(predim, h)

        dimpredim = predim.shape

        # --------------------------------------------------------------------------
        # If nslice == 1 we find the slice of maximum extent of the tumor and use
        # this for resembling a 2D segmentation
        # --------------------------------------------------------------------------
        if useslice == 'one':
            c = np.where(predim > 0)
            c = c[2]
            counts = np.bincount(c)
            if len(counts) == 0:
                slice == np.round(dimpredim[2]/2)
            else:
                slice = np.argmax(counts)
            print('Picking slice of largest tumor extent: ' + str(slice))
            im4D = im4D[:, :, :, slice].reshape(dim4D[0], dim4D[1], dim4D[2], 1)
            predim = predim[:, :, slice].reshape(dimpredim[0], dimpredim[1], 1)

        if useslice == 'one':
            voxelvol = np.prod(h[0:2])
        else:
            voxelvol = np.prod(h)

        # Number of objects
        faser, L = ndimage.label(predim)

        # Empty mask? Cannot compute texture features
        if predim.sum() == 0:
            print('Empty mask, continuing...')
            si = si + 1
            continue

        # Number of metastasis?
        imagevar.metastasis[si] = L

        # Only largest component?
        if largestOnly:
            msg = "Number of components: " + str(L)
            print(msg)
            predim = (faser == (np.bincount(faser.flat)[1:].argmax() + 1))

        faser, L = ndimage.label(predim)
        msg = "Number of components: " + str(L)
        print(msg)

        # Any negative values? GLCM does not handle it
        for seqi in np.arange(nseq):
            minval = np.min(im4D[seqi, :, :, :])
            if minval < 0:
                imh = im4D[seqi, :, :, :]
                imh[imh < 0] = 0
                im4D[seqi, :, :, :] = imh

        # Volume in ml
        if 'vol' in imagevar.stats.fn:
            vol = voxelvol * predim.sum() / 1e3
            imagevar.vol[si] = vol

        if 'normsurfvolratio' in imagevar.stats.fn:
            eroded = ndimage.binary_erosion(predim)
            perim = predim.copy()
            perim[eroded] = False
            # The radius if the tumor was a sphere
            vol = predim.sum()
            surf = perim.sum()
            r = np.power((3*vol)/(4*np.pi), 1/3)
            # Go to a unit sphere
            # V = Vbar*r³
            volbar = vol/np.power(r, 3)
            # S = Sbar*r²
            surfbar = surf / np.power(r, 2)
            imagevar.normsurfvolratio[si] = surfbar/volbar

        for seqi in np.arange(nseq):

            # The 3D image
            imh = im4D[seqi, :, :, :]

            # The image values within the masked tumor
            imvalraw = imh[predim]

            # Sort the image values
            imval = np.sort(imvalraw)

            if predim.sum() > 1:
                #  Kmeans clustering
                if ['clustindex' or 'clustersize'] in imagevar.stats.fn:

                    n = len(imval)
                    val = np.reshape(imvalraw, (n, 1))
                    ncluster = 2
                    conn = 2
                    kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(val)
                    clustim = np.zeros(imh.shape, dtype='int')
                    clustim[predim] = 1 + kmeans.labels_
                    maxn = np.zeros(1, dtype='int')
                    clustsize = np.empty((0, 1), dtype='int')
                    clustval = np.arange(1, ncluster+1)

                    for i in clustval:
                        reghere = clustim == i
                        # Remove very small objects, these are noise of error in segmentation
                        reghere = morphology.remove_small_objects(reghere, min_size=3, connectivity=conn)
                        label = measure.label(reghere, connectivity=conn)
                        m = label.max()
                        vols, iterval = bwsize(label)
                        nvols = len(iterval)
                        vols = np.reshape(vols, (nvols, 1))
                        print('Number of regions in cluster ' + str(i) + ': ' + str(m))
                        maxn = maxn + m
                        clustsize = np.concatenate((clustsize, vols), axis=0)

                # Assign to matrix
                if 'clustersize' in imagevar.stats.fn:
                    imagevar.clustersize[si, seqi] = clustsize.mean()

                if 'clustindex' in imagevar.stats.fn:
                    imagevar.clustindex[si, seqi] = maxn

            else:
                imagevar.clustersize[si, seqi] = np.nan
                imagevar.clustindex[si, seqi] = np.nan

            # Mean of intensity
            if 'meanint' in imagevar.stats.fn:
                imagevar.meanint[si, seqi] = imval.mean()

            # Mean of the X-percentile from below
            if 'mean15perc' in imagevar.stats.fn:
                n = len(imval)
                stop = np.round(0.15*n).astype('int')
                valperc = imval[0:stop]
                imagevar.mean15perc[si, seqi] = valperc.mean()

            # Std of intensity
            if 'stdint' in imagevar.stats.fn:
                imagevar.stdint[si, seqi] = imval.std()

            # Kurtosis of intensity
            if 'kurtosis' in imagevar.stats.fn:
                imagevar.kurtosis[si, seqi] = kurtosis(imval)

            # Skewness of intensity
            if 'skewness' in imagevar.stats.fn:
                imagevar.skewness[si, seqi] = skew(imval)

            # Entropy of intensity
            if 'entropy' in imagevar.stats.fn:
                hist, bin_edges = np.histogram(imval, bins=30, density=True)
                imagevar.entropy[si, seqi] = entropy(hist)

        # --------------------------------------------------------------------------
        # Compute Gabor filters
        # --------------------------------------------------------------------------
        kernels = []
        nfilter = 0
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                  sigma_x=sigma, sigma_y=sigma))
                    kernels.append(kernel)
                    nfilter = nfilter + 1

        # --------------------------------------------------------------------------
        #    Compute gray level co-occurence matrix within the tumor
        # --------------------------------------------------------------------------
        for seqi in np.arange(nseq):
            im3D = im4D[seqi, :, :, :]

            # Clip data
            c = np.where(predim > 0)
            start = np.zeros(3, dtype='int')
            stop = np.zeros(3, dtype='int')
            for ii in np.arange(3):
                start[ii] = np.min(c[ii])
                stop[ii] = np.max(c[ii])+1
            im3D = im3D[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            mask3D = predim[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            im3D[np.invert(mask3D)] = 0
            maxval = im3D.max()
            nslice = im3D.shape[2]

            glcm = Struct()
            glcm.mat = np.zeros((imagevar.glcm.m, imagevar.glcm.m), dtype='float32')
            glcm.im3D = np.intc(imagevar.glrlm.m * im3D / (maxval + 1e-6))

            glrlm = Struct()
            glrlm.uval = np.arange(imagevar.glrlm.m + 1)
            glrlm.maxrl = im3D.shape[0]
            glrlm.mat = np.zeros((len(glrlm.uval)-1, glrlm.maxrl), dtype='float32')
            glrlm.im3D = np.intc(imagevar.glrlm.m * im3D / (maxval + 1e-6))

            gabor = Struct()
            gabor.mat = np.zeros((nfilter, imagevar.gabor.n), dtype='float32')

            # Greycomatrix only works for 2D matrices
            for slice in np.arange(nslice):

                # Compute GLCM
                mat = greycomatrix(glcm.im3D[:, :, slice], imagevar.glcm.d, imagevar.glcm.theta, imagevar.glcm.m + 1,
                                       symmetric=True, normed=False)

                # Remove the zero values since they are outside the mask
                mat = mat[1:, 1:, :, :]

                # Take the mean over theta and jump intervals
                mat = np.nanmean(mat, axis=(2, 3))

                # Add to "3D" glcm matrix
                glcm.mat = glcm.mat + mat

                # GLRLM run length matrix
                mat = greyrlmatrix(glrlm.im3D[:, :, slice], glrlm.uval, glrlm.maxrl, imagevar.glrlm.theta)

                # Take away the zeros, which are background
                mat = mat[1:, :, :]

                # Take the mean over orientations
                mat = np.mean(mat, axis=2)

                # Add to "3D" glrlm matrix
                glrlm.mat = glrlm.mat + mat

                # Compute Gabor filters
                feats = compute_gabor_feats(im3D[:, :, slice], kernels)
                feats = feats/np.mean(im3D[:, :, slice])
                gabor.mat = gabor.mat + feats

            # Normalize to sum 1
            glcm.mat = glcm.mat / glcm.mat.sum()
            glrlm.mat = glrlm.mat / glrlm.mat.sum()

            # Average values of Gabor filter
            gabor.mat = gabor.mat / nfilter

            # Compute glcm properties
            d = glcm.mat.shape
            glcm.mat = np.reshape(glcm.mat, (d[0], d[1], 1, 1))
            for key in imagevar.glcm.fn:
                out = greycoprops(glcm.mat, key)
                imagevar[key][si, seqi] = out[0, 0]

            # Compute glrlm properties
            for key in imagevar.glrlm.fn:
                imagevar[key][si, seqi] = greyrlprops(glrlm.mat, key)

            # Compute varianve of gabor response
            for key in imagevar.gabor.fn:
                imagevar[key][si, seqi] = gabor.mat.var()

            # Add counter for features
            seqi = seqi + 1

        si = si + 1

    return imagevar


# Make folders
if not os.path.exists(outdir):
    os.mkdir(outdir)

# For saving of plots
curfile = os.path.basename(__file__)
curfile, ext = os.path.splitext(curfile)

# Assign list of subjects
imagevar.subj = np.asarray(subj)
imagevar.nsubj = len(imagevar.subj)

# Combine
imagevar.varlen = np.concatenate((imagevar.stats.varlen, imagevar.glcm.varlen, imagevar.glrlm.varlen))
imagevar.fn = np.concatenate((imagevar.stats.fn, imagevar.glcm.fn, imagevar.glrlm.fn))
imagevar.n = imagevar.fn.size

# Name of predictor variables
n = np.sum(imagevar.varlen)
imagevar.predname = np.empty(n, dtype=object)
keyi = 0
c = 0
for key in imagevar.fn:
    j = 0
    for j in np.arange(imagevar.varlen[keyi]):
        imagevar.predname[c] = key + '.' + imfield[j]
        c = c + 1
    keyi = keyi + 1

################### Load data and extract image variables

# Get image variables
imagevar = getImageVar(imagevar, pathdata, pathmask, useslice, resize, dimref)

# Collect the image data into one matrix
nrow = imagevar.nsubj
imagevar.dataset = Struct()
imagevar.dataset.val = np.empty((nrow, 0), dtype='float32')
for key in imagevar.fn:
    val = imagevar[key]
    imagevar.dataset.val = np.concatenate((imagevar.dataset.val, val), axis=1)

# Assign updated row names to matrix
imagevar.dataset.rowname = imagevar.subj.reshape(imagevar.nsubj, 1)
imagevar.dataset.colname = imagevar.predname

################### Print variables to text file ####################3
n = len(imagevar.dataset.colname)
header = np.empty(n+1, dtype=object)
header[0] = 'Subj'
header[1:] = imagevar.dataset.colname
dataarr = np.concatenate((imagevar.dataset.rowname, imagevar.dataset.val), axis=1)
dataarr = pd.DataFrame(data=dataarr, columns=header)
print('Raw variables: ')
print(dataarr)
#pathsave = os.path.join(outdir, curfile + addstr + '-imagevar' + '.csv')
print('Saving ' + outpath)
dataarr.to_csv(outpath, header=header, sep=',', float_format='%.4f', na_rep='NaN')
