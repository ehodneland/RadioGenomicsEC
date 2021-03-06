# Create environment
conda env update -f=environment.yml

# Activate environment
conda activate envECPaper

# Set python path
export PYTHONPATH=$PYTHONPATH:3DUnetCNN-master

# ---------------------------------------------------
# Do training of CNN network (NB lists of image files must be provided and are not included in the current repo)
# ---------------------------------------------------
# python3 tools.python/trainUNet3D.py: main file
# -networkdir: Directory for export of network
# -gpunum: gpu numger to use {0,1,...}
# -parfile: parameter file (provided in tools.python)
# -setupfile: setupfile with paths to list of images and segmentations (example file provided in tools.python, but the lists themselves must be provided)
python3 tools.python/trainUNet3D.py 
	-networkdir=networks/network-DEMO 
	-gpunum=1 
	-parfile=tools.python/settingsUNet3D.py
        -setupfile=tools.python/setupvibe.py

# ---------------------------------------------------
# Do segmentation of tumor (NB path to image data files must be replaced by local copies)
# --------------------------------------------------
# python3 tools.python/predictUNet3D.py: main file 
# -network: path to network file (the Keras network is provided in data/model.h5) 
# -gpunum: gpu number to use {0,1,...} 
# -parfile: parameter file (provided in tools.python) 
# -listdata: List of nii images used for prediction (vibe+K 2min) 
# -listpredicted: List of pathnames of images for export of masks 
# Example: 
python3 tools.python/predictUNet3D.py 
	-network data/model.h5 
	-gpunum 0 
	-parfile tools.python/settingsUNet3D.py 
	-listdata data/listdata.txt 
	-listpredicted data/listpredicted.txt

# ---------------------------------------------------
# Export image texture variables (NB path to image data files must be replaced by local copies)
# --------------------------------------------------
# python3 tools/printvars.py  
# -listdata: Path to csv file with image paths 
# -maskfield: Column header of the csv file containing the mask paths 
# -imfield: Column header of the csv file containing the image paths (for multiple images, use space for separating)
# -outpath: Full path of output file of exported texture variables 
# Example:
python3 tools.python/printvars.py 
	-listdata data/ListdataTexture.csv 
	-maskfield mask 
	-imfield vibe2min ADC b1000
	-outpath results/FeatsVibe2min.csv

# --------------------------------------------------
# Load clustering based on texture variables and plotting of heatmap in Matlab
# -------------------------------------------------
# If you want to make any changes, e.g. generating other number of clusters etc, 
# the settings can be found in settingsradiomicdata.m can be played around with if you wan

# Start matlab, then run 
setpath;
analyseradiomicdata;

