# Create environment
conda env update -f=environment.yml

# Activate environment
conda activate envECPaper

# Set python path
export PYTHONPATH=$PYTHONPATH:3DUnetCNN-master

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
# -maskfield: The header field in csv file containing the mask paths 
# -imfield: The header fields in csv file containing the image paths, separated by spaces 
# -outpath: Full path of output file written with texture variables 
# Example:
python3 tools.python/printvars.py 
	-listdata data/ListdataTexture.csv 
	-maskfield mask 
	-imfield vibe2min ADC b1000
	-outpath results/FeatsVibe2min.csv

# --------------------------------------------------
# Do clustering and plotting of heatmap in Matlab
# -------------------------------------------------
# Start matlab, then run 
setpath;
analysegeneticdata;
