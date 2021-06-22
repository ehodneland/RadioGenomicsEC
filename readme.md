# Create environment
conda env update -f=environment.yml

# Activate environment
conda activate envECPaper

# Set python path
export PYTHONPATH=$PYTHONPATH:3DUnetCNN-master

# ---------------------------------------------------
# Do segmentation of tumor (NB path to image data files must be replaced by local copies)
# --------------------------------------------------
python3 tools.python/predictUNet3D.py 
	-network data/model.h5 
	-gpunum 0 
	-parfile tools.python/settingsUNet3D.py 
	-listdata data/listdata.txt 
	-listpredicted data/listpredicted.txt

# ---------------------------------------------------
# Export image texture variables (NB path to image data files must be replaced by local copies)
# --------------------------------------------------
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
