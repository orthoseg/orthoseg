# Orthophoto segmentation

This is a collection of scripts that makes it easy to train neural networks to segment orthophoto's like aerial photo's, satellite photo's,...

Installation

1. As the scripts are written in Python, you need to use a package manager to be able to install
the packages the scripts depend on. The rest of the installation manual assumes you use anaconda and
python 3.6+. The installer for anaconda can be found here: https://www.anaconda.com/download/.

If you need some more installation instructions, have a look here:
https://conda.io/docs/user-guide/install/index.html

2. Once you have anaconda installed, you can open an anaconda terminal window and follow the
following steps:

      1. Create and activate a new conda environment
      Remark: at time of writing, keras doesn't support 3.7 yet.
      ```
      conda create --name autoseg python=3.6
      conda activate autoseg
      ```
      2. Install the dependencies for the scripts:
      I use the conda-forge channel because the packages there are generally 
      better maintained.
      ```     
      conda install --channel conda-forge keras-gpu tensorflow-gpu rasterio geopandas scikit-image owslib spyder
      pip install segmentation-models   # No conda package available
      ```
