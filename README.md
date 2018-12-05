# auto_segmentation

Installation

1. As the scripts are written in Python, you need to use a package manager to be able to install
the packages the scripts depend on. The rest of the installation manual assumes you use anaconda and
python 3.6+. The installer for anaconda can be found here: https://www.anaconda.com/download/.

If you need some more installation instructions, have a look here:
https://conda.io/docs/user-guide/install/index.html

2. Once you have anaconda installed, you can open an anaconda terminal window and follow the
following steps:

      1. Create and activate a new conda environment
      ```
      conda create --name auto_segmentation #python=3.6
      conda activate auto_segmentation
      ```
      2. Install the dependencies for the scripts:
      ```
      conda install tensorflow
      conda install keras
      conda install pandas
      conda install rasterio
      conda install shapely
      conda install fiona
      conda install scikit-image
      conda install owslib
      pip install segmentation-models
      conda install spyder           # If you use spyder as development tool...
      ```