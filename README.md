# Orthophoto segmentation

This is a collection of scripts that try to make it easy to train neural networks to segment orthophotos.

## Installation

### Anaconda

As the scripts are written in Python, you need to use a package manager to be able to install
the packages the scripts depend on. The rest of the installation manual assumes you use anaconda and
python 3.6+. The installer for anaconda can be found here: https://www.anaconda.com/download/.

If you need some more installation instructions, have a look here:
https://conda.io/docs/user-guide/install/index.html

### Dependent packages

Once you have anaconda installed, you can open an anaconda terminal window, create a new conda
environment + install needed dependencies like this:
```
conda create --name orthoseg tensorflow-gpu pillow rasterio geopandas owslib
conda activate orthoseg
pip install segmentation-models   # No conda package available
```
