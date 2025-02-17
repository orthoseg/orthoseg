
============
Installation
============

orthoseg is written in python, but it relies on several other libraries that have
dependencies written in C/C++. Those dependencies can be difficult to install, but
luckily the conda package management system also gives an easy alternative.

Python package/environment manager
----------------------------------

If you don't have conda installed yet, you can install it using the
`miniforge installer`_.

Install orthoseg and depencencies
---------------------------------

Now start the conda prompt and create a new conda environment with the following
commands: ::

    conda create -n orthoseg
    conda activate orthoseg


If you use e.g. anaconda or miniconda instead of a miniforge installation, also run
following commands to specify that all depencencies should be installed from the
conda-forge channel. Mixing packages from multiple channels is bound to give problems
sooner or later: ::

    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict


Finally, you can install orthoseg: ::

    conda install -y python=3.10 pip "cudatoolkit>=11.2,<11.3" cudnn gdown "geofileops>=0.6,<0.10" "geopandas-base>=0.12,<1.1" matplotlib-base "numpy<2" owslib pillow pycron "pygeoops>=0.2,<0.5" pyproj rasterio "shapely>=2" simplification "h5py<3.11"
    pip install orthoseg


.. _miniforge installer : https://github.com/conda-forge/miniforge#miniforge3
