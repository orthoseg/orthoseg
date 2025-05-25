
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


If you want to run orthoseg on CPU, the following command installs orthoseg
and all its dependencies: ::

    conda install -y python=3.12 pip gdal gdown "geofileops>=0.6,<0.11" "geopandas-base>=0.12,<1.1" matplotlib-base numpy owslib pillow pycron "pygeoops>=0.2,<0.5" pyproj rasterio "shapely>=2" simplification
    pip install orthoseg


If you want to run orthoseg on a GPU, and you haven't run (python) application using
CUDA GPU accelleration before, it is useful to also have a look at the tensorflow
installation instructions on https://www.tensorflow.org/install

Once you have installed the necessary NVIDIA drivers, you can use the following
instructions to install orthoseg and its dependencies in the conda environment.

When NOT using native windows, these commands install the necessary dependencies + 
orthoseg in your new environment: ::

    conda install -y python=3.12 pip cudatoolkit=12.3 cudnn gdal gdown "geofileops>=0.6,<0.11" "geopandas-base>=0.12,<1.1" matplotlib-base numpy owslib pillow pycron "pygeoops>=0.2,<0.5" pyproj rasterio "shapely>=2" simplification
    pip install orthoseg


If you want to run orthoseg on native Windows while using a GPU, things are a bit more
complicated because builds of recent versions of tensorflow don't support this.
Notheless, it is possible to get it working, e.g. by using an older version of
tensorflow (2.10) and some older versions of other dependencies. The combinations of
(older) versions are a bit sensitive, and using old versions of software is never
recommended, so another setup (linux or WSL2) is recommended, but these commands created
a working environment at the time of writing: ::

    conda install -y python=3.10 pip "cudatoolkit>=11.2,<11.3" cudnn gdal gdown "geofileops>=0.6,<0.11" "geopandas-base>=0.12,<1.1" matplotlib-base "numpy<2" owslib pillow pycron "pygeoops>=0.2,<0.5" pyproj rasterio "shapely>=2" simplification "h5py<3.11"
    pip install "tensorflow<2.11"
    pip install orthoseg


.. _miniforge installer : https://github.com/conda-forge/miniforge#miniforge3
