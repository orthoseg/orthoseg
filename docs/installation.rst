
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

If you want to run orthoseg on a GPU (optional), make sure you have installed the
necessary drivers, and that the CUDA support is working.

If you run into problems anywhere in this procedure, it can be useful to also have a
look at the tensorflow installation instructions on https://www.tensorflow.org/install

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

Use this conda environment file to install: `environment gpu win`_.

This is the full link: :samp:`environment gpu win`

    conda env create -f


On Linux or WSL2
================

On Linux or WSL2, all dependencies can be installed from conda-forge, which should lead
to the most reliable setup.

Use the following commands to install geofileops with its depencencies: ::

    conda install -y python=3.12 pip gdal gdown "geofileops>=0.6" "geopandas-base>=0.12" matplotlib-base numpy owslib pillow pycron "pygeoops>=0.2" pyproj rasterio "shapely>=2" simplification "tensorflow=2.19"
    pip install orthoseg

To use the GPU, also install the necessary CUDA packages in the environment: ::

    conda install -y cudatoolkit=12.3 cudnn


On native Windows
=================

For native Windows, no tensorflow packages are published on conda-forge. Hence, we need
to install tensorflow using pip.

If you want to run orthoseg on CPU, use the following commands to install: ::

    conda install -y python=3.12 pip "gdal<3.11" gdown "geofileops>=0.6" "geopandas-base>=0.12" matplotlib-base "numpy<2.2" owslib pillow pycron "pygeoops>=0.2" pyproj rasterio "shapely>=2" simplification
    pip install orthoseg


If you want to run orthoseg on native Windows while using a GPU, things are a bit more
complicated because builds of recent versions of tensorflow don't support this.
Notheless, it is possible to get it working, e.g. by using an older version of
tensorflow (2.10) and some older versions of other dependencies. The combinations of
(older) versions are a bit sensitive, and using old versions of software is never
recommended, so another setup (linux or WSL2) is recommended, but these commands created
a working environment at the time of writing: ::

    conda install -y python=3.10 pip "cudatoolkit>=11.2,<11.3" "cudnn=8" "gdal<3.11" gdown "geofileops>=0.6" "geopandas-base>=0.12" matplotlib-base "numpy<2" owslib pillow pycron "pygeoops>=0.2" pyproj rasterio "shapely>=2" simplification "h5py<3.11"
    pip install "tensorflow<2.11" "numpy<2" orthoseg


On MacOS
========

Use the following commands to install orthoseg: ::

    conda install -y python=3.12 pip "gdal<3.11" gdown "geofileops>=0.6" "geopandas-base>=0.12" matplotlib-base "numpy<2.2" owslib pillow pycron "pygeoops>=0.2" pyproj rasterio "shapely>=2" simplification
    pip install orthoseg

To use the GPU, also install the necessary CUDA packages in the environment: ::

    conda install -y cudatoolkit=12.3 cudnn


.. _miniforge installer : https://github.com/conda-forge/miniforge#miniforge3
.. _environment gpu win : _static/conda_envs/environment-gpu-win.yml
