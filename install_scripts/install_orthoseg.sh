#!/bin/bash

# If no parameters passed, show help...
if [ -z "$var" ]
then
  echo
  echo Hello! If you want to override some default options this is possible as such:
  echo 'install_orthoseg.sh --envname orthosegdev --envname_backup orthosegdev_bck_2020-01-01 --condadir "/home/Miniconda3" --fordev Y'
  echo 
  echo The parameters can be used as such:
  echo     - envname: the name the new environment will be given 
  echo     - envname_backup: if the environment already exist, it will be 
  echo       backupped to this environment
  echo     - condadir: the directory where cona is installed
  echo     - fordev: for development: if Y is passed, only the dependencies 
  echo       for orthoseg will be installed, not the orthoseg package itself
fi

# Extract the parameters passed
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--envname) envname="$2"; shift ;;
        -cd|--condadir) condadir="$2"; shift ;;
        -eb|--envname_backup) envname_backup="$2"; shift ;;
        -od|--fordev) fordev="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Format current date
today=$(date +%F)

# If not provided, init parameters with default values
if [ -z "$envname" ]; then envname="orthoseg" ; fi
if [ -z "$envname_backup" ]; then envname_backup="${envname}_bck_${today}" ; fi
if [ -z "$condadir" ]; then condadir="$HOME/Miniconda3" ; fi
if [ -z "$fordev" ]; then fordev="N" ; fi

# If no parameters are given, ask if it is ok to use defaults
echo
echo "The script will be ran with the following parameters:"
echo "   - envname=$envname"
echo "   - envname_backup=$envname_backup"
echo "   - condadir=$condadir"
echo "   - fordev=$fordev"
echo

read -p "Do you want to move on with these choices? (y/n)" -n 1 -r
echo    
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

# Init conda
. "$condadir/etc/profile.d/conda.sh"

#-------------------------------------
# RUN!
#-------------------------------------
echo
echo Backup existing environment
echo -------------------------------------

if [[ ! -z "$envname_backup" ]]
then
  if [[ -d "$condadir/envs/$envname/" ]]
  then
    echo "Do you want to take a backup from $envname?"
    if [[ -d "$condadir/envs/$envname_backup/" ]]
    then
      echo "REMARK: $envname_backup exists already, so will be overwritten by a new backup!"
    fi
    
    read -p "y=take backup, n=don't take backup but proceed, c=stop script (y/n/c)" -n 1 -r
    echo    
    if [[ $REPLY =~ ^[Cc]$ ]]
    then
        [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
    elif [[ $REPLY =~ ^[Yy]$ ]]
    then  
      conda create --name "$envname_backup" --clone "$envname" --offline
    fi
  else 
    echo "No existing environment $envname found to backup"
  fi
fi

echo
echo Create/overwrite environment
echo -------------------------------------
if [[ -d "$condadir/envs/$envname/" ]]
then
  echo "First remove conda environment $envname"
  conda env remove -y --name $envname

  echo "Also really delete the env directory, to evade locked file errors"
  rm -rf $condadir/envs/$envname
fi
echo "Create and install conda environment $envname"
conda create -y --name $envname
conda activate $envname
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

# Reasons for the dependencies/versions...
#
# Remark: the dependencies of tensorflow can be found here: https://libraries.io/pypi/tensorflow
#
# python: 3.8 is highest version supported by tensorflow, and I might use 3.8 syntax somewhere (not sure) 
# owslib: download images from WMS servers
# pillow: ?
# rasterio: tested till version 1.2
# geopandas: >= 0.8 can use pygeos for better performance
# pygeos: for geofileops, improves performance of geopandas operations
# pyproj: for geofileops
# libspatialite: for geofileops, > 5 is (a lot) better 
# cudatoolkit: for tensorflow+GPU, 11.0 needed for tensorflow >= 2.4
# cudnn: for tensorflow+GPU
# numpy: for tensorflow: needs 1.19, otherwise replaced with pip version: gives issues in geofileops 
# h5py: for tensorflow: needs ~=3.1
# hdf5: till tensorflow 2.4 needed 1.10.5, otherwise ugly warnings and BOOM! -> from 2.5 not anymore! 
conda install -y python=3.8 owslib pillow "rasterio>=1.0,<1.3" geofileops "libspatialite>=5.0" geopandas pygeos pyproj "cudatoolkit>=11.0,<11.1" cudnn "numpy>=1.19,<1.20" "h5py>=3.1,<3.2" #"hdf5==1.10.5"

# For the following packages, no conda package is available or -for tensorflow- no recent version.
if [[ ! $fordev =~ ^[Yy]$ ]]
then
  echo
  echo "Install the pip package"
  echo
  pip install "orthoseg>=0.2.2a3"
else
  echo
  echo "Prepare for development: conda install dev tools"
  echo
  conda install -y pylint pytest rope
 
  echo
  echo "Prepare for development: pip install dependencies that need pip"
  echo
  # Reasons for the version specifications...
  # tensorflow: starting from 2.5 compatible with libspatialite 5.0 
  # geofileops: simplify algorythms used supported from 2.0
  pip install pycron "segmentation-models>=1.0,<1.1" "tensorflow>=2.5.0rc1,<2.6" 
fi

# Deactivate new env
conda deactivate

# Clean the cache dir + deactivate base env
#conda clean --all
conda deactivate

# Pause
read -s -n 1 -p "Press any key to continue . . ."
