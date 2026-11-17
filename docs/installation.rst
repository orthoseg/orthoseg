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

Install orthoseg and dependencies
---------------------------------

If you want to run orthoseg on a GPU (optional), make sure you have installed the
necessary drivers, and that the CUDA support is working.

If you run into problems anywhere in this procedure, it can be useful to also have a
look at the tensorflow installation instructions on https://www.tensorflow.org/install

Deep-learning backends
----------------------

orthoseg makes use of keras for its deep-learning functionality. As a consequence,
orthoseg also supports multiple backends. When using Keras 2, only TensorFlow is
supported. When using Keras 3, TensorFlow, Torch and JAX are all supported.

If more than one backend is installed, Orthoseg prefers TensorFlow by default. To
select another backend, set ``KERAS_BACKEND`` before starting Python, for example
``KERAS_BACKEND=torch`` or ``KERAS_BACKEND=jax``.


On Linux or WSL2
================

On Linux or WSL2, all dependencies can be installed from conda-forge, which should lead
to the most reliable setup.

You can create a new environment with orthoseg and its dependencies installed with the
following command:

.. parsed-literal::

    conda env create -n orthoseg -f |environment|
    conda activate orthoseg


To use a CUDA GPU, also install the necessary CUDA packages in the environment: ::

    conda install -y cuda-toolkit=12.3 cudnn


On native Windows
=================

For native Windows, TensorFlow is not published on conda-forge. If you choose the
TensorFlow backend, install it using pip.

If you want to run Orthoseg on CPU with TensorFlow, you can create a new environment
with Orthoseg and its dependencies installed with the following command:

.. parsed-literal::

    conda env create -n orthoseg -f |environment-tf-pip|
    conda activate orthoseg


If you want to run Orthoseg on native Windows with a CUDA GPU using TensorFlow, things
are a bit more complicated because builds of recent TensorFlow versions don't support
this.

Nonetheless, it is possible to get it working, e.g. by using an older version of
tensorflow (2.10) and some older versions of other dependencies. The combinations of
(older) versions are a bit sensitive, and using old versions of software is never
recommended, so another setup (linux or WSL2) is recommended, but these commands created
a working environment at the time of writing:

.. parsed-literal::

    conda env create -n orthoseg -f |environment-win-gpu|
    conda activate orthoseg


On MacOS
========

On macOS, conda-forge packages exist for TensorFlow, but they can lead to crashes in
Orthoseg. If you choose the TensorFlow backend, installing TensorFlow with pip is
recommended.

You can create a new environment with orthoseg and its dependencies installed with the
following command:

.. parsed-literal::

    conda env create -n orthoseg -f |environment-tf-pip|
    conda activate orthoseg


To use a CUDA GPU, also install the necessary CUDA packages in the environment: ::

    conda install -y cuda-toolkit=12.3 cudnn


.. _miniforge installer : https://github.com/conda-forge/miniforge#miniforge3
