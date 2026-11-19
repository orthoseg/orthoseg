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

orthoseg makes use of keras for its deep-learning functionality. Because Keras supports
multiple backends, orthoseg tries to offer this as well. When using Keras 2, only
TensorFlow is supported. When using Keras 3, TensorFlow, Torch and JAX are all
supported.

Note that based on some (limited) tests, the TensorFlow backend tends to be the most
stable and performant at the time of writing, so the rest of the installation
instructions assume the use of the TensorFlow backend. If you want to use another
backend, you can modify one of the conda environment files accordingly.

If more than one backend is installed, Orthoseg prefers TensorFlow by default. To
select another backend, set ``KERAS_BACKEND`` before importing/starting orthoseg, for
example: ``KERAS_BACKEND=torch`` or ``KERAS_BACKEND=jax``.


On Linux, WSL2 or MacOS
=======================

On Linux, WSL2 or MacOS, all dependencies can be installed from conda-forge, which
should lead to the most reliable setup.

If you don't want GPU/CUDA support, you can create a new environment with orthoseg and
its dependencies installed with the following commands:

.. parsed-literal::

    conda env create -n orthoseg -f |environment-tf|
    conda activate orthoseg


On Linux or WSL2, you can also use a CUDA GPU. In that case, use the following
commands: ::

.. parsed-literal::

    conda env create -n orthoseg -f |environment-tf-gpu|
    conda activate orthoseg


On native Windows, without cuda GPU support
===========================================

For native Windows, recent versions of TensorFlow are not available on conda-forge.
If you choose the TensorFlow backend, install it using pip.

If you want to run Orthoseg on CPU with TensorFlow, you can create a new environment
with Orthoseg and its dependencies installed with the following command:

.. parsed-literal::

    conda env create -n orthoseg -f |environment-tf-pip|
    conda activate orthoseg


On native Windows, with cuda GPU support
========================================

If you want to run Orthoseg on native Windows with a CUDA GPU using TensorFlow, things
are a bit more complicated because builds of recent TensorFlow versions don't support
this. Because of this, if you have the option/choice available to run on linux, it is
recommended to do so.

Nonetheless, it is possible to get it working, e.g. by using a (very) old version of
tensorflow (2.10) and some older versions of other dependencies. The combinations of
(older) versions are a bit sensitive, and using old versions of software is never
recommended, but these commands created a working environment at the time of writing:

.. parsed-literal::

    conda env create -n orthoseg -f |environment-tf-gpu-win|
    conda activate orthoseg


.. _miniforge installer : https://github.com/conda-forge/miniforge#miniforge3
