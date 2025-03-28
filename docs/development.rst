
===========
Development
===========

The source code can be found on the |orthoseg git repository|.

If you want to do feature requests or file bug reports, that's the place to 
be as well.

Create development environment
------------------------------

The first step would be to fork and clone the |orthoseg git repository|, to be able to
run/debug the code.

The easiest way to install the dependencies of orthoseg and run it is to use the
conda package manager. If you don't have it installed yet, here is a link to the 
`miniforge installer`_.

Start the conda prompt and create a new conda environment with the following
commands: ::

    conda env create -f environment-dev.yml
    conda activate orthoseg-dev


If you use e.g. anaconda or miniconda instead of a miniforge installation, also run
following commands to specify that all depencencies should be installed from the
conda-forge channel. Mixing packages from multiple channels is bound to give problems
sooner or later: ::

    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict


It is recommended to install the pre-commit hook that will take care of the linting
checks (ruff) and formatting (ruff-format), so you don't run into linting errors when
you create a pull request: ::

    pre-commit install


.. _miniforge installer : https://github.com/conda-forge/miniforge#miniforge3

.. |orthoseg git repository| raw:: html

   <a href="https://github.com/orthoseg/orthoseg" target="_blank">orthoseg git repository</a>
