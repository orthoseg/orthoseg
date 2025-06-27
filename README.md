# Orthophoto segmentation

[![Actions Status](https://github.com/orthoseg/orthoseg/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/orthoseg/orthoseg/actions?query=workflow%3ATests)
[![Coverage Status](https://codecov.io/gh/orthoseg/orthoseg/branch/main/graph/badge.svg)](https://codecov.io/gh/orthoseg/orthoseg)
[![PyPI version](https://img.shields.io/pypi/v/orthoseg.svg)](https://pypi.org/project/orthoseg)
[![DOI](https://zenodo.org/badge/147507046.svg)](https://zenodo.org/doi/10.5281/zenodo.10340584)

A python package that makes it (relatively) easy to segment orthophotos. Any type of
georeferenced images should work, e.g. satellite, aerial or drone images, (historical)
maps, hillshades,...

No programming is needed, everything is managed via configuration files.

The typical steps:
1. create a training dataset for a topic of your choice, e.g. in QGIS
2. train a neural network to segment orthophotos
3. run the segmentation on a larger area + vectorize the result
4. apply some basic postprocessing like dissolve, simplify,...

Only open source software is needed, eg. QGIS and tensorflow.

Installation and usage instructions can be found in the [orthoseg docs](https://orthoseg.readthedocs.io)

As an example, this is an example of how the output of a tree detection on aerial images
could look:

![Result of a tree detection on aerial images](docs/_static/images/trees.jpg)
