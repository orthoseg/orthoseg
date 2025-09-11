.. orthoseg documentation main file, created by
   sphinx-quickstart on Thu Nov  5 20:17:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. currentmodule:: orthoseg

====================
orthoseg |version|
====================

As the name implies, the purpose of the project is to help you to apply
`image segmentation <https://en.wikipedia.org/wiki/Image_segmentation>`_ on
`orthophotos <https://en.wikipedia.org/wiki/Orthophoto>`_.

Image segmentation is quite a popular computer vision topic, eg. for use in self driving
cars. The same technique can also be used on orthophotos: images in the viewpoint as
available in eg. google maps.

The platform the photos were captured on (airplane, satellite, drone,...) obviously
doesn't matter, but at the moment only 3 (possibly 4 but this is not tested)
bands/channels are supported.

The image/raster layer you want to run a detection on can be a local file, a
`WMS server <https://en.wikipedia.org/wiki/Web_Map_Service>`_, a WMTS server or an XYZ
server. The output of the detection is a vector layer ready to do further analysis on.

You don't really need to have programming skills to use orthoseg, but you need to have
some knowledge of GIS technology (e.g. QGIS) and you need to be able to install and run
python scripts.

Orthoseg uses deep neural networks, so if you want to process larger areas of high
resolution (e.g. 0.25 m/pixel) imagery, access to a CUDA GPU is recommended, otherwise
you'll need to be patient.

An example of the output for a detection of tree crowns on a high resolution orthophoto
(0.25 m/pixel) is shown below.

.. image:: docs/_static/images/trees.jpg
   :alt: Result of a tree detection on aerial images
   

.. toctree::
   :maxdepth: 1

   Installation <installation>
   User guide <user_guide>
   FAQ <faq>
   Development <development>