.. orthoseg documentation master file, created by
   sphinx-quickstart on Thu Nov  5 20:17:22 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. currentmodule:: orthoseg

====================
orthoseg |version|
====================

As the name implies, the purpose of the project is to help you to apply
`image segmentation <https://en.wikipedia.org/wiki/Image_segmentation>`_ on
`orthophotos <https://en.wikipedia.org/wiki/Orthophoto>`_. Image segmentation is quite a
popular computer vision topic, eg. for use in self driving cars. The same technique can
also be used on orthophotos: images in the viewpoint as available in eg. google maps.

The platform the photos were captured on (airplane, satellite, drone,...) obviously
doesn't matter, but at the moment only 3 (possibly 4 but this is not tested)
bands/channels are supported.

The image/raster layer you want to run a detection on can be a local file, a
`WMS server <https://en.wikipedia.org/wiki/Web_Map_Service>`_, a WMTS server or an XYZ
server.

Orthoseg uses deep neural networks, so if you want to process larger areas of high
resolution (e.g. 0.25 m/pixel) imagery, access to a CUDA GPU is recommended, otherwise
you'll need to be patient.


.. toctree::
   :maxdepth: 1

   Installation <installation>
   User guide <user_guide>
   API reference <reference>
   FAQ <faq>
   Development <development>