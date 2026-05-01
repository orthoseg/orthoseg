.. currentmodule:: orthoseg

=======================
Configuration Reference
=======================

As already mentioned in the :doc:`user_guide`, orthoseg can be configured via
``.ini`` files rather than by writing Python code.

Even though it is possible to call low-level orthoseg functions in e.g. orthoseg/lib
directly from Python, this is not considered as a public API so it may change without
notice and hence API documentation is also not offered here.

There are two main types of configuration files:

- **Project configuration**: an .ini file that defines all settings for a segmentation
  project such as which image layers to use, model architecture, training
  parameters, and post-processing options.
- **Image layers configuration**: an .ini file that defines the image layers that can
  be used in project configurations. This file is shared across projects and can
  be reused in multiple project configurations.


.. toctree::
   :maxdepth: 1

   Project configuration <reference_docs/config_project>
   Image layer configuration <reference_docs/config_imagelayers>
