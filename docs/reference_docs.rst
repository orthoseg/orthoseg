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

- **Image layers configuration**: an .ini file that defines the image layers that can
  be used in orthoseg project configurations. This file is shared across projects and
  can be reused in multiple project configurations.
- **Project configuration**: an .ini file that defines all settings for a segmentation
  project such as which image layers to use, model architecture, training
  parameters, and post-processing options.


Contents
~~~~~~~~

.. toctree::
   :maxdepth: 1

   Orthoseg CLI <reference_docs/orthoseg_cli>
   Orthoseg high level API <reference_docs/orthoseg_highlevel_api>
   Image layer configuration <reference_docs/image_layers_config>
   Project configuration <reference_docs/project_config>
