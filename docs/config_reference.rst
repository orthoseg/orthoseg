Configuration Reference
=======================

As already mentioned in the :doc:`user_guide`, orthoseg can be configured via
``.ini`` files rather than by writing Python code.

There are two main types of configuration files:

- **Project configuration**: a file that defines all settings for a segmentation
  project such as which image layers to use, model architecture, training
  parameters, and post-processing options.
- **Image layers configuration**: a file that defines the image layers that can
  be used in project configurations. This file is shared across projects and can
  be reused in multiple project configurations.


.. include:: config_project.rst

.. include:: config_imagelayers.rst
