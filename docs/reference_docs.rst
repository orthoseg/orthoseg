.. currentmodule:: orthoseg

=======================
Reference documentation
=======================

To run orthoseg, you can use the command line interface (CLI) that is provided
as documented in the :doc:`orthoseg_cli` page.

As an alternative to the CLI, you can also use the high level Python API to call the
main orthoseg functions. These are documented in the :doc:`orthoseg_highlevel_api` page.

It is also possible to call low-level orthoseg functions in e.g. orthoseg/lib
directly from Python. However, this is not considered as a public API so it may change
without notice and hence API documentation is also not offered here.

As already mentioned in the :doc:`user_guide`, orthoseg projects are configured via
``.ini`` files rather than having to write custom Python code for each project.

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

   Command line interface (CLI) <reference_docs/orthoseg_cli>
   High level Python API <reference_docs/orthoseg_highlevel_api>
   Image layer configuration <reference_docs/image_layers_config>
   Project configuration <reference_docs/project_config>
