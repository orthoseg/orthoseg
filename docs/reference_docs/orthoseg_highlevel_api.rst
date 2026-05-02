.. currentmodule:: orthoseg

========================
High level API Reference
========================

This page provides an overview of all public objects, functions and methods exposed in
the high level, public API of orthoseg.

Core entry points
=================

The primary way to use orthoseg from Python is to import and call the main functions
from the top-level :py:mod:`orthoseg` module:

.. autosummary::
   :toctree: api/
   :nosignatures:

   train
   predict
   postprocess
   load_images
   validate

Detailed API
============

.. automodule:: orthoseg
   :members: train, predict, postprocess, load_images, validate
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

Low-level API note
==================

It is also possible to call low-level orthoseg functions in the :py:mod:`orthoseg.lib`
subpackage directly from Python. However, these are not considered as a public API and
may change without notice. Hence, detailed API documentation is not provided for these
low-level modules.
