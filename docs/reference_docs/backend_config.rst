=====================
Backend configuration
=====================

OrthoSeg uses Keras for model operations and supports the TensorFlow, Torch, and JAX
backends. If multiple backends are installed, OrthoSeg selects TensorFlow by default.
Select another backend by setting ``KERAS_BACKEND`` before starting OrthoSeg, for
example: ::

    KERAS_BACKEND=torch orthoseg_predict --config path/to/project.ini

TensorFlow logging
==================

OrthoSeg sets ``TF_CPP_MIN_LOG_LEVEL=2`` before TensorFlow is initialized. This hides
TensorFlow's informational and warning messages from the console. Set the variable
explicitly before starting OrthoSeg to use a different level, for example: ::

    TF_CPP_MIN_LOG_LEVEL=0 orthoseg_predict --config path/to/project.ini

The TensorFlow levels are:

- ``0``: show all messages;
- ``1``: hide informational messages;
- ``2``: hide informational and warning messages;
- ``3``: hide informational, warning, and error messages.

To keep TensorFlow's native output in a file, redirect both standard output and
standard error on the command line, for example: ::

    orthoseg_predict --config path/to/project.ini > predict.log 2>&1
