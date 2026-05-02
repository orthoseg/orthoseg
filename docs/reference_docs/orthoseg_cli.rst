======================
Orthoseg CLI Reference
======================

Orthoseg exposes a small set of command line tools for the main project workflow.
All commands are installed as console scripts when the package is installed.

Common usage pattern
====================

Most commands use the same invocation pattern:

.. code-block:: bash

	orthoseg_<command> --config path/to/project.ini [section.key=value ...]

The ``--config`` argument points to the project configuration file to use.
Most commands also accept optional config overrules as positional arguments in the
form ``<section>.<key>=<value>``. This is useful for small one-off changes,
such as predicting on another image layer without creating a separate ``.ini`` file.

Typical workflow
================

The most common sequence is:

1. ``orthoseg_load_sampleprojects`` to download the example projects (once).
2. ``orthoseg_load_images`` to fetch or prepare imagery into the local cache
   (once per prediction layer).
3. ``orthoseg_validate`` to prepare and validate training data
   (optional, validate also runs automatically at the start of ``orthoseg_train``).
4. ``orthoseg_train`` to train a model.
5. ``orthoseg_predict`` to run inference on an image layer.
6. ``orthoseg_postprocess`` to further postprocess the prediction output.

Command reference
=================

orthoseg_load_sampleprojects
----------------------------

Downloads the sample projects repository into a local ``orthoseg/sample_projects``
directory.

Usage:

.. code-block:: bash

	orthoseg_load_sampleprojects DEST_DIR [--ssl_verify true|false|PATH]

Important arguments:

- ``DEST_DIR``: base directory where the ``orthoseg/sample_projects`` directory will
  be created.
- ``--ssl_verify``: controls certificate validation for the download. It accepts
  ``true``, ``false``, or a path to a certificate bundle.

Use this command to get a working example project structure, sample data, and a
pretrained model for the football fields example.

orthoseg_load_images
--------------------

Loads images for a layer into cache directories for ``orthoseg_predict``.

The images will follow the tiling scheme configured in the image layer and project
configuration. Normally the images will have a certain overlap to avoid edge effects in
the predictions.

Usage:

.. code-block:: bash

	orthoseg_load_images --config path/to/project.ini [section.key=value ...]

Important arguments:

- ``--config``: the project configuration file.
- ``section.key=value``: optional configuration overrules.

orthoseg_validate
-----------------

Validates the training data defined by the project configuration.

This is useful if you have a training or prediction running already and you don't want
to interrupt it to validate another training dataset.

Usage:

.. code-block:: bash

	orthoseg_validate --config path/to/project.ini [section.key=value ...]

Important arguments:

- ``--config``: the project configuration file.
- ``section.key=value``: optional configuration overrules.

orthoseg_train
--------------

Runs a training session for the configured project.

This command prepares output directories, reads the configured training labels, and
trains the configured model.

Usage:

.. code-block:: bash

	orthoseg_train --config path/to/project.ini [section.key=value ...]

Important arguments:

- ``--config``: the project configuration file.
- ``section.key=value``: optional configuration overrules.

orthoseg_predict
----------------

Runs inference for the configured project on the configured ``predict.image_layer``.

It uses the newest trained model and writes prediction outputs as vector features with
some basic postprocessing being applied on the fly already.

Usage:

.. code-block:: bash

	orthoseg_predict --config path/to/project.ini [section.key=value ...]

Important arguments:

- ``--config``: the project configuration file.
- ``section.key=value``: optional configuration overrules.

Example:

.. code-block:: bash

	orthoseg_predict --config footballfields.ini predict.image_layer=BEFL-2020

orthoseg_postprocess
--------------------

Post-processes the prediction output.

Especially postprocessing steps that cannot be applied on the fly on individual
prediction tiles are applied in this step. For example, if you want to merge adjacent
polygons that are predicted in different tiles with a ``dissolve`` operation, this
can be applied here.

Usage:

.. code-block:: bash

	orthoseg_postprocess --config path/to/project.ini [section.key=value ...]

Important arguments:

- ``--config``: the project configuration file.
- ``section.key=value``: optional configuration overrules.

``orthoseg_postprocess`` is normally run after ``orthoseg_predict``.

osscriptrunner
--------------

Runs job scripts from a directory and can optionally keep watching that directory for
new work. This is the automation-oriented entry point for scheduled or unattended
processing.

Usage:

.. code-block:: bash

	osscriptrunner --script_dir path/to/jobs [--watch] [--config path/to/scriptrunner.ini]

Important arguments:

- ``--script_dir``: directory containing scripts or job files to execute.
- ``--watch``: keep monitoring the directory for new jobs.
- ``--config``: optional scriptrunner configuration file that overrides defaults.

The scriptrunner is separate from the project ``.ini`` files used by the other CLI
commands. It is intended for orchestration rather than for a single train or predict
run.

