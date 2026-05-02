.. currentmodule:: orthoseg

=====================
Prepare a new project
=====================

A few technical steps need to be taken to prepare a new segmentation project.

Only once: prepare "projects" directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If this is your very first orthoseg project, you need to prepare a directory where you
want to put your orthoseg projects. In the rest of the documentation we'll refer to this
directory as `{projects_dir}`.

It doesn't really matter where this directory is located, but these are examples that
can give you some inspiration:

* on linux: `~/orthoseg/projects` 
* on windows: `c:/users/{username}/orthoseg/projects`

The easiest way to create it is by starting from a copy of the `sample_projects`
you downloaded in the steps above to eg. your personal `orthoseg` directory and rename
it to `projects`.

This way your projects directory immediately contains:

* an `imagelayers.ini` file (with sample content)
* a `project_defaults_overrule.ini` file (with sample content)
* the `project_template` directory: the template for a new segmentation project

Add the layer(s) you want to segment on to the image layer configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The configuration for the image layers is located in `{projects_dir}/imagelayers.ini`.

Layers can be accessed from a WMS server, a WMTS server or via a file
(= a `GDAL raster dataset <https://gdal.org/en/stable/drivers/raster>`_).
The basic structure of this configuration file is as follows: every section in the
.ini file (eg. `[BEFL-2019]`) contains the configuration of one `image layer`.
In later steps of this tutorial you well need to use the "image layer names"
(for these examples `BEFL-2019` and `BEFL-2020`), they are referred to with
`{image_layer_name}`::

    # In this file, the image layers we can use in segmentations are configured. 

    [BEFL-2019]
    # Configuration info of this layer
    ...

    [BEFL-2020]
    # Configuration info of this layer
    ...

A "file layer" can be any local file in one of the many raster file types supported
by `GDAL <https://gdal.org/en/stable/drivers/raster>`_. Via the "file layer", you can 
also use the `GDAL WMS driver <https://gdal.org/en/stable/drivers/raster/wms.html>`_ by
creating an xml file with the necessary configuration. An example file to use a XYZ tile
server (eg. OpenStreetMap) can be found here:
:doc:`/file_viewers/imagelayer_osm_xml_viewer`.

A more elaborate example that can be used as a template for the configuration can be
found here: :doc:`/file_viewers/imagelayers_ini_viewer`.

Project name
^^^^^^^^^^^^

Choose a new name for the segmentation project: par example 'greenhouses', 'trees',
'buildings',... In the rest of the manual the project name will be refered to with
`{segment_subject}`.

Prepare "project" directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For your new segmentation project, make a copy of the `project_template` directory to
`{projects_dir}` and rename it to `{segment_subject}`.

Prepare project settings
^^^^^^^^^^^^^^^^^^^^^^^^

Rename the project file in the new project directory from `projectfile.ini` to
`{segment_subject}.ini`. In the file you should at least change the following
parameter:::

    [general]
    segment_subject = {segment_subject}


As you might have recognized from the small section above, orthoseg uses the good old
.ini file format for its configuration. A special detail is that it makes use of the
"ExtendedInterpolation" extension. Based on the sample files and the examples in this
manual you will probably be able to figure out how to use it, but if you want to dive
deeper, you can have a look here:
`ConfigParser-ExtendeInterpolation <https://docs.python.org/3.13/library/configparser.html#interpolation-of-values>`_.

All possible parameters that can be used in the project configuration file, including
their default values, are documented in :doc:`/reference_docs/project_config`.

To avoid having to copy/paste and repeat a lot of parameters in many project files,
you can define common project parameters in a common file and only put project-specific
parameters in your project file.

In the sample projects, a :doc:`/file_viewers/project_defaults_overrule_ini_viewer` file
is used to define common overrules of the default configuration values for all projects
in the project directory.

For a specific project, only some project-specific parameter values are overruled, like
you can see here: :doc:`/file_viewers/footballfields_ini_viewer`.

Finally, if you want e.g. a project file to run a detection on a specific image layer,
you can add yet another file that overrules the project file yet again, like you can
see here: :doc:`/file_viewers/footballfields_BEFL-2019_ini_viewer`. Note the
:confval:`general.extra_config_files_to_load` property in the project file that allows
you to specify all extra config files that will be loaded in the order listed.

Configure image layer(s)
^^^^^^^^^^^^^^^^^^^^^^^^

If the image layers you want to train/predict on aren't configured yet, configure them
in `{projects_dir}/imagelayers.ini`, the same way as the default layers provided.

Prepare training data files
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The labels folder in the new project folder contains two .gpkg files where training
examples can be added to as explained in the following page of this manual. The file
names should have the following structure: 

* {segment_subject}_{image_layer_name}_locations.gpkg 
* {segment_subject}_{image_layer_name}_polygons.gpkg 

If you want to create one segmentation with training examples based on multiple image
layers, you can create a seperate pair of .gpkg files per image layer.

Prepare GIS project
^^^^^^^^^^^^^^^^^^^

Start your prefered GIS application (eg. QGIS), and create a new project. Add the .gpkg
files from the labels directory and add the layers you want to digitize the training
examples on (eg. BEFL-2019). 

It is practical to create a layer group per image layer you want to train data on with:

   * the 2 corresponding .gpkg files
   * the image layer (eg. BEFL-2019)

Save the project to the project directory.
