.. currentmodule:: orthoseg

==========
User guide
==========

The main objective of orthoseg is to provide a simple to use but powerful API to do
fast spatial operations on large vector GIS files.


Run sample project  
------------------

Once the installation is completed, you can run the sample project included. The project
is an easy way to get started and should give a good idea on how you can start your own
segmentation project. 

It contains:

* a training dataset that can be used to train a network to detect football fields
* a sample of the basic configuration for a typical project
* a sample of the default directory structure used by orthoseg
* a QGIS project file with the training data + aerial images that will be used to train
  the neural network + to detect football fields on

Remark: the training data included is meant to show how the process works, not to give
perfect results.

Running the sample project is easy. If the orthoseg installation was successful, the
following steps should do the trick: 

1. start a conda command prompt
2. activate the orthoseg environment with::

    conda activate orthoseg

3. download the sample projects from orthoseg. You can specify the base location to
   download the sample projects to, but in this tutorial I'll assume "~" (= your home
   directory) for simplicity.::

    orthoseg_load_sampleprojects ~

4. preload the images so they are ready to detect the football fields on, using the
   sample configurations file "footballfields.ini_BEFL-2019".::
   
    orthoseg_load_images --config ~/orthoseg/sample_projects/footballfields/footballfields_BEFL-2019.ini

5. for the footballfields sample project, a pretrained neural network was downloaded in
   step orthoseg_load_sampleprojects to avoid having to train it. But, normally you
   would now train the neural network with the following command.::

   orthoseg_train --config ~/orthoseg/sample_projects/footballfields/footballfields_BEFL-2019.ini

6. detect the football fields.::

   orthoseg_predict --config ~/orthoseg/sample_projects/footballfields/footballfields_BEFL-2019.ini

Now, the directory ~/orthoseg/sample_projects/footballfields/output_vector will contain
a .gpkg file with the football fields found.

An interesting exercise might be to detect football fields on another layer (on another
location). To get reasonable results, this should be a layer with 0.25 meter pixel size,
as this was the pixel size the footballfields detection was trained on. It's best to
first read `Prepare-new-project <https://github.com/orthoseg/orthoseg/wiki/Prepare-new-project>`_
for some background information and then you could try the following steps:

1. add the layer you want to predict on to the imagelayer.ini config file 
2. make a copy of footballfields_BEFL-2019.ini and change the "predict image_layer"
   parameter in the file to point to the new layer::

    [predict]
    image_layer = BEFL-2019

3. run orthoseg_load_images to prepare the layer to predict on::

   orthoseg_load_images --config ~/orthoseg/sample_projects/footballfields/footballfields_BEFL-2019.ini

4. run the detection again.::

   orthoseg_predict --config ~/orthoseg/sample_projects/footballfields/footballfields_BEFL-2019.ini


Prepare new project
-------------------

A few technical steps need to be taken to prepare a new segmentation project.

1. Only once: prepare "projects" directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If this is your very first orthoseg project, you need to prepare a directory where you
want to put your orthoseg projects. In the rest of the documentation we'll refer to this
directory as {projects_dir}.

It doesn't really matter where this directory is located, but these are examples that
can give you some inspiration:

* on linux: ~/orthoseg/projects 
* on windows: c:/users/{username}/orthoseg/projects

The easiest way to create it is by starting from a copy of the
`sample_projects <https://github.com/orthoseg/orthoseg/tree/master/sample_projects>`_
directory to eg. your personal "orthoseg" directory and rename it to "projects".

This way your projects directory immediately contains:

* an imagelayers.ini file (with sample content)
* a project_defaults_overrule.ini file (with sample content)
* the "project_template" directory: the template for a new segmentation project

2. Add the layer(s) you want to segment on to the image layer configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The configuration for the image layers is located in {projects_dir}/imagelayers.ini.

Layers can be a local file, a WMS server, WMTS server or an XYZ server. The basic
structure of this configuration file is as follows: every section in the .ini file
(eg. [BEFL-2019]) contains the configuration of one "image layer". In later steps of
this tutorial you well need to use the "image layer names" (for these examples BEFL-2019
and BEFL-2020), they are referred to with {image_layer_name}.::

    # In this file, the image layers we can use in segmentations are configured. 

    [BEFL-2019]
    # Configuration info of this layer
    ...

    [BEFL-2020]
    # Configuration info of this layer
    ...


A more elaborate example that can be used as a template for the configuration can be
found here: `imagelayers.ini <https://github.com/orthoseg/orthoseg/blob/master/sample_projects/imagelayers.ini>`_.

3. Project name
^^^^^^^^^^^^^^^

Choose a new name for the segmentation project: par example 'greenhouses', 'trees',
'buildings',... In the rest of the manual the project name will be refered to with
{segment_subject}.

4. Prepare "project" directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For your new segmentation project, make a copy of the "project_template" directory to
{projects_dir} and rename it to {segment_subject}.

5. Prepare project settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rename the project file in the new project directory from "projectfile.ini" to
"{segment_subject}.ini". In the file you should at least change the following
parameter:::

    [general]
    segment_subject = {segment_subject}


As you might have recognized from the small section above, orthoseg uses the good old
.ini file format for its configuration, with "ExtendedInterpolation". General
information about this file format can be found here:
`ConfigParser-ExtendeInterpolation <https://docs.python.org/3.3/library/configparser.html#interpolation-of-values>`_.

The configuration can be found + modified in the following files: 

1. All existing sections and parameters + their **default values** can be found in the
   following file: `orthoseg_install_dir/orthoseg/project_defaults.ini <https://github.com/orthoseg/orthoseg/blob/master/orthoseg/project_defaults.ini>`_.
   It is highly recommended not to change anything in this file, as it will be
   overwritten when installing a new version of orthoseg anyway. But if you want to
   overrule any setting, this is the perfect spot to find all existing sections +
   parameters and copy/paste the section + parameter to one of the next files and change
   the value there to overrule it for your project(s).
1. If you want to **overrule** a parameter for all the projects in your project
   directory, add the section + parameter to the project_defaults_overrule.ini file in
   your projects directory, eg:
   `{projects_dir}/project_defaults_overrule.ini <https://github.com/orthoseg/orthoseg/blob/master/sample_projects/project_defaults_overrule.ini>`_.
1. If you want to **overrule** parameters for a specific project, you can do so in the
   project-specific config file: eg.
   `{projects_dir}/{segment_subject}/{segment_subject}.ini <https://github.com/orthoseg/orthoseg/blob/master/sample_projects/project_template/projectfile.ini>`_.

6. Configure image layer(s)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the image layers you want to train/predict on aren't configured yet, configure them
in `{projects_dir}/imagelayers.ini <https://github.com/orthoseg/orthoseg/blob/master/sample_projects/imagelayers.ini>`_,
the same way as the default layers provided.

7. Prepare training data files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The labels folder in the new project folder contains two .gpkg files where training
examples can be added to as explained in the following page of this manual. The file
names should have the following structure: 

* {segment_subject}_{image_layer_name}_locations.gpkg 
* {segment_subject}_{image_layer_name}_polygons.gpkg 

If you want to create one segmentation with training examples based on multiple image
layers, you can create a seperate pair of .gpkg files per image layer.

8. Prepare GIS project
^^^^^^^^^^^^^^^^^^^^^^

Start your prefered GIS application (eg. QGIS), and create a new project. Add the .gpkg
files from the labels directory and add the layers you want to digitize the training
examples on (eg. BEFL-2019). 

It is practical to create a layer group per image layer you want to train data on with:

   * the 2 corresponding .gpkg files
   * the image layer (eg. BEFL-2019)

Save the project to the project directory.


Run train, predict,... session
------------------------------

Once you have added a meaningful number of (extra) examples, you can do a (new) training
run. The number of added examples that is meaningful will depend on the case, but a
reasonable amount is 50.
Once the model is trained, you can continue by running a prediction and if wanted a
postprocessing step.

If you ran the sample project, these steps will look very familiar:

1. start a conda command prompt

2. activate the orthoseg environment with
   
   conda activate orthoseg


1. preload the images so they are ready to detect your {segment_subject} on, using the
   configuration file "{project_dir}{segment_subject}.ini".
   
   orthoseg_load_images --config {project_dir}{segment_subject}.ini


1. train a neural network to detect football fields.
   
   orthoseg_train --config {project_dir}{segment_subject}.ini


1. detect the football fields.

   orthoseg_predict --config {project_dir}{segment_subject}.ini


After this completes, the directory {project_dir}/output_vector will contain a .gpkg
file with the features found.

Of course it is also possible to script this in your scripting language of choice to
automate this further...

Remark:
^^^^^^^
Because tasks often take quite a while, orthoseg maximally tries to resume work that
was started but was not finished yet. Eg. when predicting a large area, OrthoSeg will
save the prediction per image, so if the prediction process is stopped for any reason
and restarted, it will continue where it stopped.
