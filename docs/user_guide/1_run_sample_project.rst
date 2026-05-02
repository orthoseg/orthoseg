.. currentmodule:: orthoseg

==================
Run sample project
==================

Once the :doc:`/installation` of orthoseg and its dependencies is completed, you can
run the sample project included. The sample project is an easy way to get started and
should give a good idea on how you can start your own segmentation project.

It contains:

* a training dataset that can be used to train a network to detect football fields
* a sample of the basic configuration for a typical project that specifies using a
  lightweight neural network so it should run smoothly on a regular computer
* a sample of the default directory structure used by orthoseg
* a QGIS project file with the training data + aerial images that will be used to train
  the neural network + to detect football fields on

.. note::

   The training data included and pretrained model based on it is meant to show how
   the process works, not to give perfect, or even decent, results.


Running the sample project is easy. If the installation was successful, the following
steps should do the trick: 

1. start a conda command prompt
2. activate the orthoseg environment with::

    conda activate orthoseg

3. download the sample projects from orthoseg. You can specify the base location to
   download the sample projects to, but in this tutorial I'll assume "~" (= your home
   directory) for simplicity::

    orthoseg_load_sampleprojects ~

4. preload the images so they are ready to detect the football fields on, using the
   sample configurations file "footballfields.ini_BEFL-2019"::
   
    orthoseg_load_images --config ~/orthoseg/sample_projects/footballfields/footballfields_BEFL-2019.ini

5. for the footballfields sample project, a pretrained neural network was downloaded in
   step orthoseg_load_sampleprojects to avoid having to train it. But, normally you
   would now train the neural network with the following command::

    orthoseg_train --config ~/orthoseg/sample_projects/footballfields/footballfields.ini

6. detect the football fields::

    orthoseg_predict --config ~/orthoseg/sample_projects/footballfields/footballfields_BEFL-2019.ini

Now, the directory `~/orthoseg/sample_projects/footballfields/output_vector` will
contain a .gpkg file with the football fields found.

An interesting exercise might be to detect football fields on another layer (on another
location). To get reasonable results, this should be a layer with 0.25 meter pixel size,
as this was the pixel size the footballfields detection was trained on. It's best to
first read :doc:`2_prepare_new_project` for some background information and then you could
try the following steps:

1. Add the layer you want to predict on to the `imagelayer.ini` config file located in
   your projects directory (`~/orthoseg/sample_projects`).
   Not a very interesting example, but if you don't have a lot of inspiration you could
   use a layer with orthophotos of 2020 on the same location as the 2019 layer above.
   This can be configured as follows::

      [BEFL-2020]
      wms_server_url = https://geo.api.vlaanderen.be/omw/wms?
      wms_layernames = OMWRGB20VL
      wms_layerstyles = default
      wms_version = 1.3.0
      projection = epsg:31370
      bbox = 174900, 176400, 175300, 176600

   Detailed information on the different available options to configure image layers
   can be found in :doc:`/reference_docs/image_layers_config`.
2. Make a copy of `footballfields_BEFL-2019.ini` and change the `predict image_layer`
   parameter in the file to point to the new layer, e.g.::

      [predict]
      image_layer = BEFL-2020
   
   As an alternative to having to create a specific .ini file for each layer you want to
   predict on, you can also overrule a key (or keys) by passing overrules as extra
   parameters to e.g. `orthoseg_predict`. For this case, to overrule
   :confval:`predict.image_layer`, you can add parameter
   ``predict.image_layer=BEFL-2020``::

      orthoseg_predict --config ~/orthoseg/sample_projects/footballfields/footballfields.ini predict.image_layer=BEFL-2020

3. Run `orthoseg_load_images` to prepare the layer to predict on::

      orthoseg_load_images --config ~/orthoseg/sample_projects/footballfields/footballfields_BEFL-2020.ini

4. Run the detection again with `orthoseg_predict`::

      orthoseg_predict --config ~/orthoseg/sample_projects/footballfields/footballfields_BEFL-2020.ini
