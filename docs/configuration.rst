Project Defaults
================

The `project_defaults.ini` file contains default settings for projects.

The config used for an orthoseg project is loaded in the following order:

1) the project defaults as "hardcoded" in orthoseg (project_defaults.ini)
2) any .ini files specified in the general.extra_config_files_to_load
   parameter in your project config file, in the order specified.
3) the parameters in your project config file.

Parameters specified in a config file loaded later in the order above
overrule the corresponding parameter values specified in a previously
loaded config file.


[general]
---------

General settings.

.. confval:: general.extra_config_files_to_load
   :type: ``str``
   :default: ``""``

   Extra config files to load for the project.

   They will be loaded in the order specified and can be specified one
   path per line, comma seperated.

   If a relative path is used it will be resolved towards the parent dir of
   the project config file.

.. confval:: general.segment_subject
   :type: ``str``
   :default: ``MUST_OVERRIDE``

   The subject that will be segmented.

   This value must be overruled in the project specific config file.

.. confval:: general.ssl_verify
   :type: ``bool``
   :default: ``True``

   Specify the way certificates of ssl requests are verified.

   Options:

   * True: use the default certificate bundle as installed on your system
   * False: disable certificate validation (NOT recommended!)
   * path to a certificate bundle file (.pem) to specify the certificate
     bundle to be used. In corporate networks using a proxy server this is
     often needed when requests give CERTIFICATE_VERIFY_FAILED errors.

.. confval:: general.nb_parallel
   :type: ``int``
   :default: ``-1``

   Specify the number of cpu's to use while doing heavy processing.

   If -1, it uses all available processors.


[download]
----------

Settings regarding the download action.

.. confval:: download.cron_schedule
   :type: ``str``
   :default: ``None``

   Schedule to control when images can be downloaded.

   If not specified there is no time limitation.


[model]
-------

Settings concerning the model you want to use for the segmentation.

.. confval:: model.architecture_id
   :type: ``int``
   :default: ``0``

   The id of the architecture used.

   Only needs to be changed if you want to compare the results of different
   architectures on the same training data, bacause orthoSeg will only train
   one model per traindata_id, architecture_id and hyperparams_id.

   If the architecture_id is 0, it won't be included in the file name
   of trained models.

.. confval:: model.architecture
   :type: ``str``
   :default: ``inceptionresnetv2+unet``

   The segmentation architecture to use.

   The architectures currently supported by orthoseg follow the encoder-decoder
   principle:

   * encoder: a (deep) neural network that detects features on object level
   * decoder: a (deep) neural network that converts the detected features
     on object level to a segmentation on pixel level

   To configure an encoder/decoder architecture, specify it in
   the following way: architecture = {encoder}+{decoder}

   The supported encoder and decoders can be found in the orthoseg
   documentation.

.. confval:: model.nb_channels
   :type: ``int``
   :default: ``3``

   The number of channels of the images to train on.


[train]
-------

Settings concerning the train process.

.. confval:: train.preload_with_previous_traindata
   :type: ``bool``
   :default: ``False``

   Preload model -> only overrule in local_overrule.ini.

.. confval:: train.force_model_traindata_id
   :type: ``int``
   :default: ``-1``

   Force to use a model trained on this traindata version (-1 to disable).

.. confval:: train.resume_train
   :type: ``bool``
   :default: ``False``

   When training, resume training on the current best existing model.

.. confval:: train.force_train
   :type: ``bool``
   :default: ``False``

   Train a model, even if a model exists already.

.. confval:: train.trainparams_id
   :type: ``int``
   :default: ``0``

   The id of the set of hyper parameters to use while training.

   Needs to be changed if you want to compare the results of different
   hyperparameter sets on the same training data because orthoSeg will only
   train one model per traindata_id, architecture_id and trainparams_id.

   If the hyperparams_id is 0, it won't be included in the file name of
   trained models.

.. confval:: train.image_pixel_width
   :type: ``int``
   :default: ``512``

   Parameters regarding the size/resolution of the images used to train on.

   The size the label_location boxes need to be digitized depends on these values.
   E.g. with `image_pixel_width` = 512 and `image_pixel_x_size` = 0.25, the boxes
   need to be 512 pixels * 0.25 meter/pixel = 128 meter wide.

   For some model architectures there are limitations on the image sizes supported.
   E.g. if you use the linknet decoder, the images pixel width and height
   has to be divisible by factor 32.

.. confval:: train.image_pixel_height
   :type: ``int``
   :default: ``512``


.. confval:: train.image_pixel_x_size
   :type: ``float``
   :default: ``0.25``


.. confval:: train.image_pixel_y_size
   :type: ``float``
   :default: ``0.25``


.. confval:: train.labelpolygons_pattern
   :type: ``str``
   :default: ``${dirs:labels_dir}/${general:segment_subject}_{image_layer}_polygons.gpkg``

   Pattern how the file paths of the label files should be formatted.
   The image_layer to get the images from is extracted from the file path.

.. confval:: train.labellocations_pattern
   :type: ``str``
   :default: ``${dirs:labels_dir}/${general:segment_subject}_{image_layer}_locations.gpkg``


.. confval:: train.labelname_column
   :type: ``str``
   :default: ``classname``

   Column where the labels for each training polygon is available.

   If the column name configured here is not available columns "label_name"
   is tried as well.

.. confval:: train.label_datasources
   :type: ``str``
   :default: ``None``

   The datasources to use for the train labels.

   It is configured as a dictionary. It can be used to add label datasources
   in addition to the ones found via the patterns above.

   It is also possible to overrule/add properties to label datasources found
   via the patterns. In this case, (only) the locations_path is mandatory.
   It will be used as link key. All other properties specified will add/overrule:

   - properties extracted from the file path, e.g. image_layer
   - more general properties: train.image_pixel_x_size, train.image_pixel_y_size.

   A typical use case is to overrule the general project `pixel_size`s for one
   or more datasources to be able to train the model on different resolutions.
   It is possible to list the same datasource multiple times to use the same
   train data to e.g. train on different resolutions.

   Examples:
   
   .. code :: python

      label_datasources = {
         "label_ds0_resolution1": {
            "locations_path": "${dirs:labels_dir}/${general:segment_subject}_BEFL-2019_locations.gpkg",
            "polygons_path": "${dirs:labels_dir}/${general:segment_subject}_BEFL-2019_polygons.gpkg",
            "image_layer": "BEFL-2020",
            "pixel_x_size": 0.5,
            "pixel_y_size": 0.5,
         },
         "label_ds0_resolution2": {
            "locations_path": "${dirs:labels_dir}/${general:segment_subject}_BEFL-2019_locations.gpkg",
            "polygons_path": "${dirs:labels_dir}/${general:segment_subject}_BEFL-2019_polygons.gpkg",
            "image_layer": "BEFL-2020",
            "pixel_x_size": 0.25,
            "pixel_y_size": 0.25,
         }
      }

.. confval:: train.image_augmentations
   :type: ``dict``
   :default:   
      ``{`` |br|
      ``   "fill_mode": "constant",``|br|
      ``   "cval": 0,`` |br|
      ``   "rotation_range": 359.0,`` |br|
      ``   "width_shift_range": 0.05,`` |br|
      ``   "height_shift_range": 0.05,`` |br|
      ``   "zoom_range": 0.1,`` |br|
      ``   "brightness_range": [0.95, 1.05],`` |br|
      ``   "horizontal_flip": True,`` |br|
      ``   "vertical_flip": True,`` |br|
      ``}``

   The augmentations to apply to the input images during training.

   Remarks:

   * the default value for cval is 0, which means that areas outside the image
     after rotation/translation/zoom will be filled with value 0 (black). If
     your subject can have area's that are black and need to be detected, you
     might want to set cval to e.g. 255 (white).
   * for orthoseg >= 0.8, rescaling should be handled by the default
     `preprocess_input` function of the model (architecture) if needed, so using
     'rescale' augmentation is not allowed anymore.

.. confval:: train.mask_augmentations
   :type: ``dict``
   :default: ``{}`` (see description)

   The augmentations to apply to the label masks during training.

   Remarks:

   * the number of randomized values must be the same as for the image,
     otherwise the random augentation factors aren't the same as the image!
   * augmentations to translate, rotate,... should be the same as for the image!
   * the mask generally shouldn't be rescaled!
   * cval values should always be 0 for the mask, even if it is different for
     the image, as the cval of the mask refers to these locations being
     of class "background".

.. confval:: train.classes
   :type: ``dict``
   :default: ``{}`` (see description)

   The classes to be trained to.

   For each class:

   * a list of label names in the training data to use for this class
   * the weight to use when training

.. confval:: train.batch_size_fit
   :type: ``int``
   :default: ``6``

   The batch size to use during fit of model.

   A proper values depends on available hardware, model used and image size.

.. confval:: train.batch_size_predict
   :type: ``int``
   :default: ``20``

   The batch size to use while predicting in the train process.

   A proper values depends on available hardware, model used and image size.

.. confval:: train.optimizer
   :type: ``str``
   :default: ``adam``

   Optimizer to use for training.

.. confval:: train.optimizer_params
   :type: ``dict``
   :default: ``{}`` (see description)

   Parameters to use for the optimizer.

.. confval:: train.loss_function
   :type: ``str``
   :default: ``None``

   Loss function to use.

   If not specified, the defaults are:

   - For keras 3+: categorical_focal_crossentropy
   - For keras <3:
       - If weights specified in the classes: weighted_categorical_crossentropy
       - If no weights are specified: categorical_crossentropy

.. confval:: train.monitor_metric
   :type: ``str``
   :default: ``({one_hot_mean_iou}+{val_one_hot_mean_iou})/2``

   The metric(s) to monitor to evaluate which network is best during training.

   This can be a single metric or a formula with placeholders
   to calculate a value based on multiple metrics. Available metrics:

   - one_hot_mean_iou: intersection over union on the training dataset
   - val_one_hot_mean_iou: intersection over union on the validation dataset
   - categorical_accuracy: accuracy on the training dataset
   - val_categorical_accuracy: accuracy on the validation dataset

.. confval:: train.monitor_metric_mode
   :type: ``str``
   :default: ``max``

   The mode of the monitor metric.

   Use max to keep the models with a high monitor_metric. Otherwise use min.

.. confval:: train.save_format
   :type: ``str``
   :default: ``None``

   Format to save the trained model in.

   Options are keras, h5 or tf. The keras format is only supported for keras >= 3.
   Defaults to keras for keras >= 3 and to h5 for older versions.

.. confval:: train.save_best_only
   :type: ``bool``
   :default: ``True``

   True to only keep the best model during training.

.. confval:: train.save_min_accuracy
   :type: ``float``
   :default: ``0.80``

   The minimum accuracy to save the model.

   Set to 0 to always save the model.

.. confval:: train.nb_epoch_with_freeze
   :type: ``int``
   :default: ``20``

   Number of epochs to train with the pretrained layers frozen.

   Keeps pretrained layers intact, which is useful for the first few (2-10)
   epochs when big adjustments are made to the network. Training is also 20%
   faster during these epochs.

.. confval:: train.max_epoch
   :type: ``int``
   :default: ``1000``

   Maximum number of epochs to train.

   These epochs are in addition to nb_epoch_with_freeze.

.. confval:: train.earlystop_patience
   :type: ``int``
   :default: ``100``

   Stop training after this number of epochs without improvement.

   Not having improvement means the `earlystop_monitor_metric` hasn't improved.

.. confval:: train.earlystop_monitor_metric
   :type: ``str``
   :default: ``one_hot_mean_iou``

   Metric to use to determine that the model doesn't improve.

   Options are: categorical_accuracy, one_hot_mean_iou

.. confval:: train.earlystop_monitor_metric_mode
   :type: ``str``
   :default: ``max``

   The mode of the earlystop metric.

   Use max if the monitor_metric should be high. Otherwise use min.

.. confval:: train.log_tensorboard
   :type: ``bool``
   :default: ``False``

   True to activate tensorboard style logging.

.. confval:: train.log_csv
   :type: ``bool``
   :default: ``True``

   True to activate csv logging.

.. confval:: train.save_augmented_subdir
   :type: ``str``
   :default: ``None``

   Subdir name to save augmented images to while training.

   This should only be used for debugging purposes as it can
   take a lot of disk space and slow down training.


[predict]
---------

Settings concerning the prediction process.

.. confval:: predict.batch_size
   :type: ``int``
   :default: ``4``

   The batch size to use while predicting.

   A proper values depends on available hardware, model used and image size.

.. confval:: predict.image_layer
   :type: ``str``
   :default: ``MUST_OVERRIDE``

   The image layer to use for the prediction.

   The image layer needs to be an existing layer in imagelayers.ini.

.. confval:: predict.image_pixel_width
   :type: ``int``
   :default: ``2048``

   Parameters regarding the size/resolution of the images to run predict on.

   For some model architectures there are limitations on the image sizes
   supported. E.g. if you use the linknet decoder, the images pixel width
   and height has to be divisible by factor 32.

.. confval:: predict.image_pixel_height
   :type: ``int``
   :default: ``2048``


.. confval:: predict.image_pixel_x_size
   :type: ``float``
   :default: ``0.25``


.. confval:: predict.image_pixel_y_size
   :type: ``float``
   :default: ``0.25``


.. confval:: predict.image_pixels_overlap
   :type: ``int``
   :default: ``128``


.. confval:: predict.min_probability
   :type: ``float``
   :default: ``0.5``

   The minimum probability for a pixel to be attributed to a class.

   If the probability for all classes is below this threshold, the pixel will
   be attributed to the background.

   Possible values are from 0.0 till 1.

.. confval:: predict.max_prediction_errors
   :type: ``int``
   :default: ``100``

   Maximum errors that can occur during prediction before stopping the process.

.. confval:: predict.filter_background_modal_size
   :type: ``int``
   :default: ``0``

   Apply a filter to the background pixels and replace background by the most
   occuring value in a rectangle around the background pixel of the size
   specified.

.. confval:: predict.reclassify_to_neighbour_query
   :type: ``str``
   :default: ``None``

   Query to specify polygons to be reclassified.

   All detected polygons that comply to the query provided will be reclassified
   to the class of the neighbour with the longest border with it.

   The query need to be in the form to be used by pandas.DataFrame.query() and
   the following columns are available to query on:

   - area: the area of the polygon, as calculated by GeoSeries.area .
   - perimeter: the perimeter of the polygon, as calculated by GeoSeries.length .
   - onborder: 1 if the polygon touches the border of the tile being predicted, 0 if
     it doesn't. It is often useful to filter with onborder == 0 to avoid eg. polygons
     being reclassified because they are small due to being on the border.

   Example:

       `reclassify_to_neighbour_query = (onborder == 0 and area <= 5)`

.. confval:: predict.simplify_algorithm
   :type: ``str``
   :default: ``LANG+``

   Algorithm to use for vector simplification.

   The simplification configured here will be executed on the fly during the
   prediction. Options are:

   * RAMER_DOUGLAS_PEUCKER:
       * simplify_tolerance: extra, mandatory parameter: specifies the distance
         tolerance to be used.
   * VISVALINGAM_WHYATT:
       * simplify_tolerance: extra, mandatory parameter: specifies the area
         tolerance to be used.
   * LANG: gives few deformations and removes many points
       * simplify_tolerance: extra, mandatory parameter: specifies the distance
         tolerance to be used.
       * simplify_lookahead: extra, mandatory parameter: specifies the number
         of points the algorithm looks ahead during simplify.
   * LANG+: gives few deformations while removeing the most points.
       * simplify_tolerance: extra, mandatory parameter: specifies the distance
         tolerance to be used.
       * simplify_lookahead: extra, mandatory parameter: specifies the number
         of points the algorithm looks ahead during simplify.
   * If simplify_algorithm is not specified, no simplification is applied.

.. confval:: predict.simplify_tolerance
   :type: ``str``
   :default: ``${image_pixel_x_size}*1.5``

   Tolerance to use for the on-the-fly simplification during the prediction.

   Remark: you can use simple math expressions, eg. 1.5*5

.. confval:: predict.simplify_lookahead
   :type: ``int``
   :default: ``8``

   The number of points to look ahead during simplify.

   Only applicable for the LANG algorithm.

.. confval:: predict.simplify_topological
   :type: ``str``
   :default: ``None``

   Use topological simplification.

   If True, the resulting polygons of the classification are converted to
   topologies so no gaps are introduced in polygons that are next to each other.
   If not specified (= None), a multi-class classification will be simplified
   topologically, a single class will be simplified the standard way.


[postprocess]
-------------

Settings concerning the postprocessing after the prediction.

.. confval:: postprocess.keep_original_file
   :type: ``bool``
   :default: ``True``

   Keep the original output file of the prediction after postprocessing.

.. confval:: postprocess.keep_intermediary_files
   :type: ``bool``
   :default: ``True``

   Keep the intermediary files of the postprocessing.

.. confval:: postprocess.dissolve
   :type: ``bool``
   :default: ``True``

   Dissolve the result.

.. confval:: postprocess.dissolve_tiles_path
   :type: ``str``
   :default: ``None``

   Tile the result of the dissolve using the grid in the file specified.

.. confval:: postprocess.reclassify_to_neighbour_query
   :type: ``str``
   :default: ``None``

   Query to specify polygons to be reclassified.

   All detected polygons that comply to the query provided will be reclassified
   to the class of the neighbour with the longest border with it.

   The query need to be in the form to be used by pandas.DataFrame.query() and
   the following columns are available to query on:

   - area: the area of the polygon, as calculated by GeoSeries.area .
   - perimeter: the perimeter of the polygon, as calculated by GeoSeries.length .
   - onborder: 1 if the polygon touches the border of the tile being predicted, 0 if
     it doesn't. It is often useful to filter with onborder == 0 to avoid eg. polygons
     being reclassified because they are small due to being on the border.

   Example:

       `reclassify_to_neighbour_query = (onborder == 0 and area <= 5)`

.. confval:: postprocess.simplify_algorithm
   :type: ``str``
   :default: ``None``

   Apply simplify (also) after dissolve.

   For more information, check out the documentation at parameter
   predict:simplify_algorithm.

.. confval:: postprocess.simplify_tolerance
   :type: ``str``
   :default: ``${predict:image_pixel_x_size}*2``

   Tolerance to use for the postprocess simplification.

   Remark: you can use simple math expressions, eg. 1.5*5

.. confval:: postprocess.simplify_lookahead
   :type: ``int``
   :default: ``8``

   The number of points to look ahead during simplify.

   Only applicable for the LANG algorithm.


[dirs]
------

Settings concerning the directories where input/output data is found/put.

Remarks:

* UNC paths are not supported on Windows, always use mapped drive letters!
* always use forward slashes, even on Windows systems
* in all paths, it is possible to use the {tempdir} placeholder, which will
  be replaced by the default system temp dir.

.. confval:: dirs.projects_dir
   :type: ``str``
   :default: ``..``

   The base projects dir.

   Here, multiple orthoseg projects can be stored. Can either be:
   * an absolute path
   * OR a relative path starting from the location of the specific projectconfig
     file of the project

   Eg.: ".." means: projects_dir is the parent dir of the dir containing the
   project config file

.. confval:: dirs.project_dir
   :type: ``str``
   :default: ``${projects_dir}/${general:segment_subject}``

   The project directory for this subject.

.. confval:: dirs.log_dir
   :type: ``str``
   :default: ``${project_dir}/log``

   Log dir.

.. confval:: dirs.labels_dir
   :type: ``str``
   :default: ``${project_dir}/labels``

   Dir containing the label data.

.. confval:: dirs.training_dir
   :type: ``str``
   :default: ``${project_dir}/training``

   Dirs used to put data during training .

.. confval:: dirs.model_dir
   :type: ``str``
   :default: ``${project_dir}/models``

   Model dir.

.. confval:: dirs.output_vector_dir
   :type: ``str``
   :default: ``${project_dir}/output_vector/${predict:image_layer}``

   Output vector dir.

.. confval:: dirs.base_image_dir
   :type: ``str``
   :default: ``${projects_dir}/_image_cache``

   Dir with the images we want predictions for.

.. confval:: dirs.predict_image_input_subdir
   :type: ``str``
   :default: ``${predict:image_pixel_width}x${predict:image_pixel_height}_${predict:image_pixels_overlap}pxOverlap``


.. confval:: dirs.predict_image_input_dir
   :type: ``str``
   :default: ``${base_image_dir}/${predict:image_layer}/${predict_image_input_subdir}``


.. confval:: dirs.predict_image_output_basedir
   :type: ``str``
   :default: ``${predict_image_input_dir}``


.. confval:: dirs.predictsample_image_input_subdir
   :type: ``str``
   :default: ``${train:image_pixel_width}x${train:image_pixel_height}``

   Dir with sample images for use during training.

   Remark: these samples are meant to check the training quality, so by default
           the train image size is used!!!

.. confval:: dirs.predictsample_image_input_dir
   :type: ``str``
   :default: ``${base_image_dir}/${predict:image_layer}_testsample/${predictsample_image_input_subdir}``


.. confval:: dirs.predictsample_image_output_basedir
   :type: ``str``
   :default: ``${predictsample_image_input_dir}``



[files]
-------

Settings concerning some specific file paths.

.. confval:: files.model_json_filepath
   :type: ``str``
   :default: ``${dirs:model_dir}/${model:architecture}.json``

   File path that will be used to save/load the keras model definition.

.. confval:: files.image_layers_config_filepath
   :type: ``str``
   :default: ``${dirs:projects_dir}/imagelayers.ini``


.. confval:: files.cancel_filepath
   :type: ``str``
   :default: ``${dirs:projects_dir}/cancel.txt``

   File path of file that if it exists cancels the current processing.


[email]
-------

Email config to use to send progress info to.

.. confval:: email.enabled
   :type: ``bool``
   :default: ``False``

   Set enabled to True to enable sending mails.

.. confval:: email.from
   :type: ``str``
   :default: ``sample@samplemail.be``

   Email address to send task status info from.

.. confval:: email.to
   :type: ``str``
   :default: ``sample@samplemail.be``

   Email address to send task status info to.

.. confval:: email.smtp_server
   :type: ``str``
   :default: ``server.for.emails.be``

   Smtp server to use .

.. confval:: email.mail_server_username
   :type: ``str``
   :default: ``""``

   Username to use to login to smtp server (in some cases optional).

.. confval:: email.mail_server_password
   :type: ``str``
   :default: ``""``

   Password to use to login to smtp server (in some cases optional).


[logging]
---------

Logging configuration.

.. confval:: logging.nb_logfiles_tokeep
   :type: ``int``
   :default: ``30``

   The number of log files to keep in a log dir.

.. confval:: logging.logconfig
   :type: ``dict``
   :default: ``{}`` (see description)

   Config to use for the logging.

   This config is in json, following the conventions as required by
   logging.dictConfig.
   https://docs.python.org/3/library/logging.config.html#logging-config-dictschema

   Mind: the location for file logging.


[cleanup]
---------

Config to manage the cleanup of old models, trainings and predictions.

.. confval:: cleanup.simulate
   :type: ``bool``
   :default: ``False``

   When simulate is True, no files are actually deleted, only logging is written.

.. confval:: cleanup.model_versions_to_retain
   :type: ``int``
   :default: ``-1``

   The number of versions to retain the models for.

   If <0, all versions are retained.

.. confval:: cleanup.training_versions_to_retain
   :type: ``int``
   :default: ``-1``

   The number of version the retain the training data directories for.

   If <0, all versions are retained.

.. confval:: cleanup.prediction_versions_to_retain
   :type: ``int``
   :default: ``-1``

   The number of versions to retain the prediction files for.

   If <0, all versions are retained.

.. |br| raw:: html

  <br/>