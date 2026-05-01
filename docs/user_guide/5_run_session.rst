.. currentmodule:: orthoseg

===========================
Run train, predict, session
===========================

Once you have added a meaningful number of (extra) examples, you can do a (new) training
run. The number of added examples that is meaningful will depend on the case, but a
reasonable amount is 50.
Once the model is trained, you can continue by running a prediction and if wanted a
postprocessing step.

If you ran the sample project, these steps will look very familiar:

1. start a conda command prompt

2. activate the orthoseg environment with::
   
      conda activate orthoseg

3. preload the images so they are ready to detect your `{segment_subject}` on, using the
   configuration file `{project_dir}{segment_subject}.ini`::
   
      orthoseg_load_images --config {project_dir}{segment_subject}.ini

4. train a neural network to detect football fields::
   
      orthoseg_train --config {project_dir}{segment_subject}.ini

5. detect the football fields::

      orthoseg_predict --config {project_dir}{segment_subject}.ini


After this completes, the directory `{project_dir}/output_vector` will contain a .gpkg
file with the features found.

Of course it is also possible to script this in your scripting language of choice to
automate this further...

.. note::

   Because tasks often take quite a while, orthoseg maximally tries to resume work that
   was started but was not finished yet. Eg. when predicting a large area, orthoseg will
   save the prediction per image, so if the prediction process is stopped for any reason
   and restarted, it will continue where it stopped.
