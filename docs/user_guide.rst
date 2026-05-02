.. currentmodule:: orthoseg

==========
User guide
==========

The main objective of orthoseg is to make it easier to automatically digitize
features on orthophotos. This manual will guide you through the process of setting up a
new project, creating a training dataset, training a neural network and running a
prediction on an orthophoto.

Smaller projects are no problem to run on a regular laptop or desktop computer. Only
if you want to use more complex models, train on larger training datasets and/or
want to process large areas using a decent CUDA GPU will become very recommended to
reduce waiting times.

Contents
~~~~~~~~

.. toctree::
   :maxdepth: 2
   :numbered:

   Run sample project <user_guide/1_run_sample_project>
   Prepare a new project <user_guide/2_prepare_new_project>
   Finetune your project <user_guide/3_finetune_your_project>
   Create training data <user_guide/4_create_training_data>
   Run train, predict, session <user_guide/5_run_session>
