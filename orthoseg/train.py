# -*- coding: utf-8 -*-
"""
Module to make it easy to start a training session.
"""

import argparse
import os
from pathlib import Path
import shlex
import sys

from orthoseg.helpers import config_helper as conf
from orthoseg.helpers import log_helper
from orthoseg.lib import prepare_traindatasets as prep
from orthoseg.lib import predicter
from orthoseg.lib import trainer
from orthoseg.model import model_factory as mf
from orthoseg.model import model_helper as mh

def train_argstr(argstr):
    args = shlex.split(argstr)
    train_args(args)

def train_args(args):

    ##### Interprete arguments #####
    parser = argparse.ArgumentParser(add_help=False)

    # Required arguments
    required = parser.add_argument_group('Required arguments')
    required.add_argument("--configfile", type=str, required=True,
            help="The project config file to use")
    
    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    # Add back help 
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
            help='Show this help message and exit')

    # Interprete arguments
    args = parser.parse_args(args)

    ##### Run! #####
    train(projectconfig_path=Path(args.configfile))

def train(
        projectconfig_path: Path,
        imagelayerconfig_path: Path = None):
    """
    Run a training session.

    Args:
        projectconfig_path (Path): Path to the projects config file.
        imagelayerconfig_path (Path, optional): Path to the imagelayer config file. If not specified, 
            the path specified in files.image_layers_config_filepath in the project config will be used. 
            Defaults to None.
    """
    ##### Init #####   
    # Load config
    config_filepaths = conf.search_projectconfig_files(projectconfig_path=projectconfig_path)
    conf.read_project_config(config_filepaths, imagelayerconfig_path)
    
    # Main initialisation of the logging
    global logger
    logger = log_helper.main_log_init(conf.dirs.getpath('log_training_dir'), __name__)      
    logger.debug(f"Config used: \n{conf.pformat_config()}")
    # TODO: add something to delete old data, predictions???

    # First check if the segment_subject has a valid name
    segment_subject = conf.general['segment_subject']
    if segment_subject == 'MUST_OVERRIDE':
        raise Exception("The segment_subject parameter needs to be overridden in the subject specific config file!!!") 
    elif '_' in segment_subject:
        raise Exception(f"The segment_subject parameter should not contain '_', so this is invalid: {segment_subject}!!!") 

    # Create the output dir's if they don't exist yet...
    for dir in [conf.dirs.getpath('project_dir'), conf.dirs.getpath('training_dir')]:
        if dir and not dir.exists():
            dir.mkdir()
    
    ##### If the training data doesn't exist yet, create it #####
    # Get the label input info, and clean it so it is practical to use
    label_files = conf.train.getdict('label_datasources')
    for label_file_key in label_files:
        # Convert the str file paths to Path objects
        label_files[label_file_key]['locations_path'] = Path(
                label_files[label_file_key]['locations_path'])
        label_files[label_file_key]['data_path'] = Path(
                label_files[label_file_key]['data_path'])
    first_label_file = list(label_files)[0]
    train_image_layer = label_files[first_label_file]['image_layer']
    train_projection = conf.image_layers[train_image_layer]['projection']
    classes = conf.train.getdict('classes')

    # Now create the train datasets (train, validation, test)
    force_model_traindata_id = conf.train.getint('force_model_traindata_id')
    if force_model_traindata_id > -1:
        training_dir = conf.dirs.getpath('training_train_basedir') / f"{force_model_traindata_id:02d}"
        traindata_id = force_model_traindata_id
    else:
        logger.info("Prepare train, validation and test data")
        training_dir, traindata_id = prep.prepare_traindatasets(
                label_files=label_files,
                classes=classes,
                image_layers=conf.image_layers,
                training_dir=conf.dirs.getpath('training_dir'),
                image_pixel_x_size=conf.train.getfloat('image_pixel_x_size'),
                image_pixel_y_size=conf.train.getfloat('image_pixel_y_size'),
                image_pixel_width=conf.train.getint('image_pixel_width'),
                image_pixel_height=conf.train.getint('image_pixel_height'))
    logger.info(f"Traindata dir to use is {training_dir}, with traindata_id: {traindata_id}")
    traindata_dir = training_dir / 'train'
    validationdata_dir = training_dir / 'validation'
    testdata_dir = training_dir / 'test'
    
    ##### Check if training is needed #####
    # Get hyper parameters from the config
    architectureparams = mh.ArchitectureParams(
            architecture=conf.model['architecture'],
            nb_classes=len(classes),   
            nb_channels=conf.model.getint('nb_channels'),
            architecture_id=conf.model.getint('architecture_id'))
    trainparams = mh.TrainParams(
            trainparams_id=conf.train.getint('trainparams_id'),
            image_augmentations=conf.train.getdict('image_augmentations'),
            mask_augmentations=conf.train.getdict('mask_augmentations'),
            class_weights=[classes[class_name]['weight'] for class_name in classes],
            batch_size=conf.train.getint('batch_size_fit'), 
            optimizer=conf.train.get('optimizer'), 
            optimizer_params=conf.train.getdict('optimizer_params'), 
            loss_function=conf.train.get('loss_function'), 
            monitor_metric=conf.train.get('monitor_metric'), 
            monitor_metric_mode=conf.train.get('monitor_metric_mode'), 
            save_format=conf.train.get('save_format'), 
            save_best_only=conf.train.getboolean('save_best_only'), 
            save_min_accuracy=conf.train.getfloat('save_min_accuracy'),
            nb_epoch=conf.train.getint('max_epoch'),
            earlystop_patience=conf.train.getint('earlystop_patience'),
            earlystop_monitor_metric=conf.train.get('earlystop_monitor_metric'))

    # Check if there exists already a model for this train dataset + hyperparameters
    model_dir = conf.dirs.getpath('model_dir')
    segment_subject = conf.general['segment_subject']
    best_model_curr_train_version = mh.get_best_model(
            model_dir=model_dir, 
            segment_subject=segment_subject,
            traindata_id=traindata_id,
            architecture_id=architectureparams.architecture_id,
            trainparams_id=trainparams.trainparams_id)

    # Determine if training is needed,...
    resume_train = conf.train.getboolean('resume_train')
    if resume_train is False:
        # If no (best) model found, training needed!
        if best_model_curr_train_version is None:
            train_needed = True
        elif conf.train.getboolean('force_train') is True:
            train_needed = True
        else:
            logger.info("JUST PREDICT, without training: preload_existing_model is false and model found")
            train_needed = False
    else:
        # We want to preload an existing model and models were found
        if best_model_curr_train_version is None:
            logger.info(f"PRELOAD model and continue TRAINING it: {best_model_curr_train_version['filename']}")
            train_needed = True
        else:
            message = "STOP: preload_existing_model is true but no model was found!"
            logger.error(message)
            raise Exception(message)
    
    ##### Train!!! #####
    if train_needed is True:

        # If a model already exists, use it to predict (possibly new) training and 
        # validation dataset. This way it is possible to have a quick check on errors
        # in (new) added labels in the datasets.

        # Get the current best model that already exists for this subject
        best_recent_model = mh.get_best_model(model_dir=model_dir)
        if best_recent_model is not None:
            try:
                logger.info(f"Load model + weights from {best_recent_model['filepath']}")    
                model = mf.load_model(best_recent_model['filepath'], compile=False)            
                logger.info("Loaded model + weights")

                # Prepare output subdir to be used for predictions
                predict_out_subdir, _ = os.path.splitext(best_recent_model['filename'])
                
                # Predict training dataset
                predicter.predict_dir(
                        model=model,
                        input_image_dir=traindata_dir / 'image',
                        output_base_dir=traindata_dir / predict_out_subdir,
                        projection_if_missing=train_projection,
                        input_mask_dir=traindata_dir / 'mask',
                        batch_size=conf.train.getint('batch_size_predict'), 
                        evaluate_mode=True)
                    
                # Predict validation dataset
                predicter.predict_dir(
                        model=model,
                        input_image_dir=validationdata_dir / 'image',
                        output_base_dir=validationdata_dir / predict_out_subdir,
                        projection_if_missing=train_projection,
                        input_mask_dir=validationdata_dir / 'mask',
                        batch_size=conf.train.getint('batch_size_predict'), 
                        evaluate_mode=True)
                del model
            except Exception as ex:
                logger.warn(f"Exception trying to predict with old model: {ex}")
        
        # Now we can really start training
        logger.info('Start training')
        model_preload_filepath = None
        if best_model_curr_train_version is not None:
            model_preload_filepath = best_model_curr_train_version['filepath']
        elif conf.train.getboolean('preload_with_previous_traindata'):
            best_model_for_architecture = mh.get_best_model(
                    model_dir=model_dir, 
                    segment_subject=segment_subject)
            if best_model_for_architecture is not None:
                model_preload_filepath = best_model_for_architecture['filepath']
        
        # Combine all hyperparameters in hyperparams object
        hyperparams = mh.HyperParams(
                architecture=architectureparams,
                train=trainparams)

        trainer.train(
                traindata_dir=traindata_dir,
                validationdata_dir=validationdata_dir,
                model_save_dir=model_dir,
                segment_subject=segment_subject,
                traindata_id=traindata_id,
                hyperparams=hyperparams,
                model_preload_filepath=model_preload_filepath,
                image_width=conf.train.getint('image_pixel_width'),
                image_height=conf.train.getint('image_pixel_height')) 
    
        # Now get the best model found during training
        best_model_curr_train_version = mh.get_best_model(
                model_dir=model_dir, 
                segment_subject=segment_subject,
                traindata_id=traindata_id)

    # Now predict on the train,... data  
    logger.info(f"PREDICT test data with best model: {best_model_curr_train_version['filename']}")
    
    # Load prediction model...
    logger.info(f"Load model + weights from {best_model_curr_train_version['filepath']}")    
    model = mf.load_model(best_model_curr_train_version['filepath'], compile=False)            
    logger.info("Loaded model + weights")
    
    # Prepare output subdir to be used for predictions
    predict_out_subdir, _ = os.path.splitext(best_model_curr_train_version['filename'])
    
    # Predict training dataset
    predicter.predict_dir(
            model=model,
            input_image_dir=traindata_dir / 'image',
            output_base_dir=traindata_dir / predict_out_subdir,
            projection_if_missing=train_projection,
            input_mask_dir=traindata_dir / 'mask',
            batch_size=conf.train.getint('batch_size_predict'), 
            evaluate_mode=True)
    
    # Predict validation dataset
    predicter.predict_dir(
            model=model,
            input_image_dir=validationdata_dir / 'image',
            output_base_dir=validationdata_dir / predict_out_subdir,
            projection_if_missing=train_projection,
            input_mask_dir=validationdata_dir / 'mask',
            batch_size=conf.train.getint('batch_size_predict'), 
            evaluate_mode=True)

    # Predict test dataset, if it exists
    if testdata_dir is not None and testdata_dir.exists():
        predicter.predict_dir(
                model=model,
                input_image_dir=testdata_dir / 'image',
                output_base_dir=testdata_dir / predict_out_subdir, 
                projection_if_missing=train_projection,
                input_mask_dir=testdata_dir / 'mask',
                batch_size=conf.train.getint('batch_size_predict'), 
                evaluate_mode=True)
    
    # Predict extra test dataset with random images in the roi, to add to 
    # train and/or validation dataset if inaccuracies are found
    # -> this is very useful to find false positives to improve the datasets
    if conf.dirs.getpath('predictsample_image_input_dir').exists():
        predicter.predict_dir(
                model=model,
                input_image_dir=conf.dirs.getpath('predictsample_image_input_dir'),
                output_base_dir=conf.dirs.getpath('predictsample_image_output_basedir') / predict_out_subdir,
                projection_if_missing=train_projection,
                batch_size=conf.train.getint('batch_size_predict'), 
                evaluate_mode=True)
    
if __name__ == '__main__':
    train_args(sys.argv[1:])
    