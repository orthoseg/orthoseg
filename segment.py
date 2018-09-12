# -*- coding: utf-8 -*-
"""
Module with high-level operations to segment images

@author: Pieter Roggemans
"""

import logging
import os
import glob
import shutil

import numpy as np
import keras as kr
import rasterio as rio
import rasterio.features as rio_features
import rasterio.plot as rio_plot
import shapely
import shapely.geometry

import model as m
import data
import postprocess as postp

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
FORMAT_GEOTIFF = 'image/geotiff'

# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def train(traindata_dir: str,
          validationdata_dir: str,
          image_subdir: str,
          mask_subdir: str,
          model_dir: str,
          model_basename: str,
          model_preload_filepath: str = None,
          batch_size: int = 32,
          train_augmented_dir: str = None):

    image_width = 512
    image_height = 512

    # These are the augmentations that will be applied to the input training images/masks
    # Remark: fill_mode + cval are defined as they are so missing pixels after eg. rotation
    #         are filled with 0, and so the mask will take care that they are +- ignored.
    data_gen_train_args = dict(rotation_range=90.0,
                               fill_mode='constant',
                               cval=0,
                               rescale=1./255,
                               width_shift_range=0.05,
                               height_shift_range=0.05,
                               shear_range=0.0,
                               zoom_range=0.05,
                               horizontal_flip=True,
                               vertical_flip=True)

    train_gen = data.create_train_generator(input_data_dir=traindata_dir,
                            image_subdir=image_subdir, mask_subdir=mask_subdir,
                            aug_dict=data_gen_train_args, batch_size=batch_size,
                            target_size=(image_width, image_height),
                            class_mode=None,
                            save_to_dir=train_augmented_dir)

    data_gen_validation_args = dict(rescale=1./255)
    validation_gen = data.create_train_generator(input_data_dir=validationdata_dir,
                            image_subdir=image_subdir, mask_subdir=mask_subdir,
                            aug_dict=data_gen_validation_args, batch_size=batch_size,
                            target_size=(image_width, image_height),
                            class_mode=None,
                            save_to_dir=None)

    # Create a unet model
    model = m.get_unet(input_width=image_width, input_height=image_height, n_channels=3,
                       n_classes=1, init_with_vgg16=True,
                       loss_mode='binary_crossentropy')
#                       loss_mode='bcedice')
    if model_preload_filepath:
        if not os.path.exists(model_preload_filepath):
            message = f"Error: preload model file doesn't exist: {model_preload_filepath}"
            logger.critical(message)
            raise Exception(message)
        model.load_weights(model_preload_filepath)

    # Define some callbacks for the training
    # TODO: nakijken of val_loss of loss gemonitored moet worden
    model_best_filepath = f"{model_dir}{os.sep}{model_basename}_best.hdf5"
    model_checkpoint_best = kr.callbacks.ModelCheckpoint(model_best_filepath, monitor='loss',
                                                         verbose=1, save_best_only=True)
#    model_detailed_filepath = f"{model_dir}{os.sep}{model_basename}" + "_{epoch:02d}_{val_loss:.5f}.hdf5"
    model_detailed_filepath = f"{model_dir}{os.sep}{model_basename}" + "_{epoch:02d}_{loss:.5f}.hdf5"
    model_checkpoint = kr.callbacks.ModelCheckpoint(model_detailed_filepath, monitor='loss',
                                                    verbose=1, save_best_only=True)
    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10, verbose=0, mode='auto')
    reduce_lr = kr.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                               patience=5, min_lr=0.001)
    tensorboard_log_dir = f"{model_dir}{os.sep}{model_basename}" + "_tensorboard_log"
    tensorboard_logger = kr.callbacks.TensorBoard(log_dir=tensorboard_log_dir)
    csv_log_filepath = f"{model_dir}{os.sep}{model_basename}" + '_log.csv'
    csv_logger = kr.callbacks.CSVLogger(csv_log_filepath, append=True, separator=';')

    # Start training
    train_dataset_size = len(glob.glob(f"{traindata_dir}{os.sep}{image_subdir}{os.sep}*.*"))
    train_steps_per_epoch = int(train_dataset_size/batch_size)
    validation_dataset_size = len(glob.glob(f"{validationdata_dir}{os.sep}{image_subdir}{os.sep}*.*"))
    validation_steps_per_epoch = int(validation_dataset_size/batch_size)
    model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=50,
                        validation_data=validation_gen,
                        validation_steps=validation_steps_per_epoch,       # Number of items in validation/batch_size
                        callbacks=[model_checkpoint, model_checkpoint_best , early_stopping,
                                   reduce_lr, tensorboard_logger, csv_logger])

def predict(model_to_use_filepath: str,
            input_image_dir: str,
            output_predict_dir: str,
            input_ext: str = '.tif',
            input_mask_dir: str = None,
            force: bool = False):

    # Check if the input parameters are correct...
    if not model_to_use_filepath or not os.path.exists(model_to_use_filepath):
        message = f"Error: input model in is mandatory, model_to_use_filepath: <{model_to_use_filepath}>!"
        logger.critical(message)
        raise Exception(message)

    # Create the output dir's if they don't exist yet...
    for dir in [output_predict_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    image_filepaths = []
    for input_ext_cur in input_ext:
        image_filepaths.extend(glob.glob(f"{input_image_dir}{os.sep}**{os.sep}*{input_ext_cur}", recursive=True))
    logger.info(f"Found {len(image_filepaths)} {input_ext} images to predict on in {input_image_dir}")
    force = False
    model = None
    for index, image_filepath in enumerate(sorted(image_filepaths)):

        # Prepare the filepath for the output
        tmp_filepath = image_filepath.replace(input_image_dir,
                                              output_predict_dir)
        image_pred_dir = os.path.split(tmp_filepath)[0]
        if not os.path.exists(image_pred_dir):
            os.mkdir(image_pred_dir)
        image_pred_filepath_noext, image_pred_ext = os.path.splitext(tmp_filepath)
        image_pred_filepath = f"{image_pred_filepath_noext}_pred{image_pred_ext}"

        # If force is false and file exists... skip
        if force is False and os.path.exists(image_pred_filepath):
            logger.info(f"Predict for image already exists and force is False, so skip: {image_filepath}")
            continue
        else:
            logger.info(f"Start predict for image {image_filepath}")

        # Read info file and get all needed info from it...
        with rio.open(image_filepath) as image_ds:
            image_profile = image_ds.profile
            logger.debug(f"image_profile: {image_profile}")
            image_width = image_profile['width']
            image_height = image_profile['height']
            image_channels = image_profile['count']
            image_transform = image_ds.transform

            # Read pixels
            image_arr = image_ds.read()
            # Change from (channels, width, height) tot (width, height, channels)
            image_arr = rio_plot.reshape_as_image(image_arr)

#        image = kr.preprocessing.image.load_img(image_filepath)
#        image_arr = kr.preprocessing.image.img_to_array(image)
#        image_arr = image_arr.reshape((image_width, image_height, image_channels))

        # Make sure the pixels values are between 0 and 1
        image_arr = image_arr / 255

        # Input of predict must be numpy array with shape: (nb images, width, height, channels)!
        image_arr = np.expand_dims(image_arr, axis=0)

        # Load model if it isn't loaded yet...
        # Remark: it is created only once, so all images need to have the same size!!!
        if not model:
            logger.info(f"Load model with weights from {model_to_use_filepath}")

            #    model = kr.models.load_model(model_to_use_filepath)
 #           model = m.get_unet(input_width=image_profile['width'], input_height=image_profile['height'],
 #                              n_channels=image_profile['count'], n_classes=1)
            model = m.get_unet(input_width=image_width, input_height=image_height,
                               n_channels=image_channels, n_classes=1)
            model.load_weights(model_to_use_filepath)

        # Predict!
        logger.info("Start prediction")
        image_pred_orig = model.predict(image_arr, batch_size=1)
        logger.debug("After prediction")

        # Only take the first image, as there is only one...
        image_pred = image_pred_orig[0]

        # Check the number of channels of the output prediction
        n_channels = image_pred.shape[2]
        if n_channels > 1:
            raise Exception(f"Not implemented: processing prediction output with multiple channels: {n_channels}")

        logger.debug("Save original prediction")
        kr.preprocessing.image.save_img(image_pred_filepath, image_pred)

        # Cleanup
        # Make the array 2 dimensial for the next algorithm. Is no problem if there
        # is only one channel
        image_pred = image_pred.reshape((image_width, image_height))

        # Cleanup the image so it becomes a clean 2 color one instead of grayscale
        logger.debug("Clean prediction")
        image_pred = postp.region_segmentation(image_pred)

        # Convert the output image to uint [0-255] instead of float [0,1]
        image_pred = (image_pred * 255).astype(np.uint8)

        # Write the output to file
        logger.debug("Write cleaned prediction")
        # TODO: should always be done using input image, but in my test data
        # doesn't contain geo yet
        if input_mask_dir:
            tmp_filepath = image_filepath.replace(input_image_dir,
                                                  input_mask_dir)
        else:
            tmp_filepath = image_filepath
        # First read the properties of the input image to copy them for the output
        with rio.open(tmp_filepath) as image_ds:
            image_profile = image_ds.profile
            image_transform = image_ds.transform

        # Use meta attributes of the source image, but set band count to 1,
        # dtype to uint8 and specify LZW compression.
        image_profile.update(dtype=rio.uint8, count=1, compress='lzw')

        image_pred_cleaned_filepath = f"{image_pred_filepath_noext}_pred_cleaned{image_pred_ext}"
        with rio.open(image_pred_cleaned_filepath, 'w', **image_profile) as dst:
            dst.write(image_pred.astype(rio.uint8), 1)

        # Polygonize result
        # Returns a list of tupples with (geometry, value)
        shapes = rio_features.shapes(image_pred.astype(rio.uint8),
                                     mask=image_pred.astype(rio.uint8),
                                     transform=image_transform)

        # Write WKT's of original + simplified shapes
        poly_wkt_filepath = f"{image_pred_filepath_noext}_pred_cleaned.wkt"
        poly_wkt_simpl_filepath = f"{image_pred_filepath_noext}_pred_cleaned_simpl.wkt"

    #    if os.path.exists(poly_wkt_filepath):
    #        os.remove(poly_wkt_filepath)
        logger.debug('Before writing orig geom wkt file')

        geoms_simpl = []
        with open(poly_wkt_filepath, 'w') as dst:
            for shape in list(shapes):
                geom, value = shape
                geom_sh = shapely.geometry.shape(geom)
                dst.write(f"{geom_sh}\n")

                # simplify and rasterize for easy comparison with original masks
                # preserve_topology is slower bu makes sure no polygons are removed
                geom_simpl = geom_sh.simplify(2, preserve_topology=True)
                if not geom_simpl.is_empty:
                    geoms_simpl.append(geom_simpl)

        logger.debug('Before writing simpl geom wkt file')

        with open(poly_wkt_simpl_filepath, 'w') as dst_simpl:
            for geom_simpl in geoms_simpl:
                dst_simpl.write(f"{geom_simpl}\n")

        # Write simplified wkt result to raster for debugging. Use the same
        # file profile as created before for writing the raw prediction result
        logger.debug('Before writing simpl rasterized file')

        image_pred_simpl_filepath = f"{image_pred_filepath_noext}_pred_cleaned_simpl2{image_pred_ext}"
        with rio.open(image_pred_simpl_filepath, 'w', **image_profile) as dst:
            # this is where we create a generator of geom, value pairs to use in rasterizing
    #        shapes = ((geom,value) for geom, value in zip(counties.geometry, counties.LSAD_NUM))
            logger.debug('Before rasterize')
            if geoms_simpl:
                out_arr = dst.read(1)
                burned = rio_features.rasterize(shapes=geoms_simpl, fill=0,
                                                default_value=255, out=out_arr,
                                                transform=image_transform)
                logger.debug(burned)
        #        dst.write_band(1, burned)
                dst.write(burned, 1)

        # Copy the mask if an input_mask_dir is specified
        if input_mask_dir and os.path.exists(input_mask_dir):
            # Prepare the file paths
            mask_filepath = image_filepath.replace(input_image_dir,
                                                   input_mask_dir)
            mask_copy_dest_filepath = mask_filepath.replace(input_mask_dir,
                                                            output_predict_dir)
            filepath_noext, filepath_ext = os.path.splitext(mask_copy_dest_filepath)
            mask_copy_dest_filepath = f"{filepath_noext}_mask{filepath_ext}"

            # Copy if the file doesn't exist yet
            if not os.path.exists(mask_copy_dest_filepath):
                shutil.copyfile(mask_filepath, mask_copy_dest_filepath)

        # Copy the input image if it doesn't exist yet in output path
        image_copy_dest_filepath = image_filepath.replace(input_image_dir,
                                                          output_predict_dir)
        if not os.path.exists(image_copy_dest_filepath):
            shutil.copyfile(image_filepath, image_copy_dest_filepath)
