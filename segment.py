# -*- coding: utf-8 -*-
"""
Module with high-level operations to segment images

@author: Pieter Roggemans
"""

import logging
import os
import glob
import shutil
import datetime
import concurrent.futures as futures

import numpy as np
import pandas as pd
import keras as kr
import rasterio as rio
import rasterio.features as rio_features
import rasterio.plot as rio_plot
import shapely
import shapely.geometry

import model_factory as m

import data
import postprocess as postp

#-------------------------------------------------------------
# First define/init some general variables/constants
#-------------------------------------------------------------
FORMAT_GEOTIFF = 'image/geotiff'

# Get a logger...
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

#-------------------------------------------------------------
# The real work
#-------------------------------------------------------------

def train(traindata_dir: str,
          validationdata_dir: str,
          image_subdir: str,
          mask_subdir: str,
          segmentation_model: str,
          backbone_name: str,
          model_dir: str,
          model_basename: str,
          model_preload_filepath: str = None,
          batch_size: int = 32,
          nb_epoch: int = 50,
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

    if validationdata_dir:
        data_gen_validation_args = dict(rescale=1./255)
        validation_gen = data.create_train_generator(input_data_dir=validationdata_dir,
                                image_subdir=image_subdir, mask_subdir=mask_subdir,
                                aug_dict=data_gen_validation_args, batch_size=batch_size,
                                target_size=(image_width, image_height),
                                class_mode=None,
                                save_to_dir=None)
    else:
        validation_gen = None

    # Define some callbacks for the training
    model_detailed_filepath = f"{model_dir}{os.sep}{model_basename}" + "_{epoch:03d}_{jaccard_coef_int:.5f}_{val_jaccard_coef_int:.5f}.hdf5"
    #model_detailed_filepath = f"{model_dir}{os.sep}{model_basename}" + "_best_val_loss.hdf5"
    model_checkpoint = kr.callbacks.ModelCheckpoint(model_detailed_filepath, 
                                                    monitor='jaccard_coef_int',
                                                    save_best_only=True,
                                                    save_weights_only=True)
    model_detailed2_filepath = f"{model_dir}{os.sep}{model_basename}" + "_{epoch:03d}_{jaccard_coef_int:.5f}_{val_jaccard_coef_int:.5f}_bestval.hdf5"
    #model_detailed2_filepath = f"{model_dir}{os.sep}{model_basename}" + "_best_loss.hdf5"
    model_checkpoint2 = kr.callbacks.ModelCheckpoint(model_detailed2_filepath, 
                                                     monitor='val_jaccard_coef_int',
                                                     save_best_only=True,
                                                     save_weights_only=True)
    reduce_lr = kr.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                               patience=10, min_lr=1e-20)
    early_stopping = kr.callbacks.EarlyStopping(monitor='jaccard_coef_int', 
                                                patience=100,  
                                                restore_best_weights=False)
    tensorboard_log_dir = f"{model_dir}{os.sep}{model_basename}" + "_tensorboard_log"
    tensorboard_logger = kr.callbacks.TensorBoard(log_dir=tensorboard_log_dir)
    csv_log_filepath = f"{model_dir}{os.sep}{model_basename}" + '_log.csv'
    csv_logger = kr.callbacks.CSVLogger(csv_log_filepath, append=True, separator=';')

    # Get the max epoch number from the log file if it exists...
    start_epoch = 0
    start_learning_rate = 1e-4  # Best set to 0.0001 to start (1e-3 is not ok)
    if os.path.exists(csv_log_filepath):
        logger.info(f"train_log csv exists: {csv_log_filepath}")
        if not model_preload_filepath:
            message = f"STOP: log file exists but preload model file not specified!!!"
            logger.critical(message)
            raise Exception(message)
        
        train_log_csv = pd.read_csv(csv_log_filepath, sep=';')
        logger.debug(f"train_log csv contents:\n{train_log_csv}")
        start_epoch = train_log_csv['epoch'].max()
        start_learning_rate = train_log_csv['lr'].min()
    logger.info(f"start_epoch: {start_epoch}, start_learning_rate: {start_learning_rate}")

    # Create a model
    model = None
    #loss_function = 'bcedice'
    #loss_function = 'binary_crossentropy'
    if not model_preload_filepath:
        # Get the model we want to use
        model = m.get_model(segmentation_model=segmentation_model, 
                            backbone_name=backbone_name,
                            n_channels=3, n_classes=1)
        # Prepare the model for training
        # Default learning rate for Adam: lr=1e-3, but doesn't seem to work well for unet
        #model.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])
        model = m.compile_model(model=model,
                                optimizer=kr.optimizers.Adam(lr=start_learning_rate), 
                                loss_mode='binary_crossentropy')
    else:
        if not os.path.exists(model_preload_filepath):
            message = f"Error: preload model file doesn't exist: {model_preload_filepath}"
            logger.critical(message)
            raise Exception(message)

        '''
        model = m.load_unet_model(filepath=model_preload_filepath,
                                  learning_rate=start_learning_rate)

        '''
        model = m.get_model(segmentation_model=segmentation_model, 
                            backbone_name=backbone_name,
                            n_channels=3, n_classes=1)
        # Prepare the model for training
        model = m.compile_model(model=model,
                                optimizer=kr.optimizers.Adam(lr=start_learning_rate), 
                                loss_mode='binary_crossentropy', 
                                metrics=['binary_accuracy'])
        
#        model = kr.models.load_model(model_preload_filepath)
#        model = m.load_unet_model(model_preload_filepath)
#        model = m.get_unet(input_width=image_width, input_height=image_height,
#                           n_channels=3, n_classes=1)
#                          init_with_vgg16=True, loss_mode='binary_crossentropy)

#        logger.info(f"Load weights from {model_preload_filepath}")
#        model.load_weights(model_preload_filepath)

    # Save the model architecture to json
    json_string = model.to_json()
    with open(f"{model_dir}{os.sep}{model_basename}.json", 'w') as dst:
        dst.write(f"{json_string}")
    
    # Start training
    train_dataset_size = len(glob.glob(f"{traindata_dir}{os.sep}{image_subdir}{os.sep}*.*"))
    train_steps_per_epoch = int(train_dataset_size/batch_size)
    validation_dataset_size = len(glob.glob(f"{validationdata_dir}{os.sep}{image_subdir}{os.sep}*.*"))
    validation_steps_per_epoch = int(validation_dataset_size/batch_size)
    model.fit_generator(train_gen, steps_per_epoch=train_steps_per_epoch, epochs=nb_epoch,
                        validation_data=validation_gen,
                        validation_steps=validation_steps_per_epoch,       # Number of items in validation/batch_size
                        callbacks=[model_checkpoint, model_checkpoint2,
                                   reduce_lr, early_stopping,
                                   tensorboard_logger, csv_logger],
                        initial_epoch=start_epoch)

def predict(model_json_filepath: str,
            model_weights_filepath: str,
            input_image_dir: str,
            output_predict_dir: str,
            border_pixels_to_ignore: int = 0,
            input_mask_dir: str = None,
            batch_size: int = 16,
            evaluate_mode: bool = False,
            force: bool = False):

    # TODO: the real predict code is now mixed with

    # Check if the input parameters are correct...
    # TODO: check on model json as well
    if not model_weights_filepath or not os.path.exists(model_weights_filepath):
        message = f"Error: input model in is mandatory, model_weights_filepath: <{model_weights_filepath}>!"
        logger.critical(message)
        raise Exception(message)

    logger.info(f"Predict for input_image_dir: {input_image_dir}")

    # Create the output dir's if they don't exist yet...
    for dir in [output_predict_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Get list of all image files to process...
    image_filepaths = []
    input_ext = ['.tif', '.jpg']
    for input_ext_cur in input_ext:
        image_filepaths.extend(glob.glob(f"{input_image_dir}{os.sep}**{os.sep}*{input_ext_cur}", recursive=True))
    logger.info(f"Found {len(image_filepaths)} {input_ext} images to predict on in {input_image_dir}")
    
    # Load model...
    logger.info(f"Load model from {model_json_filepath}")
    with open(model_json_filepath, 'r') as src:
        model_json = src.read()
        model = kr.models.model_from_json(model_json)
    logger.info(f"Load weights from {model_weights_filepath}")                
    model.load_weights(model_weights_filepath)
    logger.info("Weights loaded")
    
    #model = m.load_model(model_to_use_filepath)

    # Loop through all files to process them...
    curr_batch_image_data_arr = []
    curr_batch_image_filepath_arr = []
    curr_batch_counter = 0
    curr_predicted_counter = 0
    pool = futures.ThreadPoolExecutor(batch_size)
            
    for i, image_filepath in enumerate(sorted(image_filepaths)):

        # Prepare the filepath for the output
        tmp_filepath = image_filepath.replace(input_image_dir,
                                              output_predict_dir)
        image_pred_dir, image_pred_filename = os.path.split(tmp_filepath)
        
        # If not in evaluate mode, create the complete dir 
        if not evaluate_mode and not os.path.exists(image_pred_dir):
            os.mkdir(image_pred_dir)

        # If force is false and file exists... skip
        image_pred_filename_noext, image_pred_ext = os.path.splitext(image_pred_filename)
        image_pred_files = glob.glob(f"{image_pred_dir}{os.sep}*{image_pred_filename_noext}_pred{image_pred_ext}")
        if force is False and len(image_pred_files) > 0:
            logger.debug(f"Predict for image already exists and force is False, so skip: {image_filepath}")
            continue
        else:
            logger.debug(f"Start predict for image {image_filepath}")

        # Init start time at the first file that isn't skipped
        if curr_predicted_counter == 0:
            start_time = datetime.datetime.now()

        # Read input file and get all needed info from it...
        with rio.open(image_filepath) as image_ds:
            image_profile = image_ds.profile
            logger.debug(f"image_profile: {image_profile}")

            # Read pixels
            image_data = image_ds.read()
            # Change from (channels, width, height) tot (width, height, channels)
            image_data = rio_plot.reshape_as_image(image_data)

        # Make sure the pixels values are between 0 and 1
        image_data = image_data / 255
        
        # Check if the image size is OK for the segmentation model
        '''
        m.check_image_size(segmentation_model=segmentation_model,
                           input_width=image_data.shape[0], 
                           input_height=image_data.shape[1])
        '''

        # Input of predict must be numpy array with shape: 
        #   (images, width, height, channels)!
        curr_batch_image_data_arr.append(image_data)
        curr_batch_image_filepath_arr.append(image_filepath)
        curr_batch_counter += 1
        curr_predicted_counter += 1
        
        # If the batch size is reached or we are at the last images
        if(curr_batch_counter == batch_size
            or i == (len(image_filepaths)-1)):
    
            # Predict!
            logger.info(f"Start prediction for {batch_size} images")
            curr_batch_image_pred = model.predict(np.asarray(curr_batch_image_data_arr), batch_size=batch_size)
            
            # Postprocess all images in the batch in parallel
            logger.info("Start post-processing")
            threads = []
            future_list = []
                         
            # TODO: doing it multi-threaded is not faster at the moment, 
            # maybe if postprocessing is more complicated later reactivate?
            for j in range(len(curr_batch_image_filepath_arr)):
                
                '''
                postprocess_prediction(image_pred_orig=cur_batch_image_pred[j],
                                       image_filepath=curr_batch_image_filepath_arr[j],
                                       input_image_dir=input_image_dir,
                                       output_predict_dir=output_predict_dir,
                                       input_mask_dir=input_mask_dir,
                                       border_pixels_to_ignore=border_pixels_to_ignore,
                                       evaluate_mode=evaluate_mode,
                                       force=force)
                '''
                keyword_params = {'image_pred_orig': curr_batch_image_pred[j],
                          'image_filepath': curr_batch_image_filepath_arr[j],
                          'input_image_dir': input_image_dir,
                          'output_predict_dir': output_predict_dir,
                          'input_mask_dir': input_mask_dir,
                          'border_pixels_to_ignore': border_pixels_to_ignore,
                          'evaluate_mode': evaluate_mode,
                          'force': force}
                
                future_list.append(pool.submit(postprocess_prediction, 
                                               **keyword_params))
            
            # Wait for all postprocessing to be finished
            futures.wait(future_list)
            logger.info("Post-processing ready")
            
            # Reset variables for next batch
            curr_batch_image_data_arr = []
            curr_batch_image_filepath_arr = []
            curr_batch_counter = 0
        
            # Log the progress and prediction speed
            time_passed = (datetime.datetime.now()-start_time).seconds
            if time_passed > 0:
                images_per_hour = ((curr_predicted_counter)/time_passed) * 3600
                logger.info(f"Prediction speed: {images_per_hour:0.0f} images/hour")
     
def postprocess_prediction(image_pred_orig,
                           image_filepath: str,
                           input_image_dir: str,
                           output_predict_dir: str,
                           input_mask_dir: str = None,
                           border_pixels_to_ignore: int = 0,
                           evaluate_mode: bool = False,
                           force: bool = False):
    
    logger.info("start postprocess")
    # Prepare the filepath for the output
    # TODO: this code is +- copied from predict, check if this can be evaded.
    # if in evaluate mode, don't keep the hierarchy from the input dir
    if evaluate_mode:
        image_dir, image_filename = os.path.split(image_filepath)
        tmp_filepath = os.path.join(output_predict_dir, image_filename)
    else:
        tmp_filepath = image_filepath.replace(input_image_dir,
                                              output_predict_dir)
    image_pred_dir, image_pred_filename = os.path.split(tmp_filepath)
    if not os.path.exists(image_pred_dir):
        os.mkdir(image_pred_dir)
    image_pred_filename_noext, image_pred_ext = os.path.splitext(image_pred_filename)
    
    # Check the number of channels of the output prediction
    n_channels = image_pred_orig.shape[2]
    if n_channels > 1:
        raise Exception(f"Not implemented: processing prediction output with multiple channels: {n_channels}")
            
    # Make the array 2 dimensial for the next algorithm. Is no problem if there
    # is only one channel
    image_pred_orig = image_pred_orig.reshape((image_pred_orig.shape[0], image_pred_orig.shape[1]))
    
    # Make the pixels at the borders of the prediction black so they are ignored
    if border_pixels_to_ignore and border_pixels_to_ignore > 0:
        image_pred_orig[0:border_pixels_to_ignore,:] = 0    # Left border
        image_pred_orig[-border_pixels_to_ignore:,:] = 0    # Right border
        image_pred_orig[:,0:border_pixels_to_ignore] = 0    # Top border
        image_pred_orig[:,-border_pixels_to_ignore:] = 0    # Bottom border

    # Check if the result is entirely black... if so no cleanup needed
    all_black = False
    thresshold_ok = 0.5
    if not np.any(image_pred_orig > 0.5):
        logger.debug('Prediction is entirely black!')
        image_pred = postp.thresshold(image_pred_orig, thresshold_ok=thresshold_ok)
        all_black = True
    else:
        # Cleanup the image so it becomes a clean 2 color one instead of grayscale
        logger.debug("Clean prediction")
        image_pred = postp.region_segmentation(image_pred_orig, 
                                               thresshold_ok=thresshold_ok)

    # Convert the output image to uint [0-255] instead of float [0,1]
    image_pred_uint8 = (image_pred * 255).astype(np.uint8)
        
    # If in evaluate mode, put a prefix in the file name
    pred_prefix_str = ''
    if evaluate_mode:
        
        def jaccard_similarity(im1, im2):
            if im1.shape != im2.shape:
                message = f"Shape mismatch: input have different shape: im1: {im1.shape}, im2: {im2.shape}"
                logger.critical(message)
                raise ValueError(message)

            intersection = np.logical_and(im1, im2)
            union = np.logical_or(im1, im2)

            sum_union = float(union.sum())
            if sum_union == 0.0:
                # If 0 positive pixels in union: perfect prediction, so 1
                return 1
            else:
                sum_intersect = intersection.sum()
                return sum_intersect/sum_union

        # If there is a mask dir specified... use the groundtruth mask
        if input_mask_dir and os.path.exists(input_mask_dir):
            # Read mask file and get all needed info from it...
            mask_filepath = image_filepath.replace(input_image_dir,
                                                   input_mask_dir)

            with rio.open(mask_filepath) as mask_ds:
                # Read pixels
                mask_arr = mask_ds.read(1)

            # Make the pixels at the borders of the mask black so they are 
            # ignored in the comparison
            if border_pixels_to_ignore and border_pixels_to_ignore > 0:
                mask_arr[0:border_pixels_to_ignore,:] = 0    # Left border
                mask_arr[-border_pixels_to_ignore:,:] = 0    # Right border
                mask_arr[:,0:border_pixels_to_ignore] = 0    # Top border
                mask_arr[:,-border_pixels_to_ignore:] = 0    # Bottom border
                
            #similarity = jaccard_similarity(mask_arr, image_pred)
            # Use accuracy as similarity... is more practical than jaccard
            similarity = np.equal(mask_arr, image_pred_uint8).sum()/image_pred_uint8.size
            pred_prefix_str = f"{similarity:0.3f}_"
            
            # Copy mask file if the file doesn't exist yet
            mask_copy_dest_filepath = f"{image_pred_dir}{os.sep}{pred_prefix_str}{image_pred_filename_noext}_mask.tif"
            if not os.path.exists(mask_copy_dest_filepath):
                shutil.copyfile(mask_filepath, mask_copy_dest_filepath)

        else:
            # If all_black, no need to calculate again
            if all_black: 
                pct_black = 1
            else:
                # Calculate percentage black pixels
                pct_black = 1 - (image_pred_uint8.sum()/255)/image_pred_uint8.size
            
            # If the result after segmentation is all black, set all_black
            if pct_black == 1:
                # Force the prefix to be really high so it is clear they are entirely black
                pred_prefix_str = "1.001_"
                all_black = True
            else:
                pred_prefix_str = f"{pct_black:0.3f}_"

            # If there are few white pixels, don't save it,
            # because we are in evaluetion mode anyway...
            #if similarity >= 0.95:
                #continue
      
        # Copy the input image if it doesn't exist yet in output path
        image_copy_dest_filepath = f"{image_pred_dir}{os.sep}{pred_prefix_str}{image_pred_filename_noext}{image_pred_ext}"
        if not os.path.exists(image_copy_dest_filepath):
            shutil.copyfile(image_filepath, image_copy_dest_filepath)
       
    # First read the properties of the input image to copy them for the output
    # TODO: should always be done using input image, but in my test data
    # doesn't contain geo yet
    if input_mask_dir:
        tmp_filepath = image_filepath.replace(input_image_dir,
                                              input_mask_dir)
    else:
        tmp_filepath = image_filepath
    with rio.open(tmp_filepath) as image_ds:
        image_profile = image_ds.profile
        image_transform = image_ds.transform

    # Now write original prediction to file
    logger.debug("Save original prediction")
    #logger.info(f"image_profile: {image_profile}")
    
    # Convert the output image to uint [0-255] instead of float [0,1]
    image_pred_orig = (image_pred_orig * 255).astype(np.uint8)
    # Use meta attributes of the source image, except...
    # Rem: dtype float32 used to change as little as possible to original
    image_profile.update(dtype=rio.uint8, count=1, compress='lzw')
    image_pred_orig_filepath = f"{image_pred_dir}{os.sep}{pred_prefix_str}{image_pred_filename_noext}_pred.tif"
#    with rio.open(image_pred_orig_filepath, 'w', **image_profile) as dst:
    with rio.open(image_pred_orig_filepath, 'w', driver='GTiff', compress='lzw',
                  height=image_profile['height'], width=image_profile['width'], 
                  count=1, dtype=rio.uint8, crs=image_profile['crs'], transform=image_transform) as dst:
        dst.write(image_pred_orig.astype(rio.uint8), 1)

    # If the prediction is all black, no need to proceed...
    if all_black:
        return 
    
    # Write the output to file
    logger.debug("Save cleaned prediction")
    # Use meta attributes of the source image, except...
    image_profile.update(dtype=rio.uint8, count=1, compress='lzw')
    image_pred_cleaned_filepath = f"{image_pred_dir}{os.sep}{pred_prefix_str}{image_pred_filename_noext}_pred_cleaned.tif"
    with rio.open(image_pred_cleaned_filepath, 'w', driver='GTiff', compress='lzw',
                  height=image_profile['height'], width=image_profile['width'], 
                  count=1, dtype=rio.uint8, crs=image_profile['crs'], transform=image_transform) as dst:
        dst.write(image_pred_uint8.astype(rio.uint8), 1)

    # Polygonize result
    # Returns a list of tupples with (geometry, value)
    shapes = rio_features.shapes(image_pred_uint8.astype(rio.uint8),
                                 mask=image_pred_uint8.astype(rio.uint8),
                                 transform=image_transform)

    # Convert shapes to shapely geoms + simplify
    geoms = []
    geoms_simpl = []
    for shape in list(shapes):
        geom, value = shape
        geom_sh = shapely.geometry.shape(geom)
        geoms.append(geom_sh)

        # simplify and rasterize for easy comparison with original masks
        # preserve_topology is slower bu makes sure no polygons are removed
        geom_simpl = geom_sh.simplify(1.5, preserve_topology=True)
        if not geom_simpl.is_empty:
            geoms_simpl.append(geom_simpl)

    # Write the original geoms to wkt file
    logger.debug('Before writing orig geom wkt file')
    poly_wkt_filepath = f"{image_pred_dir}{os.sep}{pred_prefix_str}{image_pred_filename_noext}_pred_cleaned.wkt"
    with open(poly_wkt_filepath, 'w') as dst:
        for geom in geoms:
            dst.write(f"{geom}\n")

    # Write the simplified geoms to wkt file
    logger.debug('Before writing simpl geom wkt file')
    poly_wkt_simpl_filepath = f"{image_pred_dir}{os.sep}{pred_prefix_str}{image_pred_filename_noext}_pred_cleaned_simpl.wkt"
    with open(poly_wkt_simpl_filepath, 'w') as dst_simpl:
        for geom_simpl in geoms_simpl:
            dst_simpl.write(f"{geom_simpl}\n")

    # Write simplified wkt result to raster for debugging. Use the same
    # file profile as created before for writing the raw prediction result
    # TODO: doesn't support multiple classes
    logger.debug('Before writing simpl rasterized file')

    image_pred_simpl_filepath = f"{image_pred_dir}{os.sep}{pred_prefix_str}{image_pred_filename_noext}_pred_cleaned_simpl.tif"
    with rio.open(image_pred_simpl_filepath, 'w', driver='GTiff', compress='lzw',
                  height=image_profile['height'], width=image_profile['width'], 
                  count=1, dtype=rio.uint8, crs=image_profile['crs'], transform=image_transform) as dst:
        # this is where we create a generator of geom, value pairs to use in rasterizing
#            shapes = ((geom,value) for geom, value in zip(counties.geometry, counties.LSAD_NUM))
        logger.debug('Before rasterize')
        if geoms_simpl:
            out_arr = dst.read(1)
            burned = rio_features.rasterize(shapes=geoms_simpl, fill=0,
                                            default_value=255, out=out_arr,
                                            transform=image_transform)
#            logger.debug(burned)
            dst.write(burned, 1)
    