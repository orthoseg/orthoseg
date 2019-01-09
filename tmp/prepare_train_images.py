# -*- coding: utf-8 -*-
"""
Script to prepare the train images if they aren't in the correct format:
    * convert to .tif
    * crop the images to a specified width and height
    * make sure the mask images are binary masks

@author: Pieter Roggemans
"""

import os
import glob

import numpy as np
import rasterio as rio
import rasterio.plot as rio_plot

import log_helper

#********************
# FUNTIONS
#********************
def prepare_train_images(dir_image_src: str,
                         dir_image_dest: str,
                         dest_width: int,
                         dest_height: int,
                         to_black_white: bool = False,
                         dir_backup_coords: str = None):
    
    logger.info(f"Prepare images from {dir_image_src}")

    # Get list of all image files to process...
    image_filepaths = []
    input_ext = ['.tif', '.jpg']
    for input_ext_cur in input_ext:
        image_filepaths.extend(glob.glob(f"{dir_image_src}{os.sep}*{input_ext_cur}"))

    if not os.path.exists(dir_image_dest):
        os.mkdir(dir_image_dest)
    
    # Prepare all files
    nb_problems = 0
    for image_filepath in image_filepaths:
        
        logger.debug(f"Process file {image_filepath}")
        # Prepare the filepath for the output
        input_filepath_noext, input_ext = os.path.splitext(image_filepath)
        input_filename = os.path.basename(image_filepath)
                
        # Read input file and get all needed info from it...
        with rio.open(image_filepath) as image_ds:
            image_profile = image_ds.profile
           
            if not to_black_white:
                image_channels_output = image_profile['count']
            else:
                image_channels_output = 1
            image_transform = image_ds.transform
            image_bounds = image_ds.bounds

            # Read pixels
            if not to_black_white:
                image_data = image_ds.read()
                # Change from (channels, width, height) tot (width, height, channels)
                image_data = rio_plot.reshape_as_image(image_data)
            else:
                image_data = image_ds.read(1)
                # Add axis at the back to have shape (width, height, channels)
                image_data = np.expand_dims(image_data, axis=2)

        # If coordinate info in file metadata, use it
        coords_found = False
        if image_bounds.left > 0:
            logger.debug(f"Bounds found in metadata: {image_bounds}, for {input_filename}")
            coords_found = True
        else:
            # else look for the info in the filename
            input_filename_noext = os.path.split(input_filepath_noext)[1]
            
            filename_splitted = input_filename_noext.split('_')
            if len(filename_splitted) >= 4:
                logger.debug(f"Bounds found in filename: {filename_splitted}, for {input_filename}")
                coords_found = True
            elif dir_backup_coords:
                logger.debug(f"Search bounds in backup dir: {dir_backup_coords}")

                search_string_backup = f"{dir_backup_coords}{os.sep}{input_filename_noext}*"
                filepaths_backup = glob.glob(search_string_backup)
                
                if len(filepaths_backup) == 1:
                    # Read input file and get all needed info from it...
                    with rio.open(filepaths_backup[0]) as image_ds:
                        image_transform = image_ds.transform
                        image_bounds = image_ds.bounds

                    if image_bounds.left > 0:                    
                        coords_found = True
                elif len(filepaths_backup) > 1:
                    logger.error("Multiple files found in beckup dir for {search_string_backup}")
                else:
                    logger.warning(f"No backup bounds info found for {image_filepath} with search_string_backup: {search_string_backup}")
                    
        if coords_found == False:
            nb_problems += 1
            logger.debug(f"Bounds NOT FOUND for {os.path.basename(image_filepath)}")
                
        input_width = image_data.shape[0]
        input_height = image_data.shape[1]

        # Resize image           
        if input_width > dest_width or input_height > dest_height:
            x_min = (int)((input_width-dest_width)/2)
            y_min = (int)((input_height-dest_height)/2)
            x_max = (int)(x_min+dest_width)
            y_max = (int)(y_min+dest_height)
                    
            image_data = image_data[x_min:x_max, y_min:y_max, :]

            # TODO: the transfor and the file name should be changed as well
            # but it is not necessary for training... so

        # For mask images all pixels should be either black or white...
        if to_black_white:
            if image_profile['dtype'] == rio.uint8:
                image_data[image_data >= 127] = 255
                image_data[image_data < 127] = 0
            else:
                raise Exception(f"Unsupported dtype to apply to_black_white on: {image_profile['dtype']}")

        # Prepare output filepath
        pixelsize_x = (image_bounds.right - image_bounds.left)/input_width
        pixelsize_y = (image_bounds.top - image_bounds.bottom)/input_height
        output_bounds_left = image_bounds.left + ((input_width-dest_width)/2) * pixelsize_x
        output_bounds_bottom = image_bounds.bottom + ((input_height-dest_height)/2) * pixelsize_y
        output_bounds_right = image_bounds.right - ((input_width-dest_width)/2) * pixelsize_x
        output_bounds_top = image_bounds.top - ((input_height-dest_height)/2) * pixelsize_y
        
        #dest_filepath = input_filepath_noext.replace(dir_image_src, dir_image_dest) + '.tif'
        dest_filepath = f"{dir_image_dest}{os.sep}{output_bounds_left}_{output_bounds_bottom}_{output_bounds_right}_{output_bounds_top}_{dest_width}_{dest_height}.tif"

        # Write to tiff file        
        image_data = rio_plot.reshape_as_raster(image_data)
        with rio.open(dest_filepath, 'w', driver='GTiff', compress='lzw', 
                      height=dest_height, width=dest_width, count=image_channels_output, 
                      dtype=image_profile['dtype'], 
                      crs=image_profile['crs'], 
                      transform=image_transform) as dst:
            dst.write(image_data)

    logger.info(f"Number of images found without coo info: {nb_problems}, on {len(image_filepaths)} images")
           
if __name__ == '__main__':

    project_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation\\greenhouses"
    train_dir = os.path.join(project_dir, "train")
    
    # Main initialisation of the logging
    log_dir = os.path.join(project_dir, "log")
    logger = log_helper.main_log_init(log_dir, __name__)
    logger.info(f"Start prepare of train images")

    # Prepare the files
    prepare_train_images(os.path.join(train_dir, 'mask_source'),
                         os.path.join(train_dir, 'mask_prepared'),
                         dest_width=512,
                         dest_height=512,
                         to_black_white=True,
                         dir_backup_coords=os.path.join(train_dir, 'image_source'))
    prepare_train_images(os.path.join(train_dir, 'image_source'),
                         os.path.join(train_dir, 'image_prepared'),
                         dest_width=512,
                         dest_height=512,
                         dir_backup_coords=os.path.join(train_dir, 'mask_source'))
