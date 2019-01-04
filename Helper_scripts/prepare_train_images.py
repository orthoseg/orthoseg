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

#********************
# FUNTIONS
#********************
def prepare_train_images(dir_src: str,
                         dir_dest: str,
                         dest_width: int,
                         dest_height: int,
                         to_black_white: bool = False):
    
    # Get list of all image files to process...
    image_filepaths = []
    input_ext = ['.tif', '.jpg']
    for input_ext_cur in input_ext:
        image_filepaths.extend(glob.glob(f"{dir_src}{os.sep}*{input_ext_cur}"))

    if not os.path.exists(dir_dest):
        os.mkdir(dir_dest)
    # Move all files
    for image_filepath in image_filepaths:
        
        print(f"Process file {image_filepath}")
        # Prepare the filepath for the output
        input_filepath_noext, input_ext = os.path.splitext(image_filepath)
        dest_filepath = input_filepath_noext.replace(dir_src, dir_dest) + '.tif'
        
        # Read input file and get all needed info from it...
        with rio.open(image_filepath) as image_ds:
            image_profile = image_ds.profile
           
            if not to_black_white:
                image_channels_output = image_profile['count']
            else:
                image_channels_output = 1
            image_transform = image_ds.transform

            # Read pixels
            if not to_black_white:
                image_data = image_ds.read()
                # Change from (channels, width, height) tot (width, height, channels)
                image_data = rio_plot.reshape_as_image(image_data)
            else:
                image_data = image_ds.read(1)
                # Add axis at the back to have shape (width, height, channels)
                image_data = np.expand_dims(image_data, axis=2)

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

        # Write to tiff file
        image_data = rio_plot.reshape_as_raster(image_data)
        with rio.open(dest_filepath, 'w', driver='GTiff', compress='lzw', 
                      height=dest_height, width=dest_width, count=image_channels_output, 
                      dtype=image_profile['dtype'], 
                      crs=image_profile['crs'], 
                      transform=image_transform) as dst:
            dst.write(image_data)
            
#********************
# GO
#********************   
base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation\\greenhouses\\train"

# Prepare the files
prepare_train_images(os.path.join(base_dir, 'mask_source'),
                     os.path.join(base_dir, 'mask_prepared'),
                     dest_width=512,
                     dest_height=512,
                     to_black_white=True)
prepare_train_images(os.path.join(base_dir, 'image_source'),
                     os.path.join(base_dir, 'image_prepared'),
                     dest_width=512,
                     dest_height=512)
