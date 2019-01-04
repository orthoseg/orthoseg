# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:26:21 2018

@author: pierog
"""

import os

#********************
# FUNTIONS
#********************
def movefiles(dir_src,
              dir_dest,
              filename_list_filepath):
    
    # Read the filenames from file
    filenames = []
    with open(filename_list_filepath, "r") as file:
        # Read all filenames in the file
        filenames = file.read().splitlines()
        # Make sure there are no doubles...
        filenames = list(set(filenames))
        # Remove empty filenames...
        filenames = list(filter(None, filenames))

    # Move all files
    for filename in filenames:
        filepath_src = os.path.join(dir_src, filename)
        filepath_dest = os.path.join(dir_dest, filename)
        print(f"Move file from {filepath_src} to {filepath_dest}")
        if os.path.exists(filepath_src):
            os.rename(filepath_src, filepath_dest)
        else:
            print(f"Source file doesn't exist: {filepath_src}")

#********************
# GO
#********************   
base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation\\greenhouses\\train"

# Move the "easy" files
movefiles(os.path.join(base_dir, 'image'),
          os.path.join(base_dir, '_traindata_removed\\easy_image'),
          os.path.join(base_dir, "train_easy_to_remove.txt"))
movefiles(os.path.join(base_dir, 'mask'),
          os.path.join(base_dir, '_traindata_removed\\easy_mask'),
          os.path.join(base_dir, "train_easy_to_remove.txt"))

# Move the "error" files
movefiles(os.path.join(base_dir, 'image'),
          os.path.join(base_dir, '_traindata_removed\\error_image'),
          os.path.join(base_dir, "train_error_to_remove.txt"))
movefiles(os.path.join(base_dir, 'mask'),
          os.path.join(base_dir, '_traindata_removed\\error_mask'),
          os.path.join(base_dir, "train_error_to_remove.txt"))
