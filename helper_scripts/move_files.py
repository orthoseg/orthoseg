"""
Some functions to cleanup files ad-hoc...
"""

import os
from pathlib import Path


def movefiles(dir_src, dir_dest, filename_list_filepath):
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


def movefile_go():
    # ********************
    # GO
    # ********************
    base_dir = "X:\\PerPersoon\\PIEROG\\Taken\\2018\\2018-08-12_AutoSegmentation\\greenhouses\\train"

    # Move the "easy" files
    movefiles(
        os.path.join(base_dir, "image"),
        os.path.join(base_dir, "_traindata_removed\\easy_image"),
        os.path.join(base_dir, "train_easy_to_remove.txt"),
    )
    movefiles(
        os.path.join(base_dir, "mask"),
        os.path.join(base_dir, "_traindata_removed\\easy_mask"),
        os.path.join(base_dir, "train_easy_to_remove.txt"),
    )

    # Move the "error" files
    movefiles(
        os.path.join(base_dir, "image"),
        os.path.join(base_dir, "_traindata_removed\\error_image"),
        os.path.join(base_dir, "train_error_to_remove.txt"),
    )
    movefiles(
        os.path.join(base_dir, "mask"),
        os.path.join(base_dir, "_traindata_removed\\error_mask"),
        os.path.join(base_dir, "train_error_to_remove.txt"),
    )


def remove_wrong_files_from_cache(cachedir: Path):
    # Get list of all image files to process...
    cache_filepaths = cachedir.rglob("*.*")

    # Loop over all files
    nb_removed = 0
    nb_total = 0
    for cache_filepath in cache_filepaths:
        nb_total += 1
        # The cache filename should start with the name of the directory it is in...
        if not cache_filepath.name.startswith(cache_filepath.parent.name):
            print(f"Cache file {cache_filepath} shouldn't be here, so remove")
            cache_filepath.unlink()
            nb_removed += 1

    print(
        f"Cleaned {nb_removed} incorrect files of total {nb_total} files from the cache"
    )


def remove_wrong_files_from_cache_go():
    # cachedir = Path(r"\\dg3.be\alp\gistmp\orthoseg\_image_cache\BEFL-2020-ofw\3840x3840_128pxOverlap")
    # cachedir = Path(r"\\dg3.be\alp\gistmp\orthoseg\_image_cache\BEFL-2020-ofw\4000x4000_0pxOverlap")
    # cachedir = Path(r"\\dg3.be\alp\gistmp\orthoseg\_image_cache\BEFL-2018\2048x2048_128pxOverlap")
    # cachedir = Path(r"\\dg3.be\alp\gistmp\orthoseg\_image_cache\BEFL-2018-summer\2048x2048_128pxOverlap")
    cachedir = Path(
        r"\\dg3.be\alp\gistmp\orthoseg\_image_cache\BEFL-2015\2048x2048_128pxOverlap"
    )
    remove_wrong_files_from_cache(cachedir)


# If the script is ran directly...
if __name__ == "__main__":
    # movefile_go()
    remove_wrong_files_from_cache_go()
