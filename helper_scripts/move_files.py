"""Some functions to cleanup files ad-hoc..."""

from pathlib import Path


def movefiles(src_dir: Path, dest_dir: Path, filename_list_path: Path):
    """Move files from dir_src to dir_dest.

    The files to move are specified in a text file (filename_list_path).
    """
    # Read the filenames from file
    filenames = []
    with filename_list_path.open() as file:
        # Read all filenames in the file
        filenames = file.read().splitlines()
        # Make sure there are no doubles...
        filenames = list(set(filenames))
        # Remove empty filenames...
        filenames = list(filter(None, filenames))

    # Move all files
    for filename in filenames:
        filepath_src = src_dir / filename
        filepath_dest = dest_dir / filename
        print(f"Move file from {filepath_src} to {filepath_dest}")
        if filepath_src.exists():
            filepath_src.rename(filepath_dest)
        else:
            print(f"Source file doesn't exist: {filepath_src}")


def movefile_go():
    """Main entrypoint to start moving files."""
    base_dir = Path(
        r"X:\PerPersoon\PIEROG\Taken\2018\2018-08-12_AutoSegmentation\greenhouses\train"
    )

    # Move the "easy" files
    movefiles(
        base_dir / "image",
        base_dir / "_traindata_removed" / "easy_image",
        base_dir / "train_easy_to_remove.txt",
    )
    movefiles(
        base_dir / "mask",
        base_dir / "_traindata_removed" / "easy_mask",
        base_dir / "train_easy_to_remove.txt",
    )

    # Move the "error" files
    movefiles(
        base_dir / "image",
        base_dir / "_traindata_removed" / "error_image",
        base_dir / "train_error_to_remove.txt",
    )
    movefiles(
        base_dir / "mask",
        base_dir / "_traindata_removed" / "error_mask",
        base_dir / "train_error_to_remove.txt",
    )


def remove_wrong_files_from_cache(cachedir: Path):
    """Remove files from the cache that shouldn't be there."""
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
    """Remove files from the cache.."""
    cachedir = Path(
        r"\\dg3.be\alp\gistmp\orthoseg\_image_cache\BEFL-2015\2048x2048_128pxOverlap"
    )
    remove_wrong_files_from_cache(cachedir)


# If the script is ran directly...
if __name__ == "__main__":
    # movefile_go()
    remove_wrong_files_from_cache_go()
