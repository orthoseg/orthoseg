# -*- coding: utf-8 -*-
"""
Download the sample project.
"""

import argparse
from pathlib import Path
import sys

# orthoseg is higher in dir hierarchy, add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg.util import gdrive_util
from orthoseg.util import git_downloader


def main():

    # Interprete arguments
    parser = argparse.ArgumentParser(add_help=False)

    help = (
        "The directory to download the orthoseg sample projects to. "
        "Use ~ for your home directory."
    )
    parser.add_argument(
        "dest_dir",
        help=help,
    )
    args = parser.parse_args()
    dest_dir = Path(args.dest_dir).expanduser() / "orthoseg"

    dest_dir_full = dest_dir / "sample_projects"
    if dest_dir_full.exists():
        raise Exception(f"Destination directory already exists: {dest_dir_full}")

    # Download
    print(f"Start download of sample projects to {str(dest_dir_full)}")
    git_downloader.download(
        repo_url="https://github.com/orthoseg/orthoseg/tree/master/sample_projects",
        output_dir=dest_dir,
    )
    print("Download finished")
    print("Start download of footballfields pretrained neural net")
    footballfields_model_dir = dest_dir_full / "footballfields/models"
    model_hdf5_path = footballfields_model_dir / "footballfields_01_0.92512_242.hdf5"
    if model_hdf5_path.exists() is False:
        gdrive_util.download_file("1XmAenCW6K_RVwqC6xbkapJ5ws-f7-QgH", model_hdf5_path)
    model_hyperparams_path = (
        footballfields_model_dir / "footballfields_01_hyperparams.json"
    )
    if model_hyperparams_path.exists() is False:
        gdrive_util.download_file(
            "1umxcd4RkB81sem9PdIpLoWeiIW8ga1u7", model_hyperparams_path
        )
    model_modeljson_path = footballfields_model_dir / "footballfields_01_model.json"
    if model_modeljson_path.exists() is False:
        gdrive_util.download_file(
            "16qe8thBTrO3dFfLMU1T22gWcfHVXt8zQ", model_modeljson_path
        )
    print("Download finished")


if __name__ == "__main__":
    main()
