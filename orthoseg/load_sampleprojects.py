# -*- coding: utf-8 -*-
"""
Download the sample project.
"""

import argparse
import logging
from pathlib import Path
import shlex
import sys
from typing import Optional

import gdown

# orthoseg is higher in dir hierarchy, add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orthoseg.util import git_downloader


# -------------------------------------------------------------
# First define/init general variables/constants
# -------------------------------------------------------------
# Get a logger...
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# The real work
# -------------------------------------------------------------


def parse_load_sampleprojects_argstr(argstr):
    args = shlex.split(argstr)
    parse_load_sampleprojects_args(args)


def parse_load_sampleprojects_args(args) -> dict:
    # Define supported arguments
    parser = argparse.ArgumentParser(add_help=False)

    help = (
        "The directory to create the sample_projects dir in. "
        "Eg. ~/orthoseg will create orthoseg/sample_projects in your home directory."
    )
    parser.add_argument("dest_dir", help=help)

    help = (
        "True to use the default certificate bundle as installed on your system. "
        "False disables certificate validation (NOT recommended!). In corporate "
        "networks using a proxy server it is often needed to specify a customized "
        "certificate bundle (.pem file) to avoid CERTIFICATE_VERIFY_FAILED errors. "
        "It is recommended to specify the path to a custum certificate bundle file "
        "using the REQUESTS_CA_BUNDLE environment variable, but it can also passed "
        "using this switch. Parameter defaults to True."
    )
    parser.add_argument("--ssl_verify", default=True, help=help)

    # Interprete arguments
    args = parser.parse_args(args)
    dest_dir = Path(args.dest_dir).expanduser() / "orthoseg"
    ssl_verify = args.ssl_verify

    # Return arguments
    return {"dest_dir": dest_dir, "ssl_verify": ssl_verify}


def load_sampleprojects(dest_dir: Path, ssl_verify: Optional[bool] = None):
    dest_dir_full = dest_dir / "sample_projects"
    if dest_dir_full.exists():
        raise Exception(f"Destination directory already exists: {dest_dir_full}")

    # Download
    print(f"Start download of sample projects to {str(dest_dir_full)}")
    git_downloader.download(
        repo_url="https://github.com/orthoseg/orthoseg/tree/master/sample_projects",
        output_dir=dest_dir,
        ssl_verify=ssl_verify,
    )
    print("Download finished")
    print("Start download of footballfields pretrained neural net")
    verify = True if ssl_verify is None else ssl_verify
    model_dir = dest_dir_full / "footballfields/models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_hdf5_path = model_dir / "footballfields_01_0.97392_201.hdf5"
    if not model_hdf5_path.exists():
        gdown.download(
            id="1UlNorZ74ADCr3pL4MCJ_tnKRNoeZX79g",
            output=str(model_hdf5_path),
            verify=verify,
        )
    model_hyperparams_path = model_dir / "footballfields_01_hyperparams.json"
    if not model_hyperparams_path.exists():
        gdown.download(
            id="1NwrVVjx9IsjvaioQ4-bkPMrq7S6HeWIo",
            output=str(model_hyperparams_path),
            verify=verify,
        )
    model_modeljson_path = model_dir / "footballfields_01_model.json"
    if not model_modeljson_path.exists():
        gdown.download(
            id="1LNPLypM5in3aZngBKK_U4Si47Oe97ZWN",
            output=str(model_modeljson_path),
            verify=verify,
        )
    print("Download finished")


def main():
    try:
        parsed_args = parse_load_sampleprojects_args(sys.argv[1:])
        load_sampleprojects(**parsed_args)
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


if __name__ == "__main__":
    main()
