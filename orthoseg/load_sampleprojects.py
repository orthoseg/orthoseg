"""Download the sample project."""

import argparse
import logging
import sys
from pathlib import Path

import gdown

from orthoseg.util import git_downloader

# Get a logger...
logger = logging.getLogger(__name__)


def _parse_load_sampleprojects_args(args) -> dict:
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


def load_sampleprojects(dest_dir: Path, ssl_verify: bool | None = None):
    """Load the orthoseg sample projects.

    Args:
        dest_dir (Path): directory to save them to.
        ssl_verify (Optional[bool], optional): True or None to use the default
            certificate bundle as installed on your system. False disables certificate
            validation (NOT recommended!). If a path to a certificate bundle file (.pem)
            is passed, this will be used. In corporate networks using a proxy server
            this is often needed to evade CERTIFICATE_VERIFY_FAILED errors.
            Defaults to None.
    """
    dest_dir_full = dest_dir / "sample_projects"
    if dest_dir_full.exists():
        raise ValueError(f"Destination directory already exists: {dest_dir_full}")

    # Download
    print(f"Start download of sample projects to {dest_dir_full!s}")
    git_downloader.download(
        repo_url="https://github.com/orthoseg/orthoseg/tree/main/sample_projects",
        output_dir=dest_dir,
        ssl_verify=ssl_verify,
    )
    print("Download finished")
    print("Start download of footballfields pretrained neural net")
    verify = True if ssl_verify is None else ssl_verify
    footballfields_model_dir = dest_dir_full / "footballfields/models"
    footballfields_model_dir.mkdir(parents=True, exist_ok=True)
    model_hdf5_path = footballfields_model_dir / "footballfields_01_0.92512_242.hdf5"
    if model_hdf5_path.exists() is False:
        gdown.download(
            id="1XmAenCW6K_RVwqC6xbkapJ5ws-f7-QgH",
            output=str(model_hdf5_path),
            verify=verify,
        )
    model_hyperparams_path = (
        footballfields_model_dir / "footballfields_01_hyperparams.json"
    )
    if model_hyperparams_path.exists() is False:
        gdown.download(
            id="1umxcd4RkB81sem9PdIpLoWeiIW8ga1u7",
            output=str(model_hyperparams_path),
            verify=verify,
        )
    model_modeljson_path = footballfields_model_dir / "footballfields_01_model.json"
    if model_modeljson_path.exists() is False:
        gdown.download(
            id="16qe8thBTrO3dFfLMU1T22gWcfHVXt8zQ",
            output=str(model_modeljson_path),
            verify=verify,
        )
    print("Download finished")


def main():
    """Run load sampleprojects."""
    try:
        parsed_args = _parse_load_sampleprojects_args(sys.argv[1:])
        load_sampleprojects(**parsed_args)
    except Exception as ex:
        logger.exception(f"Error: {ex}")
        raise


if __name__ == "__main__":
    main()
