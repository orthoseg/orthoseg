"""Download the sample project."""

import argparse
import logging
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from orthoseg._compat import __version__

# Get a logger...
logger = logging.getLogger(__name__)


def _parse_load_sampleprojects_args(args) -> dict:
    # Define supported arguments
    parser = argparse.ArgumentParser(add_help=False)

    help_str = (
        "The directory to create the sample_projects dir in. "
        "Eg. ~/orthoseg will create orthoseg/sample_projects in your home directory."
    )
    parser.add_argument("dest_dir", help=help_str)

    help_str = (
        "True to use the default certificate bundle as installed on your system. "
        "False disables certificate validation (NOT recommended!). In corporate "
        "networks using a proxy server it is often needed to specify a customized "
        "certificate bundle (.pem file) to avoid CERTIFICATE_VERIFY_FAILED errors. "
        "It is recommended to specify the path to a custom certificate bundle file "
        "using the REQUESTS_CA_BUNDLE environment variable, but it can also passed "
        "using this switch. Parameter defaults to True."
    )
    parser.add_argument("--ssl_verify", default=True, help=help_str)

    # Interprete arguments
    args = parser.parse_args(args)
    dest_dir = Path(args.dest_dir).expanduser() / "orthoseg"
    ssl_verify = args.ssl_verify
    if isinstance(args.ssl_verify, str):
        if args.ssl_verify.lower() == "false":
            ssl_verify = False
        elif args.ssl_verify.lower() == "true":
            ssl_verify = True

    # Return arguments
    return {"dest_dir": dest_dir, "ssl_verify": ssl_verify}


def load_sampleprojects(dest_dir: Path, ssl_verify: bool | str = True):  # noqa: ARG001
    """Load the orthoseg sample projects.

    Args:
        dest_dir (Path): directory to save them to.
        ssl_verify (bool or str, optional): True to use the default certificate bundle
            as installed on your system. False disables certificate validation
            (NOT recommended!). If a path to a certificate bundle file (.pem) is passed,
            this will be used. In corporate networks using a proxy server this is often
            needed to avoid CERTIFICATE_VERIFY_FAILED errors. Defaults to True.
    """
    dest_dir = dest_dir.expanduser()
    dest_dir_full = dest_dir / "sample_projects"
    if dest_dir_full.exists():
        raise ValueError(f"Destination directory already exists: {dest_dir_full}")

    # Download
    print(f"Start download of sample projects to {dest_dir_full!s}")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        # Download the sample projects to a temporary directory first
        base_url = "https://github.com/orthoseg/orthoseg/archive/refs/tags/v"
        url = f"{base_url}{__version__}.zip"
        tmp_zip_path = Path(tmp_dir) / "orthoseg_src.zip"
        urllib.request.urlretrieve(url, tmp_zip_path)

        # Unzip the downloaded file
        tmp_proj_dir = Path(tmp_dir) / "orthoseg_src_tmp"
        with zipfile.ZipFile(tmp_zip_path, "r") as zip_file:
            zip_file.extractall(tmp_proj_dir)

        # Unzipped folder contains a single directory with the orthoseg repo contents.
        subdirs = [d for d in tmp_proj_dir.iterdir() if d.is_dir()]
        if len(subdirs) != 1:
            raise ValueError(
                f"Expected exactly one directory in the zip, but found {len(subdirs)}"
            )

        # Move the "sample_projects" subdir to the destination directory
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(subdirs[0] / "sample_projects", dest_dir)

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
