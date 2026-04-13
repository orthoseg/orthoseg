"""Tests on the config_util module."""

from pathlib import Path

import pytest

from orthoseg.util import config_util


@pytest.mark.parametrize(
    "config_path, expected_error",
    [
        ("~/relative_path.ini", "Config file specified is not an absolute path"),
        ("./relative_path.ini", "Config file specified is not an absolute path"),
        ("relative/path/config.ini", "Config file specified is not an absolute path"),
        ("non_existent_configfile.ini", "Config file specified does not exist"),
        ("config_dir", "Config file specified is not a file"),
    ],
)
def test_get_config_files_error_handling(tmp_path, config_path, expected_error):
    if "is not a file" in expected_error:
        # Create a directory with the name of the config file to trigger the error
        config_path = tmp_path / config_path
        config_path.mkdir(parents=True, exist_ok=True)
    elif "does not exist" in expected_error:
        # Make sure the config file does not exist
        config_path = tmp_path / config_path

    config_path = Path(config_path)
    with pytest.raises(ValueError, match=expected_error):
        config_util.get_config_files(config_path)
