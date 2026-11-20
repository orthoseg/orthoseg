"""Tests for module load_sampleprojects."""

import filecmp
import io
import os
import shutil
import ssl
import urllib.request
import zipfile
from packaging.version import Version
from pathlib import Path
from types import SimpleNamespace

import pytest

from orthoseg import __version__, load_sampleprojects
from orthoseg.helpers import config_helper as conf
from tests import test_helper


def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


@pytest.mark.parametrize(
    "args, exp_ssl_verify",
    [
        (["C:/Monitoring/OrthoSeg/test"], True),
        (["C:/Monitoring/OrthoSeg/test", "--ssl_verify", "FaLsE"], False),
        (["C:/Monitoring/OrthoSeg/test", "--ssl_verify", "TrUe"], True),
        (["C:/Monitoring/OrthoSeg/test", "--ssl_verify", "abc"], "abc"),
    ],
)
def test_load_images_args(args, exp_ssl_verify):
    valid_args = load_sampleprojects._parse_load_sampleprojects_args(args=args)
    assert valid_args is not None
    assert valid_args["dest_dir"] is not None
    if isinstance(exp_ssl_verify, bool):
        assert valid_args["ssl_verify"] is exp_ssl_verify
    else:
        assert valid_args["ssl_verify"] == exp_ssl_verify


def test_load_sampleprojects(request, tmp_path):
    """Test loading the sample projects.

    The checked-in samples are tested in regular runs. When testing the installed
    package, also download the release archive and compare it with them.
    """
    testing_installed = os.environ.get("TESTS_INSTALLED", "FALSE") == "TRUE"
    if testing_installed:
        # Test using ~ as used in the documentation.
        dest_dir = Path("~/orthoseg_test")
        shutil.rmtree(dest_dir.expanduser(), ignore_errors=True)
        load_sampleprojects.load_sampleprojects(dest_dir=dest_dir, ssl_verify=False)
        sampleprojects_dir = dest_dir.expanduser() / "sample_projects"
    else:
        sampleprojects_dir = tmp_path / "sample_projects"
        shutil.copytree(test_helper.sampleprojects_dir, sampleprojects_dir)

    assert sampleprojects_dir.exists()
    assert (sampleprojects_dir / "imagelayers.ini").exists()
    assert (sampleprojects_dir / "project_defaults_overrule.ini").exists()
    assert (sampleprojects_dir / "run_sportsfields.py").exists()

    sportsfields_dir = sampleprojects_dir / "sportsfields"
    assert sportsfields_dir.exists()
    conf.read_orthoseg_config(sportsfields_dir / "sportsfields.ini")
    files = list(sportsfields_dir.glob("**/*.ini"))
    assert len(files) > 0
    files = list((sportsfields_dir / "labels").glob("**/*.gpkg"))
    assert len(files) > 0
    model_paths = list((sportsfields_dir / "models").glob("*.hdf5"))
    assert model_paths
    assert all(
        model_path.stat().st_size > 40 * 1024 * 1024 for model_path in model_paths
    )

    projecttemplate_dir = sampleprojects_dir / "project_template"
    assert projecttemplate_dir.exists()
    files = list(projecttemplate_dir.glob("**/*.ini"))
    assert len(files) > 0
    files = list((projecttemplate_dir / "labels").glob("**/*.gpkg"))
    assert len(files) > 0

    # Compare the loaded files with the current sampleprojects in the repository
    # For pre-release versions, xfail as the sampleprojects may still differ
    if testing_installed and Version(__version__).is_prerelease:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="prerelease: loaded sampleprojects may differ from repository"
            )
        )

    if testing_installed:
        # Downloaded models are not present in the repository, so ignore.
        _assert_dirs_equal(
            sampleprojects_dir,
            test_helper.sampleprojects_dir,
            ignore=["log", "models"],
            ignore_crlf_suffixes=[".ini", ".jgw", ".py", ".qml", ".txt", ".xml"],
        )


def test_ssl_verify_false_is_honored(tmp_path, monkeypatch):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("source/sample_projects/example.ini", "example")

    def fake_ssl_context(protocol):
        assert protocol == ssl.PROTOCOL_TLS_CLIENT
        return SimpleNamespace(
            check_hostname=True,
            verify_mode=ssl.CERT_REQUIRED,
        )

    def fake_urlopen(_url, context):
        assert context.verify_mode == ssl.CERT_NONE
        assert context.check_hostname is False
        return io.BytesIO(zip_buffer.getvalue())

    monkeypatch.setattr(
        load_sampleprojects,
        "ssl",
        SimpleNamespace(
            SSLContext=fake_ssl_context,
            PROTOCOL_TLS_CLIENT=ssl.PROTOCOL_TLS_CLIENT,
            CERT_NONE=ssl.CERT_NONE,
        ),
    )
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    dest_dir = tmp_path / "orthoseg"
    load_sampleprojects.load_sampleprojects(dest_dir=dest_dir, ssl_verify=False)

    assert (dest_dir / "sample_projects" / "example.ini").read_text() == "example"


def _assert_dirs_equal(
    dir1: Path,
    dir2: Path,
    ignore: list[str] | None = None,
    ignore_crlf_suffixes: list[str] | None = None,
):
    """Compare two directories recursively.

    Files in each directory are assumed to be equal if their names and contents are
    equal.

    Inspired by: https://stackoverflow.com/a/6681395/10096091

    Args:
        dir1: First directory path
        dir2: Second directory path
        ignore: List of file or directory names to ignore
        ignore_crlf_suffixes: List of file name suffixes for which to ignore differences
        in line endings (CRLF vs LF)
    """
    dirs_cmp = filecmp.dircmp(dir1, dir2, ignore=ignore)
    if len(dirs_cmp.left_only) > 0:
        raise AssertionError(
            f"{dir1} contains files not found in {dir2}: {dirs_cmp.left_only}"
        )
    elif len(dirs_cmp.right_only) > 0:
        raise AssertionError(
            f"{dir2} contains files not found in {dir1}: {dirs_cmp.right_only}"
        )
    elif len(dirs_cmp.funny_files) > 0:
        raise AssertionError(
            f"files found in {dir1} and {dir2} that could not be compared: "
            f"{dirs_cmp.funny_files}"
        )

    def cmpfiles_no_crlf(path_1: Path, path_2: Path):
        l1 = l2 = "1"
        with path_1.open("r") as f1, path_2.open("r") as f2:
            while l1 and l2:
                l1 = f1.readline()
                l2 = f2.readline()
                if l1 != l2:
                    return False
        return True

    for name in dirs_cmp.common_files:
        path1 = dir1 / name
        path2 = dir2 / name
        if ignore_crlf_suffixes is not None and path1.suffix in ignore_crlf_suffixes:
            if not cmpfiles_no_crlf(path1, path2):
                raise AssertionError(f"Files {path1} and {path2} do not match")
        else:
            if not filecmp.cmp(path1, path2, shallow=False):
                raise AssertionError(f"Files {path1} and {path2} do not match")

    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = dir1 / common_dir
        new_dir2 = dir2 / common_dir
        _assert_dirs_equal(
            new_dir1, new_dir2, ignore=ignore, ignore_crlf_suffixes=ignore_crlf_suffixes
        )
