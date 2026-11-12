"""Tests for helper scripts."""

import shutil
from pathlib import Path

import geofileops as gfo
import pytest

from helper_scripts.run_convert_output_vector_filenames import (
    convert_output_vector_filenames_project,
    run_convert_output_vector_filenames,
)
from tests import test_helper


def test_run_convert_output_vector_filenames(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    # Setup test projects dir
    projects_dir = tmp_path / "sample_projects"
    shutil.copytree(test_helper.sampleprojects_dir, projects_dir)
    project_dir = projects_dir / test_helper.SportsFields.subject

    # In the sample project, output_vector already has old-style predict file:
    # sportsfields_01_276_BEFL-2025-sportsfields_predict.gpkg
    vector_dir = project_dir / "output_vector" / "BEFL-2025-sportsfields"
    old_predict = vector_dir / "sportsfields_01_276_BEFL-2025-sportsfields_predict.gpkg"
    assert old_predict.exists()

    # Also create an old-style postprocess file
    old_postp = vector_dir / "sportsfields_01_276_BEFL-2025-sportsfields.gpkg"
    shutil.copy(old_predict, old_postp)
    gfo.rename_layer(old_postp, old_postp.stem)
    assert gfo.get_only_layer(old_postp) == old_postp.stem

    # If configuration is not changed, no files should be renamed
    renamed_unconfigured = run_convert_output_vector_filenames(
        projects_dir=projects_dir, simulate=False
    )
    assert renamed_unconfigured == {}
    assert old_predict.exists()
    assert old_postp.exists()

    # Now configure custom / future default output_vector names in project config
    config_file = project_dir / f"{project_dir.name}.ini"
    with config_file.open("a") as f:
        f.write(
            "\n[files]\n"
            "output_vector_predict_name = "
            "{segment_subject}_{image_layer}_{model_version}_predict.gpkg\n"
            "output_vector_postp_name = "
            "{segment_subject}_{image_layer}_{model_version}.gpkg\n"
        )

    # Test simulate=True with configured names
    simulated = run_convert_output_vector_filenames(
        projects_dir=projects_dir, simulate=True
    )
    assert project_dir in simulated
    assert len(simulated[project_dir]) >= 2
    assert old_predict.exists()
    assert old_postp.exists()

    # Test simulate=False with configured names
    renamed = run_convert_output_vector_filenames(
        projects_dir=projects_dir, simulate=False
    )
    assert project_dir in renamed
    expected_predict = (
        vector_dir / "sportsfields_BEFL-2025-sportsfields_01_predict.gpkg"
    )
    expected_postp = vector_dir / "sportsfields_BEFL-2025-sportsfields_01.gpkg"

    assert not old_predict.exists()
    assert not old_postp.exists()
    assert expected_predict.exists()
    assert expected_postp.exists()
    assert gfo.get_only_layer(expected_postp) == expected_postp.stem

    # Running again should find no files to rename
    rerun = run_convert_output_vector_filenames(
        projects_dir=projects_dir, simulate=False
    )
    assert rerun == {}

    # Verify that the has_switched_axes warning and config_helper messages
    # were filtered out
    assert not any(
        "has_switched_axes option not found" in record.message
        for record in caplog.records
    )
    assert not any(
        record.name.startswith("orthoseg.helpers.config_helper")
        for record in caplog.records
    )


def test_convert_output_vector_filenames_empty_placeholder_warning(tmp_path: Path):
    projects_dir = tmp_path / "sample_projects"
    shutil.copytree(test_helper.sampleprojects_dir, projects_dir)
    project_dir = projects_dir / test_helper.SportsFields.subject
    vector_dir = project_dir / "output_vector" / "BEFL-2025-sportsfields"

    pred_file = vector_dir / "sportsfields_01_276_BEFL-2025-sportsfields_predict.gpkg"
    assert pred_file.exists()

    # Configure template with a placeholder that is empty for this file
    config_file = project_dir / f"{project_dir.name}.ini"
    with config_file.open("a") as f:
        f.write(
            "\n[files]\n"
            "output_vector_predict_name = "
            "{segment_subject}_{image_layer}_{model_config_name}_predict.gpkg\n"
        )

    with pytest.warns(UserWarning, match=r"placeholder\(s\) model_config_name empty"):
        renamed = convert_output_vector_filenames_project(
            project_dir=project_dir, simulate=False
        )
    assert renamed == []
    assert pred_file.exists()


def test_convert_output_vector_filenames_project_no_config(tmp_path: Path):
    empty_dir = tmp_path / "empty_project"
    empty_dir.mkdir()
    result = convert_output_vector_filenames_project(empty_dir)
    assert result == []
