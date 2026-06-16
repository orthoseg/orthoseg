"""Tests for functionalities in orthoseg.train."""

from pathlib import Path

import orthoseg

if __name__ == "__main__":
    project_dir = Path("~/orthoseg/sample_projects/sportsfields")
    config_path = project_dir / "sportsfields_train.ini"

    orthoseg.load_images(config_path)
    orthoseg.train(config_path)
    orthoseg.predict(config_path)
    orthoseg.postprocess(config_path)
