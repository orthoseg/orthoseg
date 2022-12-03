# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.train.
"""

from pathlib import Path
import sys

# Add path so the local orthoseg packages are found
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import orthoseg

if __name__ == "__main__":
    project_dir = Path("~/orthoseg/sample_projects/footballfields")
    config_train_path = "footballfields_train.ini"
    config_predictBE2019_path = "footballfields_BEFL-2019.ini"

    orthoseg.load_images(config_predictBE2019_path)
    orthoseg.train(config_train_path)
    orthoseg.predict(config_predictBE2019_path)
    orthoseg.postprocess(config_predictBE2019_path)
