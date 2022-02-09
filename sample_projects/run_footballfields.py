# -*- coding: utf-8 -*-
"""
Tests for functionalities in orthoseg.train.
"""

from pathlib import Path
import sys

# Add path so the local orthoseg packages are found 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import orthoseg

if __name__ == '__main__':
    project_config_path = Path("~/orthoseg/sample_projects/footballfields/footballfields.ini")
    orthoseg.load_images(project_config_path)
    orthoseg.train(project_config_path)
    orthoseg.predict(project_config_path)
    orthoseg.postprocess(project_config_path)
