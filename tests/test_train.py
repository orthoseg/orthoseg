import datetime
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

from orthoseg import train

def get_testdata_dir() -> Path:
    return Path(__file__).resolve().parent / 'data'

def test_search_label_files():
    labeldata_template = get_testdata_dir() / 'footballfields_{image_layer}_data.gpkg'
    labellocation_template = get_testdata_dir() / 'footballfields_{image_layer}_locations.gpkg'
    result = train.search_label_files(labeldata_template, labellocation_template)

    assert len(result) == 2

if __name__ == '__main__':
    test_search_label_files()