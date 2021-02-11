# -*- coding: utf-8 -*-
"""
Convert models in hdf5 file format to tf savedmodel.
"""

from pathlib import Path
import sys 

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable using GPU
import tensorflow as tf

# Add parent.parent dir to find util dir
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
import orthoseg.model.model_factory as mf

def convert_model(
        model_path: Path):
    """
    Convert model from hdf5 to tf.

    Args:
        model_path (Path): Path to the hdf5 model.
    """
    # Try converting model
    savedmodel_dir = model_path.parent / f"{model_path.stem}_tf"
    if not savedmodel_dir.exists():
        # If base model not yet in savedmodel format
        model = mf.load_model(model_path, compile=False)
        tf.saved_model.save(model, str(savedmodel_dir))
        del model

# If the script is ran directly...
if __name__ == '__main__':
    mode_path = Path(r"X:\Monitoring\OrthoSeg\greenhouses2\models\greenhouses2_04_0.87063_495.hdf5") 
    convert_model(mode_path)
    