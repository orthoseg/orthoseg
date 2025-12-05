"""
Convert models in .hdf5 file format to .keras.
"""
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable using GPU
import keras

import orthoseg.model.model_factory as mf


def convert_model(model_path: Path):
    """
    Convert model from .hdf5 to .keras.

    Args:
        model_path (Path): Path to the hdf5 model.
    """
    # Try converting model
    keras_path = model_path.parent / f"{model_path.stem}.keras"
    if not keras_path.exists():
        # If base model not yet in .keras format
        model = mf.load_model(model_path, compile_model=False)
        model.save(keras_path)
        del model


# If the script is ran directly...
if __name__ == "__main__":
    mode_path = Path(
        r"X:\Monitoring\OrthoSeg\greenhouses2\models\greenhouses2_24_0.96385_17.hdf5"
    )
    convert_model(mode_path)
