"""
Convert models in .hdf5 file format to .keras.
"""
import logging
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
    if not keras.__version__.startswith("3."):
        raise RuntimeError(
            "Keras version 3.x is required to convert models to .keras format."
        )

    # Try converting model
    keras_path = model_path.parent / f"{model_path.stem}.keras"
    if not keras_path.exists():
        # If model not yet in .keras format, load and save it in .keras format.
        model = mf.load_model(model_path, compile_model=False)
        model.save(keras_path)
        del model


# If the script is ran directly...
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mode_path = Path(
        r"X:\Monitoring\OrthoSeg\sealedsurfaces\models\sealedsurfaces_58_0.97395_404.hdf5"
    )
    convert_model(mode_path)
