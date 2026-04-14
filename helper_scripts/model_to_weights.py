"""Convert a model in .hdf5 file format to weights."""

import logging
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disable using GPU
import keras

import orthoseg.model.model_factory as mf


def convert_model(model_path: Path, weights_dir: Path, include_top: bool = False):
    """Convert model from .hdf5 to .h5.

    Args:
        model_path (Path): Path to the hdf5 model.
        weights_dir (Path): Directory to save the weights.
    """
    # First load model metadata
    model_hyperparams = mf.load_model_hyperparams(model_path)

    # Try converting model
    if include_top:
        weights_path = weights_dir / f"{model_path.stem}.weights.h5"
    else:
        base_stem = model_hyperparams["architecture"]["architecture"]
        weights_path = weights_dir / f"{base_stem}_notop.weights.h5"

    if weights_path.exists():
        raise FileExistsError(f"Model weights already exist at {weights_path}.")

    # Load and save it in .h5 format.
    model, _ = mf.load_model(model_path, compile_model=False)
    if not include_top:
        # Remove the top layers of the model (i.e. the model head) before saving
        # weights.
        model = keras.Model(model.input, model.layers[-2].output)
    model.save_weights(weights_path)
    del model


# If the script is ran directly...
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_path = Path(
        r"X:\Monitoring\OrthoSeg\sealedsurfaces\models\sealedsurfaces_59.5.0_0.94113_214.keras"
    )
    weights_dir = Path(r"X:\Monitoring\OrthoSeg\_weights")
    convert_model(model_path, weights_dir=weights_dir, include_top=False)
