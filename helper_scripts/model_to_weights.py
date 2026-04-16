"""Convert a model to a weights file."""

import logging
from pathlib import Path

import segmodels_keras as smk

import orthoseg.model.model_factory as mf


def convert_model(
    model_path: Path,
    weights_dir: Path,
    weight_type: str,
    include_top: bool = False,
    overwrite: bool = True,
):
    """Convert a model to a weights file.

    Args:
        model_path (Path): Path to the hdf5 model.
        weights_dir (Path): Directory to save the weights.
        weight_type (str): Type of weights to save.
        include_top (bool, optional): Whether to include the top layers of the model.
            Defaults to False.
        overwrite (bool, optional): Whether to overwrite existing weights.
            Defaults to True.
    """
    # Load and save it in .h5 format.
    model, _ = mf.load_model(model_path, compile_model=False)
    if include_top:
        weights_path = weights_dir / f"n{model_path.stem}.weights.h5"
        model.save_weights(weights_path, overwrite=overwrite)
    else:
        # Save the weights without top layers for transfer learning.
        model_hyperparams = mf.load_model_hyperparams(model_path)
        architecture = model_hyperparams["architecture"]["architecture"]
        weights_path = weights_dir / f"{architecture}_{weight_type}_notop.weights.h5"
        _encoder, decoder = architecture.split("+")
        smk.utils.save_model_weights_notop(
            model, decoder=decoder, path=weights_path, overwrite=overwrite
        )


# If the script is ran directly...
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os_dir = Path(r"X:\Monitoring\OrthoSeg")
    # mobilenetv2+linknet model
    # model_path = model_dir / "sealedsurfaces_59.5.0_0.94113_214.keras"
    # inceptionresnetv2+unet model
    model_path = (
        os_dir / "recreationfields/models/recreationfields_34.2.0_0.88137_79.keras"
    )

    weights_dir = os_dir / "_weights"
    convert_model(
        model_path,
        weights_dir=weights_dir,
        weight_type="aerial",
        include_top=False,
        overwrite=False,
    )
