"""Convert a model to a weights file."""

import logging
from pathlib import Path

import segmodels_keras as smk

import orthoseg.model.model_factory as mf


def convert_model(
    model_path: Path,
    weights_dir: Path,
    weight_type: str,
    include_top: bool,
    version: int,
    overwrite: bool = True,
):
    """Convert a model to a weights file.

    Args:
        model_path (Path): Path to the hdf5 model.
        weights_dir (Path): Directory to save the weights.
        weight_type (str): Type of weights to save.
        include_top (bool, optional): Whether to include the top layers of the model.
            Defaults to False.
        version (int): Version of the weights.
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
        weights_path = (
            weights_dir
            / f"{architecture}_{weight_type}-v{version}_notop_keras2.weights.h5"
        )
        _encoder, decoder = architecture.split("+")
        smk.utils.save_model_weights_notop(
            model, decoder=decoder, path=weights_path, overwrite=overwrite
        )


# If the script is ran directly...
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os_dir = Path(r"X:\Monitoring\OrthoSeg")

    # inceptionresnetv2+unet models
    # rpath = "sportsfields-sample/models/sportsfields-sample_03.4.0_0.88187_169.keras"
    # rpath = "recreationfields/models/recreationfields_34.2.0_0.92718_143.hdf5"
    # rpath = "hedges/models/hedges_13.1.0_0.71852_110.hdf5"
    # rpath = "greenhouses2/models/greenhouses2_26.2.0_0.95529_180.hdf5"

    # mobilenetv2+linknet model
    # rpath = "sealedsurfaces/models/sealedsurfaces_59.5.0_0.94113_214.keras"
    # rpath = "recreationfields/models/recreationfields_34.2.0_0.88137_79.keras"
    # rpath = "greenhouses2/models/greenhouses2_26.2.0_0.82914_112.keras"
    rpath = "greenhouses2/models/greenhouses2_25_0.96655_124.hdf5"

    model_path = os_dir / rpath

    weights_dir = os_dir / "_weights"
    convert_model(
        model_path,
        weights_dir=weights_dir,
        weight_type="aerial",
        version=1,
        include_top=False,
        overwrite=False,
    )
