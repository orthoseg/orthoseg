"""Convert a PyTorch smp resnet34+unet .pth weights file to a Keras saved model.

Approach: build the model using segmodels_keras (so the output is directly
compatible with orthoseg's weights_notop= loading), then map weights tensor-by-
tensor from the PyTorch state dict. No ONNX conversion is needed.

Note: Currently only supports resnet34 backbone with UNet decoder architecture.

Two output files are produced:

  - <name>.keras          — full model for inference
  - <name>_notop.weights.h5 — encoder+decoder weights for use as weights_notop=

You can create a conda environment with the needed dependencies in order to run this
script like this:

    conda env create -f helper_scripts/convert_pth2keras.yml
    conda activate pth2keras
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras

# Allow importing segmodels_keras from its source repo when not installed
# _SMK_SRC = Path(r"C:\Users\PIEROG\Projects\github\segmodels_keras")
# if _SMK_SRC.exists() and str(_SMK_SRC) not in sys.path:
#    sys.path.insert(0, str(_SMK_SRC))
import segmodels_keras as smk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions for weight mapping
# ---------------------------------------------------------------------------


def _to_np(tensor) -> np.ndarray:
    return tensor.detach().float().numpy()


def _set_conv(model: keras.Model, layer_name: str, pt_weight, pt_bias=None) -> None:
    """Set Conv2D weights, transpose from PyTorch [out,in,H,W] -> Keras [H,W,in,out]."""
    layer = model.get_layer(layer_name)
    kernel = np.transpose(_to_np(pt_weight), (2, 3, 1, 0))
    layer.set_weights([kernel] if pt_bias is None else [kernel, _to_np(pt_bias)])


def _set_bn(model: keras.Model, layer_name: str, sd: dict, pt_prefix: str) -> None:
    """Set BatchNormalization weights from a state dict prefix."""
    layer = model.get_layer(layer_name)
    layer.set_weights(
        [
            _to_np(sd[f"{pt_prefix}.weight"]),  # gamma
            _to_np(sd[f"{pt_prefix}.bias"]),  # beta
            _to_np(sd[f"{pt_prefix}.running_mean"]),  # moving_mean
            _to_np(sd[f"{pt_prefix}.running_var"]),  # moving_var
        ]
    )


# ---------------------------------------------------------------------------
# ResNet34+UNet weight mapping
# ---------------------------------------------------------------------------


def _map_resnet34_unet_weights(
    model: keras.Model,
    state_dict: dict,
    pt_prefix: str,
) -> None:
    """Map all weights from the PyTorch state dict onto the segmodels_keras model.

    This function is specific to resnet34+unet architecture.

    Mapping logic:
      - Encoder naming: PyTorch encoder.layer{L}.{B} -> Keras conv{L+1}_block{B+1}
      - Decoder naming: PyTorch decoder.blocks.{i}.conv{j}
          -> Keras decoder_stage{i}[a|b]
      - Head: PyTorch segmentation_head.0 -> Keras final_conv
    """
    sd = {
        k.removeprefix(pt_prefix): v
        for k, v in state_dict.items()
        if k.startswith(pt_prefix)
    }

    # Stem: conv1 + bn1
    _set_conv(model, "conv1_conv", sd["encoder.conv1.weight"])
    _set_bn(model, "conv1_bn", sd, "encoder.bn1")

    # Encoder stages: layer1-4 in PyTorch → conv2-5 in Keras
    # ResNet34 has 3, 4, 6, 3 blocks in layers 1-4 respectively
    layer_blocks = {1: 3, 2: 4, 3: 6, 4: 3}
    for L, n_blocks in layer_blocks.items():
        keras_stage = L + 1  # layer1 → conv2, layer2 → conv3, ...
        for B in range(n_blocks):
            keras_block = B + 1  # 0-indexed → 1-indexed
            pt = f"encoder.layer{L}.{B}"
            pfx = f"conv{keras_stage}_block{keras_block}_"

            _set_conv(model, f"{pfx}1_conv", sd[f"{pt}.conv1.weight"])
            _set_bn(model, f"{pfx}1_bn", sd, f"{pt}.bn1")
            _set_conv(model, f"{pfx}2_conv", sd[f"{pt}.conv2.weight"])
            _set_bn(model, f"{pfx}2_bn", sd, f"{pt}.bn2")

            # Shortcut/downsample (only on first block of each stage when dims change)
            if f"{pt}.downsample.0.weight" in sd:
                _set_conv(model, f"{pfx}0_conv", sd[f"{pt}.downsample.0.weight"])
                _set_bn(model, f"{pfx}0_bn", sd, f"{pt}.downsample.1")

    # Decoder: smp decoder.blocks.{i} → segmodels_keras decoder_stage{i}a/b
    for i in range(5):
        pt = f"decoder.blocks.{i}"
        _set_conv(model, f"decoder_stage{i}a_conv", sd[f"{pt}.conv1.0.weight"])
        _set_bn(model, f"decoder_stage{i}a_bn", sd, f"{pt}.conv1.1")
        _set_conv(model, f"decoder_stage{i}b_conv", sd[f"{pt}.conv2.0.weight"])
        _set_bn(model, f"decoder_stage{i}b_bn", sd, f"{pt}.conv2.1")

    # Segmentation head
    _set_conv(
        model,
        "final_conv",
        sd["segmentation_head.0.weight"],
        sd["segmentation_head.0.bias"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_pt_to_keras(
    pth_path: Path,
    encoder: str = "resnet34",
    decoder: str = "unet",
    nb_channels: int = 5,
    nb_classes: int = 19,
    activation: str = "softmax",
    state_dict_prefix: str = "model.seg_model.",
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Convert a PyTorch smp resnet34+unet .pth to segmodels_keras-compatible files.

    Args:
        pth_path: Path to the .pth weights file from a resnet34+unet model.
        encoder: Name of the encoder backbone architecture. Must be "resnet34".
            Defaults to "resnet34".
        decoder: Name of the decoder architecture. Must be "unet".
            Defaults to "unet".
        nb_channels: Number of input channels in the source PyTorch model.
        nb_classes: Number of output classes.
        activation: Output activation function (e.g. "softmax" or "sigmoid").
        state_dict_prefix: Prefix to strip from the state dict keys.
        overwrite: If ``True``, overwrite existing output files. Defaults to
            ``False``.

    Returns:
        Tuple of (keras_path, weights_path, weights_notop_path):
        - keras_path: full model saved as .keras (for inference).
        - weights_path: full model weights saved as .weights.h5 (for inference).
        - weights_notop_path: encoder+decoder weights saved as _notop.weights.h5
          (for use as weights_notop= when initializing a new model).

    Raises:
        ValueError: If encoder is not "resnet34" or if decoder is not "unet".
    """
    # Validate encoder and decoder
    if encoder != "resnet34":
        raise ValueError(
            f"Encoder '{encoder}' is not supported. "
            "Currently, only 'resnet34' encoder is supported for weight conversion."
        )
    if decoder != "unet":
        raise ValueError(
            f"Decoder '{decoder}' is not supported. "
            "Currently, only 'unet' decoder is supported for weight conversion."
        )

    pth_path = Path(pth_path)

    keras_path = pth_path.with_suffix(".keras")
    weights_path = pth_path.with_name(f"{pth_path.stem}.weights.h5")
    weights_notop_path = pth_path.with_name(f"{pth_path.stem}_notop.weights.h5")

    if (
        not overwrite
        and keras_path.exists()
        and weights_path.exists()
        and weights_notop_path.exists()
    ):
        logger.info(
            "Output files already exist, skipping. Use overwrite=True to overwrite."
        )
        return keras_path, weights_path, weights_notop_path

    logger.info(f"Building segmodels_keras {encoder}+{decoder} model...")
    model = smk.Unet(
        backbone_name=encoder,
        input_shape=(None, None, nb_channels),
        classes=nb_classes,
        activation=activation,
        encoder_weights=None,
    )

    logger.info(f"Loading PyTorch weights from: {pth_path}")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)

    logger.info("Mapping weights to Keras model...")
    _map_resnet34_unet_weights(model, state_dict, state_dict_prefix)

    if overwrite or not keras_path.exists():
        logger.info(f"Saving full Keras model to: {keras_path}")
        model.save(str(keras_path))

    if overwrite or not weights_path.exists():
        logger.info(f"Saving weights to: {weights_path}")
        model.save_weights(str(weights_path))

    if overwrite or not weights_notop_path.exists():
        # Build the notop sub-model (everything except final_conv and activation),
        # matching the model that build_unet loads weights_notop into.
        notop_output = model.get_layer("decoder_stage4b_relu").output
        notop_model = keras.Model(model.input, notop_output)
        logger.info(f"Saving notop weights to: {weights_notop_path}")
        notop_model.save_weights(str(weights_notop_path))

    logger.info("Done.")
    return keras_path, weights_path, weights_notop_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nb_channels = 3
    encoder = "resnet34"
    decoder = "unet"

    keras_path, weights_path, weights_notop_path = convert_pt_to_keras(
        pth_path=Path(
            r"X:\Monitoring\OrthoSeg\landcover15\FLAIR"
            r"\FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth"
        ),
        encoder=encoder,
        decoder=decoder,
        nb_channels=nb_channels,
        # The segmentation head has 19 output channels despite "15cl" in filename
        nb_classes=19,
        activation="softmax",
        state_dict_prefix="model.seg_model.",
        overwrite=True,
    )

    # Load the saved notop weights into a new model to verify compatibility.
    # Test loading the full model
    print(f"Test loading full model from: {keras_path}")
    model = keras.models.load_model(keras_path)
    assert model is not None
    model = None

    # Test loading the model weights
    print(f"Test loading model weights from: {weights_path}")
    model = smk.Unet(
        encoder,
        input_shape=(None, None, nb_channels),
        classes=19,
        activation="softmax",
        encoder_weights=None,
        weights=weights_path,
    )
    assert model is not None
    model = None

    # Test loading the model weights without the top layers
    print(
        f"Test loading model weights without the top layers from: {weights_notop_path}"
    )
    model = smk.Unet(
        encoder,
        input_shape=(None, None, nb_channels),
        classes=19,
        activation="softmax",
        encoder_weights=None,
        weights_notop=weights_notop_path,
        freeze_notop=True,
    )
    assert model is not None
    model = None
