#!/usr/bin/env python3
"""Detect water with sat-water and draw water-object boundaries in blue."""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

# sat-water 0.1.0 defaults to legacy Keras, while its published .weights.h5
# files use the Keras 3 weight format. Select Keras 3 before importing TF.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")
os.environ.setdefault("SM_FRAMEWORK", "keras")

import numpy as np
from PIL import Image, ImageFilter


def water_mask(image: Image.Image, model_name: str) -> Image.Image:
    """Run sat-water and return a full-resolution binary (L mode) mask."""
    try:
        import keras
        import tensorflow as tf
        # segmentation-models 1.0.1 expects this pre-Keras-3 compatibility name.
        if not hasattr(keras.utils, "generic_utils"):
            keras.utils.generic_utils = types.SimpleNamespace(  # type: ignore[attr-defined]
                get_custom_objects=keras.utils.get_custom_objects
            )
        import segmentation_models as sm
        from satwater.weights import get_weights_path
    except ImportError as exc:
        raise RuntimeError(
            "sat-water with TensorFlow support is required. Install it with "
            "`python -m pip install \"sat-water[tf]\"`."
        ) from exc

    size = 512 if model_name == "resnet34_512" else 256
    model = sm.Unet(
        "resnet34",
        classes=2,
        activation="softmax",
        encoder_weights=None,
        input_shape=(size, size, 3),
    )
    model.load_weights(get_weights_path(model_name))

    # This mirrors satwater.inference.Inference preprocessing and prediction.
    values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    values = tf.image.resize(values / 255.0, (size, size), method="bilinear")
    values = sm.get_preprocessing("resnet34")(values)
    values = (values - tf.reduce_min(values)) / (
        tf.reduce_max(values) - tf.reduce_min(values) + 1e-8
    )
    prediction = np.argmax(model.predict(values[None, ...], verbose=0), axis=-1)[0]
    if model_name == "resnet34_512":
        prediction = 1 - prediction
    if prediction.ndim != 2:
        raise RuntimeError(f"Unexpected sat-water mask shape: {prediction.shape}")

    mask = Image.fromarray((prediction > 0).astype(np.uint8) * 255, mode="L")
    # Class labels must never be interpolated: NEAREST preserves the binary mask.
    return mask.resize(image.size, Image.Resampling.NEAREST)


def mask_boundary(mask: Image.Image, width: int, image: Image.Image) -> Image.Image:
    """Return a morphological gradient around every connected water region."""
    if width < 1:
        raise ValueError("--line-width must be at least 1")
    kernel = 2 * width + 1
    dilated = np.asarray(mask.filter(ImageFilter.MaxFilter(kernel)), dtype=np.uint8)
    eroded = np.asarray(mask.filter(ImageFilter.MinFilter(kernel)), dtype=np.uint8)
    boundary = dilated != eroded

    # Orthorectified rasters often have a black NoData frame. Do not mistake
    # its edge for a shoreline; retain contours only inside valid imagery.
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    valid = Image.fromarray(np.where(np.max(rgb, axis=2) > 2, 255, 0).astype(np.uint8))
    valid_inside = np.asarray(valid.filter(ImageFilter.MinFilter(kernel)), dtype=np.uint8) > 0
    boundary &= valid_inside
    return Image.fromarray(np.where(boundary, 255, 0).astype(np.uint8), mode="L")


def draw_boundaries(image: Image.Image, boundary: Image.Image) -> Image.Image:
    """Paint boundary pixels pure blue while preserving all other pixels."""
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    edge = np.asarray(boundary, dtype=np.uint8) > 0
    base[edge] = (0, 0, 255)
    return Image.fromarray(base, mode="RGB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="input JPG, PNG or TIFF")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="output image (default: <input>_water_boundaries.png)",
    )
    parser.add_argument(
        "--model",
        choices=("resnet34_256", "resnet34_512"),
        default="resnet34_512",
        help="sat-water pretrained model (default: resnet34_512)",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="boundary half-width in pixels (default: 2)",
    )
    parser.add_argument(
        "--mask-output",
        type=Path,
        help="optionally save the full-resolution binary water mask",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.image.is_file():
        print(f"error: input image does not exist: {args.image}", file=sys.stderr)
        return 2

    output = args.output or args.image.with_name(
        f"{args.image.stem}_water_boundaries.png"
    )
    try:
        with Image.open(args.image) as opened:
            image = opened.convert("RGB")
        mask = water_mask(image, args.model)
        boundary = mask_boundary(mask, args.line_width, image)
        result = draw_boundaries(image, boundary)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output)
        if args.mask_output:
            args.mask_output.parent.mkdir(parents=True, exist_ok=True)
            mask.save(args.mask_output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    water_pixels = int(np.count_nonzero(np.asarray(mask)))
    boundary_pixels = int(np.count_nonzero(np.asarray(boundary)))
    total_pixels = mask.width * mask.height
    print(f"Created: {output}")
    print(f"Water coverage: {100.0 * water_pixels / total_pixels:.2f}%")
    print(f"Blue boundary pixels: {boundary_pixels}")
    if args.mask_output:
        print(f"Mask: {args.mask_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
