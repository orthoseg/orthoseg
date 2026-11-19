"""Convert training PNG images to JPEG for all projects under a projects directory."""

import logging
import tempfile
from pathlib import Path

from PIL import Image

from orthoseg.helpers import config_helper as conf

logger = logging.getLogger(__name__)

_TRAIN_DATA_SPLITS = ("train", "validation", "test")
_JPEG_QUALITY = 95


def main() -> None:
    """Run the converter for all projects in the configured directory."""
    run_convert_train_images_png_to_jpg(
        projects_dir=Path(r"X:\Monitoring\OrthoSeg"), simulate=True
    )


def run_convert_train_images_png_to_jpg(
    projects_dir: Path, simulate: bool = False
) -> dict[Path, list[tuple[Path, Path]]]:
    """Convert PNG training images for each project in ``projects_dir``.

    Only files in each training version's ``train``, ``validation``, and ``test``
    ``image`` directories are converted; mask directories are not modified.

    Args:
        projects_dir: Directory containing the project directories.
        simulate: If True, log intended conversions without changing any files.

    Returns:
        A mapping from project directories to converted (or planned) image paths.
    """
    results: dict[Path, list[tuple[Path, Path]]] = {}
    projects = sorted(
        path
        for path in projects_dir.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
    for project_dir in projects:
        converted = convert_train_images_png_to_jpg_project(
            project_dir=project_dir, simulate=simulate
        )
        if converted:
            results[project_dir] = converted
    return results


def convert_train_images_png_to_jpg_project(
    project_dir: Path, simulate: bool = False
) -> list[tuple[Path, Path]]:
    """Convert PNG images in a project's training, validation, and test splits.

    Matching ``.pgw`` world files are renamed to ``.jgw`` alongside the JPEGs.
    If the JPEG or its matching world file already exists, that PNG is skipped
    to avoid overwriting existing data.
    """
    config_path = project_dir / f"{project_dir.name}.ini"
    if not config_path.is_file():
        logger.info("Config path does not exist: %s", config_path)
        return []

    conf.read_orthoseg_config(config_path=config_path)
    training_dir = conf.dirs.getpath("training_dir")
    if not training_dir.is_dir():
        logger.info("Training directory does not exist: %s", training_dir)
        return []

    converted: list[tuple[Path, Path]] = []
    version_dirs = sorted(path for path in training_dir.iterdir() if path.is_dir())
    for version_dir in version_dirs:
        for split in _TRAIN_DATA_SPLITS:
            image_dir = version_dir / split / "image"
            if not image_dir.is_dir():
                continue
            png_paths = sorted(
                path
                for path in image_dir.rglob("*")
                if path.is_file() and path.suffix.lower() == ".png"
            )
            for png_path in png_paths:
                jpg_path = png_path.with_suffix(".jpg")
                png_worldfile = png_path.with_suffix(".pgw")
                jpg_worldfile = png_path.with_suffix(".jgw")
                if jpg_path.exists():
                    logger.warning(
                        "JPEG already exists; skipping conversion of %s", png_path
                    )
                    continue
                if jpg_worldfile.exists():
                    logger.warning(
                        "JPEG world file already exists; skipping conversion of %s",
                        png_path,
                    )
                    continue

                converted.append((png_path, jpg_path))
                if simulate:
                    logger.info("Simulate conversion: %s -> %s", png_path, jpg_path)
                    if png_worldfile.exists():
                        logger.info(
                            "Simulate world file rename: %s -> %s",
                            png_worldfile,
                            jpg_worldfile,
                        )
                    continue

                with tempfile.NamedTemporaryFile(
                    dir=png_path.parent, suffix=".jpg", delete=False
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                try:
                    with Image.open(png_path) as image:
                        image.convert("RGB").save(
                            temporary_path, format="JPEG", quality=_JPEG_QUALITY
                        )
                    temporary_path.replace(jpg_path)
                finally:
                    temporary_path.unlink(missing_ok=True)

                if png_worldfile.exists():
                    png_worldfile.rename(jpg_worldfile)
                png_path.unlink()
                logger.info("Converted %s -> %s", png_path, jpg_path)

    return converted


if __name__ == "__main__":
    main()
