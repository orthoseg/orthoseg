"""Helper script to convert output_vector filenames in orthoseg project directories."""

import logging
import string
import warnings
from pathlib import Path

import geofileops as gfo

from orthoseg.helpers import config_helper as conf
from orthoseg.util import log_util
from orthoseg.util.data import aidetection_info

logger = logging.getLogger(__name__)


def main():
    """Run convert for all projects in the specified directory."""
    run_convert_output_vector_filenames(
        projects_dir=Path(r"X:\Monitoring\OrthoSeg"), simulate=True
    )


def run_convert_output_vector_filenames(
    projects_dir: Path, simulate: bool = False
) -> dict[Path, list[tuple[Path, Path]]]:
    """Convert output_vector files for all orthoseg projects in the given directory.

    Args:
        projects_dir: Path to directory containing project folders.
        simulate: If True, only log intended renames without renaming.
            Defaults to False.

    Returns:
        Dictionary mapping project directory to list of (src, dst) renamed paths.
    """
    results: dict[Path, list[tuple[Path, Path]]] = {}
    projects = [
        subdir
        for subdir in projects_dir.iterdir()
        if subdir.is_dir() and not subdir.name.startswith("_")
    ]
    for project in projects:
        renamed = convert_output_vector_filenames_project(
            project_dir=project, simulate=simulate
        )
        if renamed:
            results[project] = renamed

    return results


def convert_output_vector_filenames_project(
    project_dir: Path, simulate: bool = False
) -> list[tuple[Path, Path]]:
    """Convert output_vector files in a project directory to the configured format.

    Args:
        project_dir: Path to the project directory.
        simulate: If True, only log intended renames without renaming.
            Defaults to False.

    Returns:
        List of (src, dst) renamed file tuples.
    """
    global logger  # noqa: PLW0603

    config_path = project_dir / f"{project_dir.name}.ini"
    if not config_path.exists():
        logger.info(f"Config_path ({config_path}) doesn't exist.")
        return []

    with _LogFilterContext() as filter_ctx:
        try:
            conf.read_orthoseg_config(config_path=config_path)
        except Exception as ex:
            logger.warning(f"Error reading config ({config_path}): {ex}")
            return []
        log_dir = conf.dirs.getpath("log_dir")
        if log_dir.exists():
            logger = log_util.main_log_init(
                log_dir=log_dir,
                log_basefilename=run_convert_output_vector_filenames.__name__,
            )
            filter_ctx.add_filter()

    output_vector_dir = conf.dirs.getpath("output_vector_dir")
    output_vector_parent_dir = output_vector_dir.parent
    if not output_vector_parent_dir.exists():
        logger.info(f"Directory doesn't exist: {output_vector_parent_dir}")
        return []

    # Search for vector prediction files in output_vector_parent_dir and subdirectories
    candidate_files = [
        p
        for p in output_vector_parent_dir.rglob("*.*")
        if p.is_file() and p.suffix.lower() in (".gpkg", ".geojson", ".shp")
    ]

    renamed_files: list[tuple[Path, Path]] = []
    for file in sorted(candidate_files):
        try:
            ai_info = aidetection_info(file)
        except ValueError:
            logger.debug(f"File not an ai detection file, skipping: {file}")
            continue

        model_version = ai_info.model_version
        model_config_name = (
            model_version.split(".", 1)[1] if "." in model_version else ""
        )
        format_kwargs = {
            "segment_subject": ai_info.subject,
            "image_layer": ai_info.image_layer,
            "model_version": model_version,
            "basefilename": f"{ai_info.subject}_{model_version}",
            "epoch": "" if ai_info.epoch is None else str(ai_info.epoch),
            "traindata_id": ai_info.traindata_version,
            "model_config_name": model_config_name,
        }

        if ai_info.postprocessing == "predict":
            template = conf.files["output_vector_predict_name"]
        else:
            template = conf.files["output_vector_postp_name"]

        # Check if any placeholder in the template evaluates to empty
        used_placeholders = [
            field_name
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name
        ]
        empty_placeholders = [
            field
            for field in used_placeholders
            if format_kwargs.get(field) is None
            or str(format_kwargs.get(field)).strip() == ""
        ]
        if empty_placeholders:
            warnings.warn(
                f"Cannot convert {file.name}: placeholder(s) "
                f"{', '.join(empty_placeholders)} empty for template '{template}', "
                "skipping rename.",
                stacklevel=2,
            )
            continue

        if ai_info.postprocessing == "predict":
            new_filename = template.format(**format_kwargs)
        elif ai_info.postprocessing == "":
            new_filename = template.format(**format_kwargs)
        else:
            base_postp = Path(template.format(**format_kwargs))
            new_filename = f"{base_postp.stem}_{ai_info.postprocessing}{file.suffix}"

        new_path = file.parent / new_filename
        if new_path == file:
            logger.debug(f"File already has target name, skipping: {file.name}")
            continue

        if new_path.exists():
            logger.warning(
                f"Target file already exists, skipping rename of {file} to {new_path}"
            )
            continue

        renamed_files.append((file, new_path))
        if simulate:
            logger.info(f"Simulate rename: {file.name} -> {new_filename}")
        else:
            logger.info(f"Rename: {file.name} -> {new_filename}")
            file.rename(new_path)
            if new_path.suffix == ".gpkg":
                _rename_gpkg_layer(new_path, old_stem=file.stem, new_stem=new_path.stem)
            _rename_sidecars(file, new_path, simulate=False)

    return renamed_files


class _ConvertScriptLogFilter(logging.Filter):
    """Filter out noisy messages during conversion script execution."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "orthoseg.helpers.config_helper" or record.name.startswith(
            "orthoseg.helpers.config_helper."
        ):
            return False
        try:
            return "has_switched_axes option not found" not in record.getMessage()
        except Exception:
            return True


class _LogFilterContext:
    """Context manager to install and uninstall conversion log filter."""

    def __init__(self) -> None:
        self.filter = _ConvertScriptLogFilter()
        self.targets: list[logging.Logger | logging.Handler] = []

    def add_filter(self) -> None:
        for name in (
            "orthoseg.util.image_util",
            "orthoseg.helpers.config_helper",
        ):
            logger_obj = logging.getLogger(name)
            if not any(
                isinstance(f, _ConvertScriptLogFilter) for f in logger_obj.filters
            ):
                logger_obj.addFilter(self.filter)
                self.targets.append(logger_obj)

        root = logging.getLogger()
        if not any(isinstance(f, _ConvertScriptLogFilter) for f in root.filters):
            root.addFilter(self.filter)
            self.targets.append(root)

        for handler in root.handlers:
            if not any(isinstance(f, _ConvertScriptLogFilter) for f in handler.filters):
                handler.addFilter(self.filter)
                self.targets.append(handler)

    def __enter__(self) -> "_LogFilterContext":
        self.add_filter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for target in self.targets:
            try:
                target.removeFilter(self.filter)
            except Exception:
                pass
        self.targets.clear()


def _rename_gpkg_layer(gpkg_path: Path, old_stem: str, new_stem: str) -> None:
    """Rename GeoPackage layer if it matched old file stem."""
    try:
        layer_name = gfo.get_only_layer(gpkg_path)
        if layer_name == old_stem:
            gfo.rename_layer(gpkg_path, new_stem)
    except Exception as ex:
        logger.debug(f"Could not rename layer in {gpkg_path}: {ex}")


def _rename_sidecars(old_path: Path, new_path: Path, simulate: bool = False) -> None:
    """Rename sidecar files that share the old stem."""
    for sidecar in old_path.parent.glob(f"{old_path.stem}.*"):
        if (
            sidecar != old_path
            and sidecar.is_file()
            and not sidecar.name.startswith(new_path.stem)
        ):
            target_sidecar = old_path.parent / f"{new_path.stem}{sidecar.suffix}"
            if not target_sidecar.exists():
                if simulate:
                    logger.info(
                        f"Simulate rename sidecar: {sidecar.name} -> "
                        f"{target_sidecar.name}"
                    )
                else:
                    logger.info(
                        f"Rename sidecar: {sidecar.name} -> {target_sidecar.name}"
                    )
                    sidecar.rename(target_sidecar)


# If the script is ran directly...
if __name__ == "__main__":
    main()
