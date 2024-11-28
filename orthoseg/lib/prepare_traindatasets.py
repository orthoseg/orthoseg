"""Module to prepare the training datasets."""

import logging
import math
import pprint
import shutil
import warnings
from pathlib import Path

import geofileops as gfo
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.features as rio_features
import rasterio.profiles as rio_profiles
import shapely
import shapely.geometry as sh_geom
from PIL import Image

from orthoseg.util import image_util, vector_util
from orthoseg.util.progress_util import ProgressLogger

# Get a logger...
logger = logging.getLogger(__name__)


class LabelInfo:
    """Information needed to find train labels."""

    def __init__(
        self,
        locations_path: Path,
        polygons_path: Path,
        image_layer: str,
        pixel_x_size: float | None = None,
        pixel_y_size: float | None = None,
        locations_gdf: gpd.GeoDataFrame | None = None,
        polygons_gdf: gpd.GeoDataFrame | None = None,
    ):
        """Conctructor of LabelInfo.

        Args:
            locations_path (Path): file path to the locations file.
            polygons_path (Path): file path to the polygons file.
            image_layer (str): image layer for the label.
            pixel_x_size (Optional[float], optional): _description_. Defaults to None.
            pixel_y_size (Optional[float], optional): _description_. Defaults to None.
            locations_gdf (Optional[gpd.GeoDataFrame], optional): _description_.
                Defaults to None.
            polygons_gdf (Optional[gpd.GeoDataFrame], optional): _description_.
                Defaults to None.
        """
        self.locations_path = locations_path
        self.polygons_path = polygons_path
        self.image_layer = image_layer
        self.pixel_x_size = pixel_x_size
        self.pixel_y_size = pixel_y_size
        self.locations_gdf = locations_gdf
        self.polygons_gdf = polygons_gdf

    def __repr__(self) -> str:
        """Return the object information as readable string.

        Returns:
            str: readable string representation of object.
        """
        repr = (
            f"LabelInfo(image_layer={self.image_layer}, locations_path="
            f"{self.locations_path}, polygons_path={self.polygons_path},...)"
        )
        return repr


class ValidationError(ValueError):
    """A validation exception.

    Args:
        ValueError (_type_): _description_
    """

    def __init__(self, message, errors):
        """Conctructor of ValidationError.

        Args:
            message (str): error message.
            errors (list[str]): List of validation errors.
        """
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors

    def __repr__(self):
        """Formats validation errors to string.

        Returns:
            str: returns the validation errors as string.
        """
        repr = super().__repr__()
        if self.errors is not None and len(self.errors) > 0:
            repr += f"\n  -> Errors: {pprint.pformat(self.errors, indent=4, width=100)}"
        return repr

    def __str__(self) -> str:
        """Formats validation errors to string.

        Returns:
            str: returns the validation errors as string.
        """
        repr = super().__str__()
        if self.errors is not None and len(self.errors) > 0:
            repr += f"\n  -> Errors: {pprint.pformat(self.errors, indent=4, width=100)}"
        return repr

    def to_html(self) -> str:
        """Formats validation errors to html.

        Returns:
            str: the validation errors as html.
        """
        repr = super().__str__()
        if self.errors is not None and len(self.errors) > 0:
            errors_df = pd.DataFrame(self.errors, columns=["error"])
            repr += f"{errors_df.to_html()}"
        return repr


def prepare_traindatasets(
    label_infos: list[LabelInfo],
    classes: dict,
    image_layers: dict,
    training_dir: Path,
    labelname_column: str = "classname",
    image_pixel_x_size: float = 0.25,
    image_pixel_y_size: float = 0.25,
    image_pixel_width: int = 512,
    image_pixel_height: int = 512,
    ssl_verify: bool | str = True,
    only_validate: bool = False,
    force: bool = False,
) -> tuple[Path, int]:
    """This function prepares training data for the vector labels provided.

    It will:
        * get orthophoto images from the correct image_layer
        * create the corresponding label mask for each image

    Returns a tuple with (output_dir, dataversion):
        output_dir: the dir where the traindataset was created/found
        dataversion: a version number for the dataset created/found

    Args:
        label_infos (list[LabelInfo]): paths to the files with label polygons
            and locations to generate images for.
        classes (dict): dict with the classes to detect as keys. The values
            are the following:
                - labelnames: list of labels to use for this class
                - weight:
                - burn_value:
        image_layers (dict): the image layers available with their properties.
        training_dir (Path): the directory to save the training data to.
        labelname_column (str): the column where the label names are stored in
            the polygon files. If the column name specified is not found, column
            "label_name" is used if it exists for backwards compatibility.
        image_pixel_x_size (float): pixel size in x direction. Defaults to 0.25.
        image_pixel_y_size (float): pixel size in y direction. Defaults to 0.25.
        image_pixel_width (int): width of the image in pixels. Defaults to 512.
        image_pixel_height (int): height of the image in pixels. Defaults to 512.
        ssl_verify (bool or str, optional): True to use the default
            certificate bundle as installed on your system. False disables
            certificate validation (NOT recommended!). If a path to a
            certificate bundle file (.pem) is passed, this will be used.
            In corporate networks using a proxy server this is often needed
            to evade CERTIFICATE_VERIFY_FAILED errors. Defaults to True.
        only_validate (bool, optional): True to only validate the input label data.
        force (bool, opitional): True to force recreation of output files.
            Defaults to False.
    """
    # Check if the first class is named "background"
    if len(classes) == 0:
        raise Exception("No classes specified")
    elif next(iter(classes)).lower() != "background":
        classes_str = pprint.pformat(classes, sort_dicts=False, width=50)
        raise Exception(
            f"By convention, the first class must be called background!\n{classes_str}"
        )

    # Check if the latest version of training data is already ok
    # Determine the current data version based on existing output data dir(s),
    # If dir ends on _TMP_* ignore it, as it (probably) ended with an error.
    output_dirs = [
        output_dir
        for output_dir in training_dir.glob("[0-9]*/")
        if "_TMP_" not in output_dir.name
    ]

    reuse_traindata = False
    dataversion_mostrecent = None
    if len(output_dirs) == 0:
        dataversion_new = 1
    else:
        # Get the output dir with the highest version (=first if sorted desc)
        output_dir_mostrecent = sorted(output_dirs, reverse=True)[0]
        dataversion_mostrecent = int(output_dir_mostrecent.name)

        # If none of the input files changed since previous run, reuse dataset
        for label_file in label_infos:
            reuse_traindata = True
            labellocations_output_mostrecent_path = (
                output_dir_mostrecent / label_file.locations_path.name
            )
            labeldata_output_mostrecent_path = (
                output_dir_mostrecent / label_file.polygons_path.name
            )
            if not (
                labellocations_output_mostrecent_path.exists()
                and labeldata_output_mostrecent_path.exists()
                and gfo.cmp(
                    label_file.locations_path, labellocations_output_mostrecent_path
                )
                and gfo.cmp(label_file.polygons_path, labeldata_output_mostrecent_path)
            ):
                reuse_traindata = False
                break
        if reuse_traindata:
            dataversion_new = dataversion_mostrecent
            logger.info(
                "Input label file(s) didn't change since last prepare_traindatasets, "
                f"so reuse version {dataversion_new}"
            )
        else:
            dataversion_new = dataversion_mostrecent + 1
            logger.info(
                "Input label file(s) changed since last prepare_traindatasets, "
                f"so create new training data version {dataversion_new}"
            )

    # Determine the output dir
    training_dataversion_dir = training_dir / f"{dataversion_new:02d}"

    # If the train data is already ok, just return
    if reuse_traindata is True:
        return (training_dataversion_dir, dataversion_new)

    # The input labels have changed: create new train version
    # -------------------------------------------------------
    # read, validate and prepare data
    labeldata = prepare_labeldata(
        label_infos=label_infos,
        classes=classes,
        labelname_column=labelname_column,
        image_pixel_x_size=image_pixel_x_size,
        image_pixel_y_size=image_pixel_y_size,
        image_pixel_width=image_pixel_width,
        image_pixel_height=image_pixel_height,
    )

    # Reading label data was succesfull.
    if only_validate:
        return (training_dataversion_dir, dataversion_new)

    # Prepare temp dir to put training dataset in.
    # Use a temp dir, so it can be removed/ignored if an error occurs later on.
    output_tmp_dir = create_tmp_dir(
        training_dir, f"{dataversion_new:02d}", remove_existing=True
    )

    # Copy the label input files to dest dir + to a backup dir.
    for label_file in label_infos:
        gfo.copy(label_file.locations_path, output_tmp_dir)
        gfo.copy(label_file.polygons_path, output_tmp_dir)
        backup_dir = label_file.locations_path.parent / f"backup_v{dataversion_new:02d}"
        shutil.rmtree(backup_dir, ignore_errors=True)
        backup_dir.mkdir(parents=True, exist_ok=True)
        gfo.copy(label_file.locations_path, backup_dir)
        gfo.copy(label_file.polygons_path, backup_dir)

    # Now create the images/masks for the new train version for the different traindata
    # types.
    traindata_types = ["train", "validation", "test"]
    nb_todo = 0
    for labellocations_gdf, _ in labeldata:
        labellocations_curr_gdf = labellocations_gdf[
            labellocations_gdf["traindata_type"].isin(traindata_types)
        ]
        nb_todo += len(labellocations_curr_gdf)
    progress = ProgressLogger(message="prepare training images", nb_steps_total=nb_todo)
    logger.info(f"Get images for {nb_todo} labels")

    for traindata_type in traindata_types:
        if dataversion_mostrecent is not None:
            previous_dataversion_dir = training_dir / f"{dataversion_mostrecent:02d}"
            previous_imagedata_image_dir = (
                previous_dataversion_dir / traindata_type / "image"
            )
        # Create output dirs...
        output_imagedatatype_dir = output_tmp_dir / traindata_type
        output_imagedata_image_dir = output_imagedatatype_dir / "image"
        output_imagedata_mask_dir = output_imagedatatype_dir / "mask"
        for dir in [
            output_imagedatatype_dir,
            output_imagedata_mask_dir,
            output_imagedata_image_dir,
        ]:
            if dir and not dir.exists():
                dir.mkdir(parents=True, exist_ok=True)

        for labellocations_gdf, labels_to_burn_gdf in labeldata:
            try:
                # Get the label locations for this traindata type
                labellocations_curr_gdf = labellocations_gdf[
                    labellocations_gdf["traindata_type"] == traindata_type
                ]

                # Loop trough all locations labels to get an image for each of them
                for i, label_tuple in enumerate(labellocations_curr_gdf.itertuples()):
                    img_bbox = label_tuple.geometry
                    image_layer = getattr(label_tuple, "image_layer")

                    # Prepare file name for the image
                    assert labellocations_gdf.crs is not None
                    output_filename = image_util.create_filename(
                        crs=labellocations_gdf.crs,
                        bbox=img_bbox.bounds,
                        size=(image_pixel_width, image_pixel_height),
                        image_format=image_util.FORMAT_PNG,
                        layername="_".join(
                            image_layers[image_layer]["layersources"][0].layernames
                        ),
                    )

                    # If the image exists already in the previous version, reuse it.
                    if (
                        dataversion_mostrecent is not None
                        and (previous_imagedata_image_dir / output_filename).exists()
                    ):
                        image_filepath = shutil.copy(
                            src=previous_imagedata_image_dir / output_filename,
                            dst=output_imagedata_image_dir / output_filename,
                        )
                        pgw_filename = output_filename.replace(".png", ".pgw")
                        if (previous_imagedata_image_dir / pgw_filename).exists():
                            shutil.copy(
                                src=previous_imagedata_image_dir / pgw_filename,
                                dst=output_imagedata_image_dir / pgw_filename,
                            )
                    else:
                        # Get the image from the WMS service.
                        image_filepath = image_util.load_image_to_file(
                            layersources=image_layers[image_layer]["layersources"],
                            output_dir=output_imagedata_image_dir,
                            crs=labellocations_gdf.crs,
                            bbox=img_bbox.bounds,
                            size=(image_pixel_width, image_pixel_height),
                            ssl_verify=ssl_verify,
                            image_format=image_util.FORMAT_PNG,
                            # image_format_save=image_util.FORMAT_TIFF,
                            image_pixels_ignore_border=image_layers[image_layer][
                                "image_pixels_ignore_border"
                            ],
                            transparent=False,
                            layername_in_filename=True,
                            output_filename=output_filename,
                        )

                    # Create a mask corresponding with the image file
                    # Mask should never be in a lossy format -> png!
                    mask_filepath = Path(
                        str(image_filepath)
                        .replace(
                            str(output_imagedata_image_dir),
                            str(output_imagedata_mask_dir),
                        )
                        .replace(".jpg", ".png")
                    )
                    nb_classes = len(classes)

                    # Only keep the labels that are meant for this image layer
                    if "image_layer" not in labels_to_burn_gdf.columns:
                        print("odd")
                    labels_for_layer_gdf = (
                        labels_to_burn_gdf.loc[
                            labels_to_burn_gdf["image_layer"] == image_layer
                        ]
                    ).copy()
                    if len(labels_for_layer_gdf) == 0:
                        logger.info(f"No polygons to burn for {image_layer=}!")
                    _create_mask(
                        input_image_filepath=image_filepath,
                        output_mask_filepath=mask_filepath,
                        labels_to_burn_gdf=labels_for_layer_gdf,
                        nb_classes=nb_classes,
                        force=force,
                    )

                    # Log the progress and prediction speed
                    progress.step()

            except Exception as ex:
                raise ex

    # If everything went fine, rename output_tmp_dir to the final output_dir
    output_tmp_dir.rename(training_dataversion_dir)

    return (training_dataversion_dir, dataversion_new)


def prepare_labeldata(
    label_infos: list[LabelInfo],
    classes: dict,
    labelname_column: str,
    image_pixel_x_size: float | None,
    image_pixel_y_size: float | None,
    image_pixel_width: int,
    image_pixel_height: int,
) -> list[tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]:
    """Prepare and validate the data in the labelinfos.

    The GeoDataFrames returned can be used to fetch train images and burn masks.

    Args:
        label_infos (list[LabelInfo]): the label files/data.
        classes (dict): dict with classes and their corresponding
            label class names + weights.
        labelname_column (str): the column name in the label polygon files where the
            label classname can be found. Defaults to "classname".
        image_pixel_x_size (float): pixel size in x direction.
        image_pixel_y_size (float): pixel size in y direction.
        image_pixel_width (int): width of the image in pixels.
        image_pixel_height (int): height of the image in pixels.

    Raises:
        ValidationError: if the data is somehow not valid. All issues are listed in the
            errors property.

    Returns:
        list[tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]]: returns a list of tuples with
            geodataframes with labellocations and labelpolygons to burn.
    """
    labeldata_result = []
    validation_errors = []
    validation_warnings = []
    train_locations_found = False
    validation_locations_found = False
    for label_info in label_infos:
        # Read label locations if needed
        if label_info.locations_gdf is not None:
            labellocations_gdf = label_info.locations_gdf
        else:
            logger.debug(f"Read label locations from {label_info.locations_path}")
            labellocations_gdf = gfo.read_file(label_info.locations_path)
            labellocations_gdf["path"] = str(label_info.locations_path)
            labellocations_gdf["image_layer"] = label_info.image_layer
            # Remark: geopandas 0.7.0 drops fid column internally!
            labellocations_gdf["row_nb_orig"] = labellocations_gdf.index
            if len(labellocations_gdf) == 0:
                logger.warning(
                    f"No label locations found in {label_info.locations_path}"
                )
                continue

        # Read label polygons if needed
        if label_info.polygons_gdf is not None:
            labelpolygons_gdf = label_info.polygons_gdf
        else:
            logger.debug(f"Read label data from {label_info.polygons_path}")
            labelpolygons_gdf = gfo.read_file(label_info.polygons_path)
            labelpolygons_gdf["path"] = str(label_info.polygons_path)
            labelpolygons_gdf["image_layer"] = label_info.image_layer
            if len(labelpolygons_gdf) == 0:
                logger.warning(f"No label polygons found in {label_info.polygons_path}")

        assert labellocations_gdf is not None
        assert labelpolygons_gdf is not None

        if labellocations_gdf is None or len(labellocations_gdf) == 0:
            validation_errors.append("No label locations found in labellocations_gdf")
            continue

        # Validate + process the location data
        # ------------------------------------
        if "traindata_type" not in labellocations_gdf.columns:
            validation_errors.append(
                "Mandatory column traindata_type not found in "
                f"{label_info.locations_path}"
            )

        # Check + correct the geom of the location to make sure it matches the size of
        # the training images/masks to be generated...
        # Prepare pixel size: if defined on label info level, use that.
        pixel_x_size = label_info.pixel_x_size
        pixel_x_size = pixel_x_size if pixel_x_size is not None else image_pixel_x_size
        pixel_y_size = label_info.pixel_y_size
        pixel_y_size = pixel_y_size if pixel_y_size is not None else image_pixel_y_size
        assert pixel_x_size is not None
        assert pixel_y_size is not None

        # Tile width/height in units of crs
        image_crs_width = math.fabs(image_pixel_width * pixel_x_size)
        image_crs_height = math.fabs(image_pixel_height * pixel_y_size)

        locations_none = []
        for location in labellocations_gdf.itertuples():
            if location.geometry is None or location.geometry.is_empty:
                logger.warning(
                    f"No or empty geometry found in file {Path(location.path).name} "
                    f"for index {location.Index}, it will be ignored"
                )
                locations_none.append(location.Index)
                continue

            # Check if the traindata_type is valid
            traindata_types = ["train", "validation", "test", "todo"]
            if location.traindata_type not in traindata_types:
                validation_errors.append(
                    f"Invalid traindata_type in {Path(location.path).name}: "
                    f"{location.traindata_type} not in {traindata_types}, wkt:"
                    f"{location.geometry.wkt}"
                )

            # Check if the geometry is valid
            is_valid_reason = shapely.is_valid_reason(
                labellocations_gdf.geometry[location.Index]
            )
            if is_valid_reason != "Valid Geometry":
                validation_errors.append(
                    f"Invalid geometry in {Path(location.path).name}: "
                    f"{is_valid_reason}"
                )
                continue

            geom_bounds = location.geometry.bounds
            xmin = geom_bounds[0] - (geom_bounds[0] % image_pixel_x_size)
            ymin = geom_bounds[1] - (geom_bounds[1] % image_pixel_y_size)
            xmax = xmin + image_crs_width
            ymax = ymin + image_crs_height
            location_geom_aligned = sh_geom.box(xmin, ymin, xmax, ymax)

            # Check if the realigned geom overlaps good enough with the original
            intersection = location_geom_aligned.intersection(
                labellocations_gdf.at[location.Index, "geometry"]
            )
            assert image_pixel_x_size is not None
            assert image_pixel_y_size is not None
            area_1row_1col = (
                image_pixel_x_size * image_crs_width
                + image_pixel_y_size * image_crs_height
            )
            if intersection.area < (location_geom_aligned.area - area_1row_1col):
                # Original geom was digitized too small
                validation_errors.append(
                    f"Location geometry skewed or too small ({intersection.area}, "
                    f"based on train config expected {location_geom_aligned.area}) in "
                    f"{Path(location.path).name}: {location.geometry.wkt}"
                )
            elif sh_geom.box(*geom_bounds).area > location_geom_aligned.area * 1.1:
                validation_warnings.append(
                    f"Location geometry too large ({sh_geom.box(*geom_bounds).area}, "
                    f"based on train config expected {location_geom_aligned.area}) "
                    f"in file {Path(location.path).name}: {location.geometry.wkt}"
                )
            labellocations_gdf.at[location.Index, "geometry"] = location_geom_aligned

        # Remove locations with None or point/line geoms
        labellocations_gdf = labellocations_gdf[
            ~labellocations_gdf.index.isin(locations_none)
        ]

        # Check if labellocations has a proper crs
        if labellocations_gdf.crs is None:
            validation_errors.append(
                "No crs in labellocations, labellocation_gdf.crs: "
                f"{labellocations_gdf.crs}"
            )

        # Check that there is at least one train location and one validation location.
        if len(labellocations_gdf.query("traindata_type == 'train'")) > 0:
            train_locations_found = True
        if len(labellocations_gdf.query("traindata_type == 'validation'")) > 0:
            validation_locations_found = True

        # Validate + process the polygons data
        # ------------------------------------
        if labelpolygons_gdf is None:
            validation_errors.append("No labelpolygons in the training data!")
            continue

        # Create list with only the input polygons that need to be burned in the mask
        labels_to_burn_gdf = labelpolygons_gdf.copy()
        labels_to_burn_gdf["burn_value"] = None

        if labelname_column not in labelpolygons_gdf.columns:
            # For backwards compatibility, also support old default column name
            labelname_column = "label_name"

        if labelname_column in labelpolygons_gdf.columns:
            # If there is a column labelname_column, use the burn values specified in
            # the configuration
            for classname in classes:
                labels_to_burn_gdf.loc[
                    (
                        labels_to_burn_gdf[labelname_column].isin(
                            classes[classname]["labelnames"]
                        )
                    ),
                    "burn_value",
                ] = classes[classname]["burn_value"]

            # Check if there are invalid class names
            invalid_gdf = labels_to_burn_gdf.loc[
                labels_to_burn_gdf["burn_value"].isnull()
            ]
            for _, invalid_row in invalid_gdf.iterrows():
                validation_errors.append(
                    f"Invalid classname in {Path(invalid_row['path']).name}: "
                    f"{invalid_row[labelname_column]}"
                )

            # Filter away rows that are going to burn 0, as this is useless...
            labels_to_burn_gdf = labels_to_burn_gdf.loc[
                labels_to_burn_gdf["burn_value"] != 0
            ].copy()

        elif len(classes) == 2:
            # There is no column with label names, but there are only 2 classes
            # (background + subject), so no problem...
            logger.info(
                f"Column ({labelname_column}) not found, so use all polygons in "
                f"{label_info.polygons_path.name}"
            )
            labels_to_burn_gdf.loc[:, "burn_value"] = 1

        else:
            # There is no column with label names, but more than two classes, so stop.
            validation_errors.append(
                f"Column {labelname_column} is mandatory in labeldata if multiple "
                f"classes specified: {classes}"
            )

        # Filter away None and empty geometries... they cannot be burned
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)
            labels_to_burn_gdf = labels_to_burn_gdf[
                ~labels_to_burn_gdf.geometry.is_empty
                & labels_to_burn_gdf.geometry.notna()
            ].copy()

        # Make sure all label polygons are valid
        labels_to_burn_gdf.loc[:, "is_valid_reason"] = vector_util.is_valid_reason(
            labels_to_burn_gdf.geometry
        )
        invalid_gdf = labels_to_burn_gdf.query("is_valid_reason != 'Valid Geometry'")
        for invalid in invalid_gdf.itertuples():
            validation_errors.append(
                f"Invalid geometry in {Path(invalid.path).name}: "
                f"{invalid.is_valid_reason}"
            )

        assert isinstance(labels_to_burn_gdf, gpd.GeoDataFrame)
        labeldata_result.append((labellocations_gdf, labels_to_burn_gdf))

    # Check that there is at least one train location and one validation location.
    if not train_locations_found:
        validation_errors.append(
            "No labellocations with traindata_type == 'train' found in any file! At "
            "least one needed"
        )
    if not validation_locations_found:
        validation_errors.append(
            "No labellocations with traindata_type == 'validation' found in any file! "
            "At least one needed, but ~10% of number of 'train' locations recommended."
        )

    # If errors found, raise
    if len(validation_errors) > 0:
        raise ValidationError(
            f"Errors found in label data: {len(validation_errors)}", validation_errors
        )

    return labeldata_result


def create_tmp_dir(
    parent_dir: Path, dir_name: str, remove_existing: bool = False
) -> Path:
    """Helper function to create a 'TMP' dir based on a directory name.

    The temp dir will be named like this:
        parent_dir / <dir_name>_TMP_<sequence>

    Use: if you want to write data to a directory in a "transactional way",
    it is the safest to write to a temp dir first, and then rename it to the
    final name. This way, if a hard crash occurs while writing the data, it
    is clear that the directory wasn't ready. Additionally, in case of a hard
    crash, file locks can remain which makes it impossible to remove a
    directory for a while.

    Args:
        parent_dir (Path): The dir to create the temp dir in.
        dir_name (str): The name of the dir to base the temp dir on.
        remove_existing (bool, optional): If True, existing TMP directories
            will be removed if possible. Defaults to False.

    Raises:
        Exception: [description]

    Returns:
        Path: [description]
    """
    # The final dir should not exists yet!
    final_dir = parent_dir / dir_name
    if final_dir.exists():
        raise Exception(
            f"It is not supported to create a TMP dir for an existing dir: {final_dir}"
        )

    # Try to delete all existing TMP dir's
    if remove_existing is True:
        existing_tmp_dirs = parent_dir.glob(f"{dir_name}_TMP_*")
        for existing_tmp_dir in existing_tmp_dirs:
            try:
                shutil.rmtree(existing_tmp_dir)
            except Exception:
                tmp_dir = None

    # Create the TMP dir
    tmp_dir = None
    for i in range(100):
        tmp_dir = parent_dir / f"{dir_name}_TMP_{i:02d}"

        if tmp_dir.exists():
            # If it (still) exists try next sequence
            tmp_dir = None
            continue
        else:
            # If it doesn't exist, try to create it
            try:
                tmp_dir.mkdir(parents=True)
                break
            except Exception:
                # If it fails to create, try next sequence
                tmp_dir = None
                continue

    # If no output tmp dir could be found/created... stop...
    if tmp_dir is None:
        raise Exception(
            f"Error creating/replacing TMP dir for {dir_name} in {parent_dir}"
        )

    return tmp_dir


def _create_mask(
    input_image_filepath: Path,
    output_mask_filepath: Path,
    labels_to_burn_gdf: gpd.GeoDataFrame,
    nb_classes: int = 1,
    output_imagecopy_filepath: Path | None = None,
    minimum_pct_labeled: float = 0.0,
    force: bool = False,
) -> bool | None:
    # If file exists already and force is False... stop.
    if force is False and output_mask_filepath.exists():
        logger.debug(
            f"Output file exist, and force is False, return: {output_mask_filepath}"
        )
        return None

    # Create a mask corresponding with the image file
    # First read the properties of the input image to copy them for the output
    logger.debug(f"Create mask to {output_mask_filepath}")
    with rio.open(input_image_filepath) as image_ds:
        image_input_profile = image_ds.profile
        image_transform_affine = image_ds.transform
        bounds = image_ds.bounds

    # Prepare the file profile for the mask depending on output type
    output_ext_lower = output_mask_filepath.suffix.lower()
    if output_ext_lower == ".tif":
        image_output_profile = rio_profiles.DefaultGTiffProfile(
            count=1, transform=image_transform_affine, crs=image_input_profile["crs"]
        )
    if output_ext_lower == ".png":
        image_output_profile = rio_profiles.Profile(driver="PNG", count=1)
    else:
        raise Exception(
            f"Unsupported mask suffix (should be lossless format!): {output_ext_lower}"
        )
    image_output_profile.update(
        width=image_input_profile["width"],
        height=image_input_profile["height"],
        dtype=rio.uint8,
    )

    # Filter the vectors that intersect the image bounds
    labels_to_burn_gdf = labels_to_burn_gdf.loc[
        labels_to_burn_gdf.intersects(
            sh_geom.box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        )
    ].copy()

    # Burn the vectors in a mask
    burn_shapes = [
        (geom, value)
        for geom, value in zip(
            labels_to_burn_gdf.geometry, labels_to_burn_gdf.burn_value
        )
        if geom is not None and geom.is_empty is False
    ]
    if len(burn_shapes) > 0:
        try:
            mask_arr = rio_features.rasterize(
                shapes=burn_shapes,
                transform=image_transform_affine,
                dtype=rio.uint8,
                fill=0,
                out_shape=(
                    image_output_profile["width"],
                    image_output_profile["height"],
                ),
            )

        except Exception as ex:
            raise Exception(f"Error creating mask for {image_transform_affine}") from ex
    else:
        mask_arr = np.zeros(
            shape=(
                image_output_profile["width"],
                image_output_profile["height"],
            ),
            dtype=rio.uint8,
        )

    # Check if the mask meets the requirements to be written...
    if minimum_pct_labeled > 0:
        nb_pixels = np.size(mask_arr, 0) * np.size(mask_arr, 1)
        nb_pixels_data = nb_pixels - np.sum(
            mask_arr == 0
        )  # np.count_nonnan(image == NoData_value)
        logger.debug(
            f"nb_pixels: {nb_pixels}, nb_pixels_data: {nb_pixels_data}, "
            f"pct data: {nb_pixels_data / nb_pixels}"
        )

        if nb_pixels_data / nb_pixels < minimum_pct_labeled:
            return False

    # Write the labeled mask as .png (so without transform/crs info)
    im = Image.fromarray(mask_arr)
    im.save(output_mask_filepath)

    return True
