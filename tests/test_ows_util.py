"""
Tests for functionalities in orthoseg.train.
"""
import pyproj
import pytest
import rasterio as rio

from orthoseg.util import ows_util
from tests import test_helper
from tests.test_helper import TestData


def test_get_images_for_grid(tmp_path):
    # Init some stuff
    filelayer_path = (
        TestData.sampleprojects_dir
        / "fields/input_raster"
        / "BEFL-TEST-s2_2023-05-01_2023-07-01_B08-B04-B03_min_byte.tif"
    )
    # File bounds: left: 484500, bottom: 5642970, right: 499500, top: 5651030

    crs = "epsg:32631"
    pixsize_x = 5
    pixsize_y = pixsize_x
    width_pix = 512
    height_pix = 256
    width_crs = width_pix * pixsize_x
    height_crs = height_pix * pixsize_y
    pixels_overlap = 64
    xmin = 484500
    ymin = 5642970
    image_gen_bbox = (xmin, ymin, xmin + width_crs * 2, ymin + height_crs * 2)

    # Test getting the images for this grid
    # -------------------------------------
    layersource_rgb = ows_util.FileLayerSource(path=filelayer_path, layernames=["S1"])
    ows_util.get_images_for_grid(
        layersources=[layersource_rgb],
        output_image_dir=tmp_path,
        crs=crs,
        image_gen_bbox=image_gen_bbox,
        image_crs_pixel_x_size=pixsize_x,
        image_crs_pixel_y_size=pixsize_y,
        image_pixel_width=width_pix,
        image_pixel_height=height_pix,
        nb_concurrent_calls=1,
        image_format=ows_util.FORMAT_GEOTIFF,
        pixels_overlap=pixels_overlap,
        ssl_verify=True,
    )

    assert tmp_path.exists()
    image_paths = list(tmp_path.glob("**/*.tif"))

    assert len(image_paths) == 9
    for image_path in image_paths:
        assert image_path.exists()
        assert image_path.stat().st_size > 50000, "Image should be larger than 50kB"
        with rio.open(image_path) as image_file:
            assert image_file.width == width_pix + 2 * pixels_overlap
            assert image_file.height == height_pix + 2 * pixels_overlap


@pytest.mark.parametrize(
    "crs_epsg, has_switched_axes",
    [
        [4326, False],
        [31370, False],
        [3059, True],
        [31468, True],
    ],
)
def test_has_switched_axes(crs_epsg: int, has_switched_axes: bool):
    assert ows_util._has_switched_axes(pyproj.CRS(crs_epsg)) is has_switched_axes


def test_getmap_to_file_filelayer(tmp_path):
    # Init some stuff
    filelayer_path = (
        test_helper.sampleprojects_dir
        / "fields/input_raster"
        / "BEFL-TEST-s2_2023-05-01_2023-07-01_B08-B04-B03_min_byte.tif"
    )
    # File bounds: left: 484500, bottom: 5642970, right: 499500, top: 5651030
    with rio.open(filelayer_path) as filelayer:
        file_bounds = filelayer.bounds

    crs = "epsg:32631"
    pixsize_x = 5
    pixsize_y = pixsize_x
    width_pix = 512
    height_pix = 256
    width_crs = width_pix * pixsize_x
    height_crs = height_pix * pixsize_y
    xmin = 484400
    ymin = 5642900
    bbox = (xmin, ymin, xmin + width_crs, ymin + height_crs)

    # Align box to pixel size + make sure width stays the asked number of pixels
    bbox = ows_util.align_bbox_to_grid(
        bbox,
        grid_xmin=file_bounds[0],
        grid_ymin=file_bounds[1],
        pixel_size_x=pixsize_x,
        pixel_size_y=pixsize_y,
    )
    bbox = (bbox[0], bbox[1], bbox[0] + width_crs, bbox[1] + height_crs)

    # Test getting a standard 3 band image from a file layer
    # ------------------------------------------------------
    layersource_rgb = ows_util.FileLayerSource(path=filelayer_path, layernames=["S1"])
    image_path = ows_util.getmap_to_file(
        layersources=layersource_rgb,
        output_dir=tmp_path,
        crs=crs,
        bbox=bbox,
        size=(width_pix, height_pix),
        image_pixels_ignore_border=0,
        transparent=False,
        layername_in_filename=True,
    )

    assert image_path is not None
    assert image_path.exists()
    assert image_path.stat().st_size > 50000, "Image should be larger than 50kB"
    with rio.open(image_path) as image_file:
        assert tuple(image_file.bounds) == bbox
        assert image_file.width == width_pix
        assert image_file.height == height_pix


def test_getmap_to_file_wmslayer(tmp_path):
    # Init some stuff
    projection = "epsg:31370"
    pixsize_x = 0.25
    pixsize_y = pixsize_x
    width_pix = 512
    height_pix = 256
    width_crs = width_pix * pixsize_x
    height_crs = height_pix * pixsize_y
    xmin = 160000
    ymin = 170000
    bbox = (xmin, ymin, xmin + width_crs, ymin + height_crs)

    # Test getting a standard 3 band image from a WMS layer
    # -----------------------------------------------------
    layersource_rgb = ows_util.WMSLayerSource(
        wms_server_url="https://geo.api.vlaanderen.be/omw/wms?",
        layernames=["OMWRGB20VL"],
        layerstyles=["default"],
    )
    image_path = ows_util.getmap_to_file(
        layersources=layersource_rgb,
        output_dir=tmp_path,
        crs=projection,
        bbox=bbox,
        size=(width_pix, height_pix),
        image_format=ows_util.FORMAT_PNG,
        image_pixels_ignore_border=0,
        transparent=False,
        layername_in_filename=True,
    )

    assert image_path is not None
    assert image_path.exists()
    assert image_path.stat().st_size > 50000, "Image should be larger than 50kB"
    with rio.open(image_path) as image_file:
        # assert tuple(image_file.bounds) == bbox
        assert image_file.width == width_pix
        assert image_file.height == height_pix

    # Test creating greyscale image
    # -----------------------------
    # If band -1 is specified, a greyscale version of the rgb image will be created
    layersource_dhm_ortho_grey = ows_util.WMSLayerSource(
        wms_server_url="https://geo.api.vlaanderen.be/ogw/wms?",
        layernames=["OGWRGB13_15VL"],
        bands=[-1],
    )
    image_path = ows_util.getmap_to_file(
        layersources=layersource_dhm_ortho_grey,
        output_dir=tmp_path,
        crs=projection,
        bbox=bbox,
        size=(width_pix, height_pix),
        image_format=ows_util.FORMAT_PNG,
        # image_format_save=ows_util.FORMAT_TIFF,
        image_pixels_ignore_border=0,
        transparent=False,
        layername_in_filename=True,
    )

    assert image_path is not None
    assert image_path.exists() is True
    with rio.open(image_path) as image_file:
        # assert tuple(image_file.bounds) == bbox
        assert image_file.width == width_pix
        assert image_file.height == height_pix

    # Test combining bands of 3 different WMS layers
    # ----------------------------------------------
    # Layer sources for for skyview and hillshade
    wms_server_url_dhm = "https://geo.api.vlaanderen.be/DHMV/wms?"
    layersource_dhm_skyview = ows_util.WMSLayerSource(
        wms_server_url=wms_server_url_dhm,
        layernames=["DHMV_II_SVF_25cm"],
        bands=[0],
    )
    layersource_dhm_hill = ows_util.WMSLayerSource(
        wms_server_url=wms_server_url_dhm,
        layernames=["DHMV_II_HILL_25cm"],
        bands=[0],
    )
    layersources = [
        layersource_dhm_skyview,
        layersource_dhm_hill,
        layersource_dhm_ortho_grey,
    ]
    image_path = ows_util.getmap_to_file(
        layersources=layersources,
        output_dir=tmp_path,
        crs=projection,
        bbox=bbox,
        size=(width_pix, height_pix),
        image_format=ows_util.FORMAT_PNG,
        # image_format_save=ows_util.FORMAT_TIFF,
        image_pixels_ignore_border=0,
        transparent=False,
        layername_in_filename=True,
    )

    assert image_path is not None
    assert image_path.exists() is True
    with rio.open(image_path) as image_file:
        # assert tuple(image_file.bounds) == bbox
        assert image_file.width == width_pix
        assert image_file.height == height_pix
