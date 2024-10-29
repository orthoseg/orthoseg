import pytest
from osgeo import gdal


@pytest.mark.skip(reason="No actual test, just a demo for gdal bug")
@pytest.mark.parametrize("input_format", ["VRT", "GTiff"])
def test_gdal_bug(tmp_path, input_format):
    # WMTS url
    wmts_url = (
        "WMTS:https://geo.api.vlaanderen.be/omw/wmts/1.0.0/WMTSCapabilities.xml,"
        "layer=omwrgb23vl,"
        "tilematrixset=BPL72VL"
    )

    if input_format == "VRT":
        # Create a virtual raster file from the WMTS
        input_path = tmp_path / "input.vrt"
        gdal.Translate(
            input_path,
            wmts_url,
            options=gdal.TranslateOptions(
                format="VRT", projWin=[174816, 176672, 175136, 176352]
            ),
        )
    elif input_format == "GTiff":
        # Create a tif file from the WMTS
        input_path = tmp_path / "input.tif"
        gdal.Translate(
            input_path,
            wmts_url,
            options=gdal.TranslateOptions(
                format="GTiff", projWin=[174816, 176672, 175136, 176352]
            ),
        )

    # Define the bounding box and projection window
    bbox = [174816, 176352, 175136, 176672]
    projwin = [bbox[0], bbox[3], bbox[2], bbox[1]]

    # Create a temporary directory
    output_path = tmp_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # First methode (fails when input_format is VRT)
    # There is a shift of about one pixel in the output
    output_1_path = output_path / "output_1.tif"
    gdal.Translate(
        output_1_path,
        input_path,
        options=gdal.TranslateOptions(
            projWin=projwin,
            resampleAlg="nearest",
            width=1280,
            height=1280,
        ),
    )

    # Second methode (works for both input_format)
    # There is no shift in the output
    # Here we first create a temporary tif file with the correct projection window
    # and then create the final output with the correct size
    gdal.Translate(
        output_path / "temp.tif",
        input_path,
        options=gdal.TranslateOptions(projWin=projwin),
    )
    output_2_path = output_path / "output_2.tif"
    gdal.Translate(
        output_2_path,
        output_path / "temp.tif",
        options=gdal.TranslateOptions(
            resampleAlg="nearest",
            width=1280,
            height=1280,
        ),
    )
