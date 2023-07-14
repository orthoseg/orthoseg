from datetime import datetime
import logging
from pathlib import Path

from osgeo import gdal
import openeo
import openeo.processes
import pyproj


def s2_for_fields():

    # Init logging
    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger(__name__)

    # Set some variables
    roi_name = "BEFL"
    roi = {"west": 469110.0, "south": 5613690.0, "east": 707210.0, "north": 5709000.0}
    # roi_name = "BEFL-TEST"
    # roi = {"west": 484500.0, "south": 5643000.0, "east": 499500.0, "north": 5651000.0}
    # roi_name = "BEFL-TEST2"
    # roi = {"west": 484500.0, "south": 5643000.0, "east": 485500.0, "north": 5644000.0}

    roi_crs = "epsg:32631"
    collection = "TERRASCOPE_S2_TOC_V2"
    max_cloud_cover = 30
    output_dir = Path("X:/Monitoring/orthoseg/fields/input_raster")
    start_date = datetime(year=2023, month=5, day=1)
    end_date = datetime(year=2023, month=7, day=1)
    bands = ["B08", "B04", "B03"]
    time_dimension_reducers = ["min"]
    rescale = False

    job_options = {
        "executor-memory": "4G",
        "executor-memoryOverhead": "2G",
        "executor-cores": "2",
    }

    # Check and process input
    # -----------------------
    if end_date > datetime.now():
        print(f"end_date is in the future: {end_date}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Prepare output path
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    bands_str = "-".join(bands)
    name = (
        f"{roi_name}-s2_{start_date_str}_{end_date_str}_{bands_str}_"
        f"{'-'.join(time_dimension_reducers)}.tif"
    )
    uint16_path = output_dir / name
    output_path = uint16_path.with_stem(f"{uint16_path.stem}_byte")
    if output_path.exists():
        raise ValueError(f"Output path exists already: {output_path}")

    # Generate image on openeo server
    # -------------------------------
    if not uint16_path.exists():
        # Convert roi to WGS84 if it is not epsg:4326 already
        if pyproj.CRS(roi_crs).to_epsg() != 4326:
            transformer = pyproj.Transformer.from_crs(
                roi_crs, "epsg:4326", always_xy=True
            )
            roi["west"], roi["south"] = transformer.transform(roi["west"], roi["south"])
            roi["east"], roi["north"] = transformer.transform(roi["east"], roi["north"])

        # Connect with VITO openeo backend
        conn = openeo.connect("https://openeo.vito.be").authenticate_oidc(
            provider_id="egi"
        )

        bands_to_load = list(bands)
        scl_band_name = "SCL"
        bands_to_load.append(scl_band_name)

        # Load cube of relevant images
        # You can look which layers are available here:
        # https://openeo.cloud/data-collections/
        cube = conn.load_collection(
            collection,
            spatial_extent=roi,
            temporal_extent=[start_date_str, end_date_str],  # type: ignore
            bands=bands_to_load,
            max_cloud_cover=max_cloud_cover,
        )

        # Use mask_scl_dilation for "aggressive" cloud mask based on SCL
        if scl_band_name is not None:
            cube = cube.process(
                "mask_scl_dilation",
                data=cube,
                scl_band_name=scl_band_name,
            )

        cube = cube.filter_bands(bands=bands)

        # Apply time reducer(s)
        cube_reduced = None
        for reducer in time_dimension_reducers:
            reduced = cube.reduce_dimension(dimension="t", reducer=reducer)
            reduced = reduced.rename_labels(
                dimension="bands",
                source=bands,
                target=[f"{band}-{reducer}" for band in bands],
            )
            if cube_reduced is None:
                cube_reduced = reduced
            else:
                cube_reduced = cube_reduced.merge_cubes(reduced)

        # Reducing finished, so consolidate to cube
        cube = cube_reduced
        assert cube is not None

        if rescale:
            cube_rescaled = None
            for band in [band.name for band in cube.metadata.band_dimension.bands]:
                if (
                    band.startswith("B02")
                    or band.startswith("B03")
                    or band.startswith("B04")
                ):
                    band_rescaled = cube.filter_bands(band).linear_scale_range(
                        0, 2000, 0, 255
                    )
                elif band.startswith("B08"):
                    band_rescaled = cube.filter_bands(band).linear_scale_range(
                        0, 5000, 0, 255
                    )
                else:
                    raise ValueError(f"rescale not implemented for band {band}")

                # Add (rescaled) band to new cube
                if cube_rescaled is None:
                    cube_rescaled = band_rescaled
                else:
                    cube_rescaled = cube_rescaled.merge_cubes(band_rescaled)

            # Rescaling finished
            cube = cube_rescaled
            assert cube is not None

        # Result should be saved as GeoTiff
        result = cube.save_result(format="GTiff")

        job = result.create_job(title=uint16_path.name, job_options=job_options)
        job.start_and_wait()
        job.get_results().download_file(target=uint16_path)
        job.delete_job()

    # Rescale image from uint16 to byte
    # ---------------------------------
    params = gdal.TranslateOptions(
        outputType=gdal.GDT_Byte,
        bandList=[1, 2, 3],
        widthPct=100,
        heightPct=100,
        creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2"],
        scaleParams=[(0, 5000, 0, 255), (0, 2000, 0, 255), (0, 2000, 0, 255)],
        noData="none",
    )
    ds_input = gdal.Open(str(uint16_path))
    gdal.Translate(str(output_path), ds_input, options=params)


if __name__ == "__main__":
    s2_for_fields()
