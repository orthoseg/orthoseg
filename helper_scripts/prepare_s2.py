from pathlib import Path
from osgeo import gdal

s2_path = Path("//dg3.be/alp/Datagis/satellite_periodic/BEFL/s2-agri/")
s2_path = s2_path / "s2-agri_2023-06-05_2023-06-11_B02-B03-B04-B08-B11-B12.tif"

output_dir = Path("X:/Monitoring/OrthoSeg/boundaries/input_raster")
roi_xmin = 484500
roi_ymin = 5643000
roi_bounds = (roi_xmin, roi_ymin + 8000, roi_xmin + 15000, roi_ymin)

if roi_bounds is None:
    output_name = f"BEFL_{s2_path.stem}.tif"
else:
    output_name = f"BEFL-TEST_{s2_path.stem}.tif"

output_name = output_name.replace("_B02-B03-B04-B08-B11-B12.", "_B08-B04-B03.")
output_name = output_name.replace("s2-agri_", "s2_")
output_path = output_dir / output_name
params = gdal.TranslateOptions(
    outputType=gdal.GDT_Byte,
    bandList=[4, 3, 2],
    widthPct=100,
    heightPct=100,
    creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=2"],
    projWin=roi_bounds,
    scaleParams=[(100, 5050, 1, 255), (150, 2150, 1, 255), (250, 1800, 1, 255)],
    noData="none",
)
ds_input = gdal.Open(str(s2_path))
gdal.Translate(str(output_path), ds_input, options=params)
