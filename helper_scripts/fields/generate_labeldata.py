"""
Convert models in hdf5 file format to tf savedmodel.
"""

from pathlib import Path

import geopandas as gpd
import shapely


# Set some variables
output_dir = Path("X:/Monitoring/orthoseg/boundaries")
locations_name = (
    "boundaries_BEFL-TEST=s2=2023-06-05=2023-06-11=B08-B04-B03_locations.gpkg"
)
locations_path = output_dir / locations_name
polygons_name = (
    "boundaries_BEFL-TEST=s2=2023-06-05=2023-06-11=B08-B04-B03_polygons.gpkg"
)
polygons_path = output_dir / polygons_name

input_poly_path = (
    Path("X:/Monitoring/OrthoSeg/boundaries/input_vector") / "perc_2023_bufm10.gpkg"
)
roi_path = (
    Path("X:/GIS/GIS DATA/Gewesten/2019-01-01_geopunt/Vlaanderen")
    / "refgew_vlaanderen.gpkg"
)
nb_locations = 1500
width = 640
height = width
output_crs = "epsg:32631"

# Start processing
print("Read input files")
polygons_gdf = gpd.read_file(input_poly_path, engine="pyogrio")
for col in polygons_gdf.columns:
    if col != "geometry":
        polygons_gdf.drop(columns=[col], inplace=True)
polygons_gdf["label_name"] = "parcel"
polygons_gdf["description"] = ""
polygons_gdf.geometry = polygons_gdf.geometry.to_crs(output_crs)
assert isinstance(polygons_gdf, gpd.GeoDataFrame)

roi_gdf = gpd.read_file(roi_path, engine="pyogrio")

print("Determine random points")
random_points_gdf = gpd.GeoDataFrame(
    geometry=roi_gdf.geometry.sample_points(nb_locations, method="uniform").to_list(),
    crs=roi_gdf.crs,
)  # type: ignore
random_points_gdf.geometry = random_points_gdf.geometry.to_crs(output_crs)
random_points = random_points_gdf.geometry[0]

locations = []
nb_train = nb_locations * 0.9
for idx, point in enumerate(random_points.geoms):
    geometry = shapely.box(
        xmin=point.x, ymin=point.y, xmax=point.x + width, ymax=point.y + height
    )
    traindata_type = "train" if idx < nb_train else "validation"
    locations.append(
        {"geometry": geometry, "traindata_type": traindata_type, "description": ""}
    )

locations_gdf = gpd.GeoDataFrame(locations)

print("Write output files")
locations_gdf.to_file(locations_path, layer="locations", engine="pyogrio")
polygons_gdf.to_file(polygons_path, layer="polygons", engine="pyogrio")
