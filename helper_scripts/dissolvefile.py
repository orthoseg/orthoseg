import geopandas as gpd
from datetime import datetime

input_filepath = "X:\\Monitoring\\Markers\\PlayGround\\PIEROG\\inputdata\\Prc_BEFL_2018_2018-08-02.shp"
output_filepath = "X:\\Monitoring\\Markers\\PlayGround\\PIEROG\\inputdata\\Prc_BEFL_2018_2018-08-02_diss.shp"

print(f"read start: {datetime.now()}")
input_gdf = gpd.read_file(input_filepath)
print(f"read ready: {datetime.now()}, columns: {input_gdf.columns}")
print(f"dissolve start: {datetime.now()}")
diss_gdf = input_gdf.dissolve(by='GWSCOD_H')
print(f"dissolve stop: {datetime.now()}")
print(f"write start: {datetime.now()}")
diss_gdf.to_file(output_filepath)
print(f"write ready: {datetime.now()}")
