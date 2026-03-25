import geopandas as gpd
from urllib.parse import quote 

# this imports we need to handle the folders, independent of the OS
from pathlib import Path
from pprint import pprint

# this is the Generic class, that basically handles all the workflow from beginning to the end
from ost import Generic

import pylab

#----------------------------
# Area of interest
#----------------------------
# load fields
fields = gpd.read_parquet("/home/johan/Thesis/Sentinel_1/ost/s1/S1_Search_Download/preprocessed_field_geometries_skane.parquet").to_crs(4326)

# load skane boundary
skane = gpd.read_file("/home/johan/Thesis/Sentinel_1/ost/s1/Example_Fields/examplefields.shp").to_crs(4326)

# keep only fields within the boundary
fields_filtered = gpd.clip(fields, skane)
fields_filtered = fields_filtered.reset_index(drop=True)
fields_filtered.to_parquet("/home/johan/Thesis/Sentinel_1/ost/s1/Example_Fields/example_fields.parquet")

print(f"Original: {len(fields)} fields")
print(f"Filtered: {len(fields_filtered)} fields")

# choose union or convex hull
hull_geom = fields_filtered.geometry.unary_union.convex_hull
aoi = hull_geom.wkt
aoi_enc = quote(aoi)

# export convex hull for inspection in ArcGIS
hull_gdf = gpd.GeoDataFrame(geometry=[hull_geom], crs=4326)
hull_gdf.to_file("/home/johan/Thesis/Sentinel_1/ost/s1/convex_hull.shp")

#----------------------------
# Time of interest
#----------------------------
# we set only the start date to today - 30 days
start = '2024-06-01'
end = '2024-09-01'

#----------------------------
# Project folder
#----------------------------

# get home folder
home = Path.home()

# create a processing directory
project_dir = home.joinpath('OST_Search_Download', 'Download_S1_SLC')

#------------------------------
# Print out AOI and start date
#------------------------------
print('AOI: ', aoi)
print('TOI start: ', start)
print('TOI end: ', end)
print('Project Directory: ', project_dir)

# create an OST Generic class instance
ost_generic = Generic(
    project_dir=project_dir,
    aoi=aoi, 
    start=start, 
    end=end
)

# Uncomment below to see the list of folders inside the project directory (UNIX only):
print('')
print('We use the linux ls command for listing the directories inside our project folder:')
#ls {project_dir}

# Default config as created by the class initialisation
print(' Before customisation')
print('---------------------------------------------------------------------')
pprint(ost_generic.config_dict)
print('---------------------------------------------------------------------')

# customisation
ost_generic.config_dict['download_dir'] = '/download'
ost_generic.config_dict['temp_dir'] = '/tmp'

print('')
print(' After customisation (note the change in download_dir and temp_dir)')
print('---------------------------------------------------------------------')
pprint(ost_generic.config_dict)

# the import of the Sentinel1 class
from ost import Sentinel1

# initialize the Sentinel1 class
ost_s1 = Sentinel1(
    project_dir=project_dir,
    aoi=aoi, 
    start=start, 
    end=end,
    product_type='SLC',
    beam_mode='IW',
    polarisation='VV VH',
)

"""
#---------------------------------------------------
# for plotting purposes we use this iPython magic
%matplotlib inline
%pylab inline
pylab.rcParams['figure.figsize'] = (13, 13)
#---------------------------------------------------
"""
# search command
ost_s1.search()

# DEBUG: check if inventory was loaded
print("DEBUG: inventory is None?", ost_s1.inventory is None)
if ost_s1.inventory is not None:
    print("DEBUG: inventory rows:", len(ost_s1.inventory))
    print("DEBUG: inventory columns:", ost_s1.inventory.columns.tolist())
else:
    print("DEBUG: inventory_file exists?", ost_s1.inventory_file.exists() if ost_s1.inventory_file else "file is None")

# uncomment in case you have issues with the registration procedure 
#ost_s1.search(base_url='https://scihub.copernicus.eu/dhus')

# we plot the full Inventory on a map
ost_s1.plot_inventory(transparency=.1)

print('-----------------------------------------------------------------------------------------------------------')
print(' INFO: We found a total of {} products for our project definition'.format(len(ost_s1.inventory)))
print('-----------------------------------------------------------------------------------------------------------')
print('')
# combine OST class attribute with pandas head command to print out the first 5 rows of the 
print('-----------------------------------------------------------------------------------------------------------')
print('The columns of our inventory:')
print('')
print(ost_s1.inventory.columns)
print('-----------------------------------------------------------------------------------------------------------')

print('')
print('-----------------------------------------------------------------------------------------------------------')
print(' The last 5 rows of our inventory:')
print(ost_s1.inventory.tail(5))

ost_s1.refine_inventory()

pylab.rcParams['figure.figsize'] = (19, 19)

#Filter inventory for a specific product key, e.g. 'ASCENDING_VVVH', relative orbit (track) 73 and plot the inventory for that specific product key and track
key = 'ASCENDING_VVVH'
ascending_all = ost_s1.refined_inventory_dict[key]
ost_s1.plot_inventory(ost_s1.refined_inventory_dict[key], 0.1)
ascending_t73 = ascending_all[ascending_all['relativeorbit'] == 73].reset_index(drop=True)

ost_s1.download(ascending_t73, concurrent=10)