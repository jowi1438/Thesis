#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel-1 SLC search and download — time series version.

Filters field geometries to an AOI, computes a convex hull, searches the
Copernicus Dataspace for Sentinel-1 SLC scenes, refines the inventory to
a specific ascending track, and downloads the scenes via ASF.

Usage:
    python search_download_timeseries_args.py \
        --fields_file /home/johan/Thesis/Sentinel_1/ost/s1/S1_Search_Download/preprocessed_field_geometries_skane_crop4.parquet \
        --aoi_shp     /home/johan/Thesis/Sentinel_1/ost/s1/Example_Fields/examplefields.shp \
        --out_parquet /home/johan/Thesis/Sentinel_1/ost/s1/Example_Fields/example_fields.parquet \
        --hull_shp    /home/johan/Thesis/Sentinel_1/ost/s1/convex_hull.shp \
        --project_dir /home/johan/OST_Search_Download/Download_S1_SLC \
        --start       2024-06-01 \
        --end         2024-09-01 \
        --track       73

Arguments:
    --fields_file   Path to preprocessed field geometries GeoParquet
    --aoi_shp       Path to AOI shapefile used to clip fields
    --out_parquet   Path to save the filtered fields GeoParquet
    --hull_shp      Path to save the convex hull shapefile (for inspection in ArcGIS)
    --project_dir   OST project directory for inventory and download output
    --start         Start date for scene search (YYYY-MM-DD), default: 2024-06-01
    --end           End date for scene search (YYYY-MM-DD), default: 2024-09-01
    --track         Relative orbit number to filter on, default: 73
"""

import argparse
import geopandas as gpd
from urllib.parse import quote
from pathlib import Path
from pprint import pprint
from ost import Generic, Sentinel1
import pylab


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Search and download Sentinel-1 SLC time series",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fields_file", required=True,
                        help="Path to preprocessed field geometries GeoParquet")
    parser.add_argument("--aoi_shp",     required=True,
                        help="Path to AOI shapefile used to clip fields")
    parser.add_argument("--out_parquet", required=True,
                        help="Path to save the filtered fields GeoParquet")
    parser.add_argument("--hull_shp",    required=True,
                        help="Path to save the convex hull shapefile")
    parser.add_argument("--project_dir", required=True,
                        help="OST project directory for inventory and download output")
    parser.add_argument("--start",       default="2024-06-01",
                        help="Start date for scene search (YYYY-MM-DD)")
    parser.add_argument("--end",         default="2024-09-01",
                        help="End date for scene search (YYYY-MM-DD)")
    parser.add_argument("--track",       type=int, default=73,
                        help="Relative orbit number to filter on")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    #----------------------------
    # Area of interest
    #----------------------------
    fields = gpd.read_parquet(args.fields_file).to_crs(4326)
    skane  = gpd.read_file(args.aoi_shp).to_crs(4326)

    # keep only fields within the AOI boundary
    fields_filtered = gpd.clip(fields, skane).reset_index(drop=True)
    fields_filtered.to_parquet(args.out_parquet)

    print(f"Original: {len(fields)} fields")
    print(f"Filtered: {len(fields_filtered)} fields")

    # convex hull of filtered fields → used as AOI for S1 search
    hull_geom = fields_filtered.geometry.unary_union.convex_hull
    aoi       = hull_geom.wkt
    aoi_enc   = quote(aoi)

    # export convex hull for inspection in ArcGIS
    gpd.GeoDataFrame(geometry=[hull_geom], crs=4326).to_file(args.hull_shp)

    #----------------------------
    # Time of interest
    #----------------------------
    start = args.start
    end   = args.end

    #----------------------------
    # Project folder
    #----------------------------
    project_dir = Path(args.project_dir)

    print('AOI: ',               aoi)
    print('TOI start: ',         start)
    print('TOI end: ',           end)
    print('Project Directory: ', project_dir)

    # OST Generic class — sets up folder structure
    ost_generic = Generic(
        project_dir=project_dir,
        aoi=aoi,
        start=start,
        end=end,
    )

    print('\n Before customisation')
    print('---------------------------------------------------------------------')
    pprint(ost_generic.config_dict)
    print('---------------------------------------------------------------------')

    ost_generic.config_dict['download_dir'] = '/download'
    ost_generic.config_dict['temp_dir']     = '/tmp'

    print('\n After customisation (note the change in download_dir and temp_dir)')
    print('---------------------------------------------------------------------')
    pprint(ost_generic.config_dict)

    #----------------------------
    # Sentinel-1 search
    #----------------------------
    ost_s1 = Sentinel1(
        project_dir=project_dir,
        aoi=aoi,
        start=start,
        end=end,
        product_type='SLC',
        beam_mode='IW',
        polarisation='VV VH',
    )

    ost_s1.search()

    # DEBUG
    print("DEBUG: inventory is None?", ost_s1.inventory is None)
    if ost_s1.inventory is not None:
        print("DEBUG: inventory rows:",    len(ost_s1.inventory))
        print("DEBUG: inventory columns:", ost_s1.inventory.columns.tolist())
    else:
        print("DEBUG: inventory_file exists?",
              ost_s1.inventory_file.exists() if ost_s1.inventory_file else "file is None")

    ost_s1.plot_inventory(transparency=.1)

    print('-----------------------------------------------------------------------------------------------------------')
    print(' INFO: We found a total of {} products for our project definition'.format(len(ost_s1.inventory)))
    print('-----------------------------------------------------------------------------------------------------------')
    print('')
    print('-----------------------------------------------------------------------------------------------------------')
    print('The columns of our inventory:')
    print('')
    print(ost_s1.inventory.columns)
    print('-----------------------------------------------------------------------------------------------------------')
    print('')
    print('-----------------------------------------------------------------------------------------------------------')
    print(' The last 5 rows of our inventory:')
    print(ost_s1.inventory.tail(5))

    #----------------------------
    # Refine and filter inventory
    #----------------------------
    ost_s1.refine_inventory()

    pylab.rcParams['figure.figsize'] = (19, 19)

    key           = 'ASCENDING_VVVH'
    ascending_all = ost_s1.refined_inventory_dict[key]
    ost_s1.plot_inventory(ascending_all, 0.1)

    # Filter to specified relative orbit (track)
    ascending_filtered = ascending_all[
        ascending_all['relativeorbit'] == args.track
    ].reset_index(drop=True)

    print(f"\nAscending scenes after track {args.track} filter: {len(ascending_filtered)}")
    print(ascending_filtered[['identifier', 'acquisitiondate', 'relativeorbit', 'orbitdirection']])

    #----------------------------
    # Download
    #----------------------------
    ost_s1.download(ascending_filtered, concurrent=10)


if __name__ == "__main__":
    main()