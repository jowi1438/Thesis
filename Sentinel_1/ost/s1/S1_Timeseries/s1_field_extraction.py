#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel-1 field extraction — GeoTIFF output (time series version).

For each field and each dated S1 mosaic, produces GeoTIFFs aligned
to the field mask grid:
    s1_bs_<date>_<field_id>.tif       2 bands: VV, VH (gamma° in dB)
    s1_pol_<date>_<field_id>.tif      3 bands: H, A, Alpha
    s1_dprvi_<date>_<field_id>.tif    1 band:  DpRVI

A single metadata.json is written to the dataset root listing all
unique acquisition timestamps across all products as TIMESTAMPS.

Usage:
    python s1_field_extraction.py \
        --fields_file /home/johan/Thesis/Sentinel_1/ost/s1/Example_Fields/example_fields.parquet \
        --dataset_path /home/johan/Thesis/Sentinel_1/ost/s1/S1_Timeseries/dataset \
        --s1_root /home/johan/OST_processing/out_timeseries \
        --id_field field_id
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from shapely.geometry import box
import geopandas as gpd


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def calculate_intersection_area(large_bounds, small_bounds):
    large_geom = box(*large_bounds)
    small_geom = box(*small_bounds)
    return large_geom.intersection(small_geom).area


def get_intersection_mask(large_bounds, small_raster, small_transform):
    large_geom = box(*large_bounds)
    mask = geometry_mask(
        [large_geom],
        out_shape=(small_raster.height, small_raster.width),
        transform=small_transform,
        invert=True,
    )
    return mask


# ---------------------------------------------------------------------------
# Core reprojection function
# ---------------------------------------------------------------------------

def fill_from_multiple_rasters(s1_files, mask_file, out_file):
    """
    Reproject S1 mosaic(s) to the field mask grid and save as GeoTIFF.

    Parameters
    ----------
    s1_files : list[Path]
        One or more S1 mosaic GeoTIFFs covering the field bbox.
    mask_file : Path
        Field mask GeoTIFF produced by initialize_dataset.py.
    out_file : Path
        Output GeoTIFF path.
    """
    with rasterio.open(mask_file) as mask_src:
        target_transform = mask_src.transform
        target_height    = mask_src.height
        target_width     = mask_src.width
        target_crs       = mask_src.crs
        target_bounds    = mask_src.bounds
        target_profile   = mask_src.profile.copy()

    nodata_value = np.nan

    # Rank input files by intersection area (largest first)
    intersections = []
    for f in s1_files:
        with rasterio.open(f) as src:
            area = calculate_intersection_area(src.bounds, target_bounds)
            if area <= 0:
                continue
            intersections.append({"file": f, "area": area, "bounds": src.bounds})

    if not intersections:
        raise ValueError(f"No overlap between any S1 mosaic and field mask {mask_file}")

    intersections.sort(key=lambda x: x["area"], reverse=True)

    # Initialise output array
    with rasterio.open(intersections[0]["file"]) as first:
        band_count = first.count
        dtype      = first.dtypes[0]

    output_data = np.full(
        (band_count, target_height, target_width), nodata_value, dtype=dtype
    )
    filled_mask = np.zeros((target_height, target_width), dtype=bool)

    for item in intersections:
        with rasterio.open(item["file"]) as src:
            src_data  = src.read()
            resampled = np.empty(
                (src.count, target_height, target_width), dtype=dtype
            )
            reproject(
                source        = src_data,
                destination   = resampled,
                src_transform = src.transform,
                src_crs       = src.crs,
                dst_transform = target_transform,
                dst_crs       = target_crs,
                resampling    = Resampling.bilinear,
                src_nodata    = src.nodata,
                dst_nodata    = nodata_value,
            )

        with rasterio.open(mask_file) as mask_src:
            coverage_mask = get_intersection_mask(
                item["bounds"], mask_src, target_transform
            )
        pixels_to_fill = coverage_mask & ~filled_mask

        for b in range(band_count):
            output_data[b][pixels_to_fill] = resampled[b][pixels_to_fill]

        filled_mask |= pixels_to_fill

    total_pixels = target_height * target_width
    assert np.sum(filled_mask) == total_pixels, (
        f"Coverage incomplete: {np.sum(filled_mask)}/{total_pixels} pixels filled"
    )

    # Write GeoTIFF
    target_profile.update({
        "driver":    "GTiff",
        "height":    target_height,
        "width":     target_width,
        "transform": target_transform,
        "crs":       target_crs,
        "count":     band_count,
        "dtype":     dtype,
        "nodata":    nodata_value,
    })
    with rasterio.open(out_file, "w", **target_profile) as dst:
        dst.write(output_data)


# ---------------------------------------------------------------------------
# Discover dated mosaics
# ---------------------------------------------------------------------------

def discover_dated_mosaics(merged_dir, product_key):
    """
    Find all dated mosaics for a given product key.
    Returns list of (date_str, Path) tuples sorted by date.

    Dated mosaics have the format: YYYYMMDD_<product_key>_SWEREF99TM_10m.tif
    Scene-level files are excluded (they don't start with 8 digits).
    """
    pattern = f"*_{product_key}_SWEREF99TM_10m.tif"
    all_files = sorted(merged_dir.glob(pattern))

    dated = []
    for f in all_files:
        prefix = f.stem.split("_")[0]
        if prefix.isdigit() and len(prefix) == 8:
            dated.append((prefix, f))

    return dated


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def generate_metadata_json(out_path, product_mosaics):
    """
    Write a single metadata.json to the dataset root containing all
    unique acquisition timestamps across all products as TIMESTAMPS.
    """
    all_dates = set()
    for dated_mosaics in product_mosaics.values():
        for date_str, _ in dated_mosaics:
            all_dates.add(date_str)

    timestamps = sorted(
        datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d")
        for d in all_dates
    )

    metadata = {"TIMESTAMPS": timestamps}

    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Metadata written to {out_path} ({len(timestamps)} timestamps)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="S1 time series -> per-field GeoTIFFs aligned to field mask grid"
    )
    parser.add_argument("--fields_file",  required=True,
                        help="GeoParquet with field polygons")
    parser.add_argument("--dataset_path", required=True,
                        help="Root dataset folder (same as used for initialize_dataset.py)")
    parser.add_argument("--s1_root",      required=True,
                        help="out_root used when running the S1 processing scripts")
    parser.add_argument("--id_field",     default=None,
                        help="Column name to use as field ID")
    parser.add_argument("--crs",          default="EPSG:3006")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    s1_root      = Path(args.s1_root)
    merged_dir   = s1_root / "merged_SWEREF99TM_10m"

    if not merged_dir.exists():
        raise FileNotFoundError(f"merged_SWEREF99TM_10m not found at {merged_dir}")

    # Discover all dated mosaics per product
    product_keys = ["bs", "pol", "dprvi"]
    product_mosaics = {}
    for key in product_keys:
        mosaics = discover_dated_mosaics(merged_dir, key)
        if mosaics:
            product_mosaics[key] = mosaics
            print(f"Found {len(mosaics)} dated mosaics for '{key}': "
                  f"{[d for d, _ in mosaics]}")
        else:
            print(f"[WARN] No dated mosaics found for '{key}' in {merged_dir}")

    if not product_mosaics:
        raise RuntimeError("No dated mosaics found for any product. "
                           "Run the S1 processing scripts first.")

    # Write single metadata.json to dataset root
    generate_metadata_json(
        out_path=dataset_path / "metadata.json",
        product_mosaics=product_mosaics,
    )

    fields = gpd.read_parquet(args.fields_file).to_crs(args.crs)

    for i, field in tqdm(enumerate(fields.itertuples()), total=len(fields),
                         desc="Fields"):
        field_id   = (
            str(getattr(field, args.id_field)) if args.id_field else f"field{i}"
        )
        field_path = dataset_path / "data" / field_id
        mask_file  = field_path / f"mask_{field_id}.tif"

        if not mask_file.exists():
            print(f"[WARN] No mask found for {field_id}, skipping")
            continue

        for product_key, dated_mosaics in product_mosaics.items():
            for date_str, mosaic_file in dated_mosaics:
                out_file = field_path / f"s1_{product_key}_{date_str}_{field_id}.tif"

                if out_file.exists():
                    continue  # already processed

                try:
                    fill_from_multiple_rasters(
                        s1_files  = [mosaic_file],
                        mask_file = mask_file,
                        out_file  = out_file,
                    )
                except Exception as e:
                    print(f"[ERROR] {field_id} / {product_key} / {date_str}: {e}")