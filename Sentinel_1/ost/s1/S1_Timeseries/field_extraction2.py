#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
field_extraction2.py — FINAL VERSION
================================================================================

WHAT IT DOES
------------
For each field and each dated S1 mosaic, reprojects all S1 products to the
field mask grid and writes output matching Sebastian's format:

  {field_id}/
    ├── geom_{field_id}.parquet        ← Field geometry
    ├── {field_id}_{YYYYMMDD}.tif      ← 6-band S1 raster (all pixels)
    └── mask_{field_id}.tif            ← Separate mask (uint8: 0/1)

Output raster: 6 bands (float32):
  Band 1: VV          — gamma0 backscatter (dB)
  Band 2: VH          — gamma0 backscatter (dB)
  Band 3: Alpha       — normalised 0-1
  Band 4: Anisotropy  — normalised 0-1
  Band 5: Entropy     — normalised 0-1
  Band 6: DpRVI       — normalised 0-1

When concatenated with S2 [B2,B3,B4,B8,B11,B12] the full 12-band input is:
  [VV, VH, Alpha, Anisotropy, Entropy, DpRVI, B2, B3, B4, B8, B11, B12]

Also writes a single metadata.json to the dataset root:
  {"timestamps": ["2024-06-07", "2024-06-19", ...]}

Only dates where ALL three products (bs, pol, dprvi) are available are
included in the output and in metadata.json.

USAGE
-----
  # Example with Hannicat's paths:
  python field_extraction2.py \
    --fields_file /home/johan/Thesis/Sentinel_1/ost/s1/Example_Fields/example_fields.parquet \
    --dataset_path /home/johan/Thesis/Sentinel_1/ost/s1/S1_Timeseries/dataset \
    --s1_root /home/johan/OST_processing/out_timeseries \
    --id_field field_id \
    --overwrite

REQUIREMENTS
------------
  pip install rasterio geopandas numpy shapely tqdm

OUTPUT
------
  dataset/metadata.json
  dataset/data/{field_id}/geom_{field_id}.parquet
  dataset/data/{field_id}/{field_id}_{YYYYMMDD}.tif   (6 bands, float32)
  dataset/data/{field_id}/mask_{field_id}.tif         (1 band, uint8)
================================================================================
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import box
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Fixed band order
# ---------------------------------------------------------------------------
# Each entry is (product_key, 1-indexed band in that product's mosaic TIF).
# pol band order verified with rasterio.descriptions:
#   band 1 = Alpha, band 2 = Anisotropy, band 3 = Entropy

PRODUCT_BAND_MAP = [
    ("bs",    1),   # Band 1 → VV
    ("bs",    2),   # Band 2 → VH
    ("pol",   1),   # Band 3 → Alpha
    ("pol",   2),   # Band 4 → Anisotropy
    ("pol",   3),   # Band 5 → Entropy
    ("dprvi", 1),   # Band 6 → DpRVI
]

BAND_NAMES = ["VV", "VH", "Alpha", "Anisotropy", "Entropy", "DpRVI"]
N_BANDS    = len(PRODUCT_BAND_MAP)   # 6


# ---------------------------------------------------------------------------
# Reproject one source band onto the target grid
# ---------------------------------------------------------------------------

def reproject_band(src_path: Path, band_idx: int,
                   target_transform, target_crs,
                   target_height: int, target_width: int) -> np.ndarray:
    """
    Reproject a single band from src_path onto the target grid.

    Args:
        src_path:         source GeoTIFF
        band_idx:         1-indexed band number to read
        target_transform: affine transform of the target grid
        target_crs:       CRS of the target grid
        target_height:    pixel height of target grid
        target_width:     pixel width of target grid

    Returns:
        (target_height, target_width) float32 array, NaN where no data
    """
    with rasterio.open(src_path) as src:
        src_data  = src.read(band_idx)
        src_nodata = src.nodata
        resampled = np.full(
            (target_height, target_width), np.nan, dtype=np.float32
        )
        reproject(
            source        = src_data,
            destination   = resampled,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = target_transform,
            dst_crs       = target_crs,
            resampling    = Resampling.bilinear,
            src_nodata    = src_nodata,
            dst_nodata    = np.nan,
        )
    return resampled


# ---------------------------------------------------------------------------
# Stack all S1 bands into one GeoTIFF + save mask + save geometry
# ---------------------------------------------------------------------------

def extract_field(product_files: dict, mask_file: Path, out_file: Path,
                  field_geom, field_crs: str):
    """
    Reproject and stack all 6 S1 bands into a single GeoTIFF aligned to
    the field mask grid. Also writes mask file and geometry parquet.

    Follows Sebastian's output format:
      - Raster: all pixels (no masking applied during extraction)
      - Mask: separate uint8 file (mask_{field_id}.tif)
      - Geometry: GeoDataFrame as parquet (geom_{field_id}.parquet)

    Args:
        product_files: dict  product_key -> Path to mosaic TIF
                       e.g. {'bs': Path(...), 'pol': Path(...), 'dprvi': Path(...)}
        mask_file:     Path to field mask GeoTIFF (defines the target grid)
        out_file:      Path for output 6-band stacked GeoTIFF
        field_geom:    Shapely geometry of the field
        field_crs:     CRS of the field geometry
    """
    with rasterio.open(mask_file) as mask_src:
        target_transform = mask_src.transform
        target_height    = mask_src.height
        target_width     = mask_src.width
        target_crs       = mask_src.crs
        target_profile   = mask_src.profile.copy()
        mask_data        = mask_src.read(1)  # Read mask array

    output = np.full(
        (N_BANDS, target_height, target_width), np.nan, dtype=np.float32
    )

    for out_band_idx, (product_key, src_band_idx) in enumerate(PRODUCT_BAND_MAP):
        if product_key not in product_files:
            raise ValueError(
                f"Missing product '{product_key}' needed for output "
                f"band {out_band_idx + 1} ({BAND_NAMES[out_band_idx]})"
            )

        band_data = reproject_band(
            src_path         = product_files[product_key],
            band_idx         = src_band_idx,
            target_transform = target_transform,
            target_crs       = target_crs,
            target_height    = target_height,
            target_width     = target_width,
        )
        # DON'T apply mask here — keep all pixels like Sebastian's approach
        output[out_band_idx] = band_data

    # ── Write stacked GeoTIFF (all pixels) ──
    target_profile.update({
        "driver":    "GTiff",
        "height":    target_height,
        "width":     target_width,
        "transform": target_transform,
        "crs":       target_crs,
        "count":     N_BANDS,
        "dtype":     "float32",
        "nodata":    np.nan,
        "compress":  "deflate",
    })
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_file, "w", **target_profile) as dst:
        dst.write(output)
        for i, name in enumerate(BAND_NAMES, start=1):
            dst.update_tags(i, name=name)

    # ── Write separate mask file (uint8: 0/1) ──
    field_id = out_file.stem.split("_")[0]
    mask_path = out_file.parent / f"mask_{field_id}.tif"  # ← CORRECTED
    if not mask_path.exists():
        mask_profile = target_profile.copy()
        mask_profile.update({
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
        })
        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write((mask_data > 0).astype(np.uint8), 1)

    # ── Write geometry as parquet ──
    geom_path = out_file.parent / f"geom_{field_id}.parquet"
    if not geom_path.exists():
        gdf_field = gpd.GeoDataFrame(
            {'geometry': [field_geom]},
            crs=field_crs
        )
        gdf_field.to_parquet(geom_path)


# ---------------------------------------------------------------------------
# Discover dated mosaics
# ---------------------------------------------------------------------------

def discover_dated_mosaics(merged_dir: Path, product_key: str) -> dict:
    """
    Find all per-date mosaics for a product key.

    Dated mosaics follow the naming convention:
      YYYYMMDD_{product_key}_SWEREF99TM_10m.tif

    Returns:
        dict: date_str (YYYYMMDD) -> Path, sorted by date
    """
    pattern   = f"*_{product_key}_SWEREF99TM_10m.tif"
    all_files = sorted(merged_dir.glob(pattern))

    dated = {}
    for f in all_files:
        prefix = f.stem.split("_")[0]
        if prefix.isdigit() and len(prefix) == 8:
            dated[prefix] = f

    return dict(sorted(dated.items()))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_metadata_json(out_path: Path, date_strings: list):
    """
    Write metadata.json with sorted ISO timestamps.

    Uses lowercase key 'timestamps' to match crop_dataset.py:
      meta['timestamps']
    """
    timestamps = sorted(
        datetime.strptime(d, "%Y%m%d").strftime("%Y-%m-%d")
        for d in date_strings
    )
    with open(out_path, "w") as f:
        json.dump({"timestamps": timestamps}, f, indent=2)
    print(f"  Written: {out_path}  ({len(timestamps)} timestamps)")
    for ts in timestamps:
        print(f"    {ts}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="S1 time series -> per-field extraction (Sebastian format)"
    )
    p.add_argument("--fields_file",  required=True,
                   help="GeoParquet with field polygons")
    p.add_argument("--dataset_path", required=True,
                   help="Root dataset folder")
    p.add_argument("--s1_root",      required=True,
                   help="out_root used when running S1 processing scripts")
    p.add_argument("--id_field",     default=None,
                   help="Column name to use as field ID")
    p.add_argument("--crs",          default="EPSG:3006",
                   help="Target CRS (default: EPSG:3006 SWEREF99TM)")
    p.add_argument("--overwrite",    action="store_true",
                   help="Reprocess and overwrite existing output TIFs")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    s1_root      = Path(args.s1_root)
    merged_dir   = s1_root / "merged_SWEREF99TM_10m"

    if not merged_dir.exists():
        raise FileNotFoundError(
            f"merged_SWEREF99TM_10m not found at {merged_dir}\n"
            f"Run processing scripts first."
        )

    # ── Discover dated mosaics ─────────────────────────────────────────────
    print("\n── Discovering dated mosaics ──")
    product_keys    = ["bs", "pol", "dprvi"]
    product_mosaics = {}

    for key in product_keys:
        mosaics = discover_dated_mosaics(merged_dir, key)
        product_mosaics[key] = mosaics
        if mosaics:
            print(f"  {key:6s}: {len(mosaics)} dates")
        else:
            print(f"  {key:6s}: [WARN] no dated mosaics found")

    # Only process dates where ALL three products are present
    sets_of_dates = [set(v.keys()) for v in product_mosaics.values() if v]
    if not sets_of_dates:
        raise RuntimeError("No dated mosaics found for any product.")

    complete_dates = sorted(set.intersection(*sets_of_dates))
    all_dates      = set.union(*sets_of_dates)
    missing_dates  = all_dates - set(complete_dates)

    if missing_dates:
        print(f"\n  [WARN] {len(missing_dates)} date(s) missing at least one product")
        print(f"         Skipped: {sorted(missing_dates)}")

    print(f"\n  Complete dates (all 3 products): {len(complete_dates)}")
    for d in complete_dates:
        print(f"    {d}")

    if not complete_dates:
        raise RuntimeError(
            "No dates have all three products (bs, pol, dprvi)."
        )

    # ── Write metadata.json ────────────────────────────────────────────────
    print("\n── Writing metadata.json ──")
    write_metadata_json(
        out_path     = dataset_path / "metadata.json",
        date_strings = complete_dates,
    )

    # ── Load fields ────────────────────────────────────────────────────────
    print("\n── Loading fields ──")
    fields = gpd.read_parquet(args.fields_file).to_crs(args.crs)
    print(f"  {len(fields)} fields in {args.crs}")

    # ── Per-field extraction ───────────────────────────────────────────────
    print("\n── Extracting per-field S1 stacks ──")
    print(f"  Band order: {' | '.join(BAND_NAMES)}")
    print(f"  Output format (Sebastian style):")
    print(f"    {{field_id}}/")
    print(f"      ├── {{field_id}}_{{YYYYMMDD}}.tif  (6 bands)")
    print(f"      ├── mask_{{field_id}}.tif")
    print(f"      └── geom_{{field_id}}.parquet\n")

    n_ok = n_skip = n_err = 0

    for i, field in tqdm(enumerate(fields.itertuples()),
                         total=len(fields), desc="Fields"):
        field_id   = (
            str(getattr(field, args.id_field))
            if args.id_field else f"field{i:05d}"
        )
        field_path = dataset_path / "data" / field_id
        mask_file  = field_path / f"mask_{field_id}.tif"

        if not mask_file.exists():
            tqdm.write(f"[WARN] No mask for {field_id} — skipping")
            n_err += 1
            continue

        for date_str in complete_dates:
            # New naming: field_id_YYYYMMDD.tif
            out_file = field_path / f"{field_id}_{date_str}.tif"

            if out_file.exists() and not args.overwrite:
                n_skip += 1
                continue

            # Build product_key -> mosaic Path dict for this date
            product_files = {
                key: product_mosaics[key][date_str]
                for key in product_keys
            }

            try:
                extract_field(
                    product_files = product_files,
                    mask_file     = mask_file,
                    out_file      = out_file,
                    field_geom    = field.geometry,
                    field_crs     = args.crs,
                )
                n_ok += 1
            except Exception as e:
                tqdm.write(f"[ERROR] {field_id} / {date_str}: {e}")
                n_err += 1

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n── Done ──")
    print(f"  Written:  {n_ok}")
    print(f"  Skipped:  {n_skip}  (use --overwrite to redo)")
    print(f"  Errors:   {n_err}")
    print(f"\n  Output: {dataset_path / 'data'}")


if __name__ == "__main__":
    main()