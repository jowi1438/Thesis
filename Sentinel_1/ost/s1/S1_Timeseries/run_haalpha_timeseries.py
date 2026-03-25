#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sentinel-1 SLC → H-A-Alpha decomposition
Per-burst → GeoTIFF → SWEREF99 TM 10m → per-scene mosaic → per-date mosaic.
Processes ALL zip files in the given scene directory.
Includes automatic retry: failed bursts are reset and reprocessed up to
--max_retries times.

Usage:
  python run_haalpha_timeseries.py \
    --scene_dir "/home/johan/OST_Search_Download/Download_S1_SLC/download/SAR/SLC/2024" \
    --out_root "/home/johan/OST_processing/out_timeseries" \
    --temp_dir "/home/johan/OST_processing/tmp" \
    --aoi "/home/johan/Thesis/Sentinel_1/ost/s1/Example_Fields/example_fields.parquet"
"""

import argparse
import json
import sys
import re
from pathlib import Path
from itertools import groupby
import pandas as pd
import rasterio
import numpy as np

from ost.s1.s1scene import Sentinel1Scene as S1Scene
from burst_to_ard_FIXED import burst_to_ard
from processing_utils import *

PRODUCT_KEY = "pol"
FINAL_PATTERN = "*pol_SWEREF99TM_10m.tif"
MARKER = ".pol.processed"


def normalize_entropy_band(tif_path):
    """
    Normalize Entropy band (band 0) from 0-100 to 0-1 in-place.
    Anisotropy (band 1) and Alpha (band 2) stay unchanged.
    """
    try:
        with rasterio.open(tif_path, 'r+') as src:
            entropy = src.read(1).astype(np.float32)
            entropy_normalized = entropy / 100.0
            src.write(entropy_normalized, 1)
        print(f"    ✓ Entropy normalized (0-100 → 0-1)")
    except Exception as e:
        print(f"    ⚠ Could not normalize entropy: {e}")


def extract_date_from_scene_id(scene_id: str) -> str:
    """
    Extract acquisition date (YYYYMMDD) from a Sentinel-1 scene ID.
    e.g. S1A_IW_SLC__1SDV_20240401T054512_... → '20240401'
    """
    match = re.search(r'_(\d{8})T\d{6}_', scene_id)
    if match:
        return match.group(1)
    match = re.search(r'(\d{8})', scene_id)
    if match:
        return match.group(1)
    return "unknown_date"


def process_bursts(burst_gdf, scene_id, scene_zip, out_root, config_file):
    """Process all bursts, return list of final tif paths."""
    n = len(burst_gdf)
    final_tifs = []

    for idx, row in burst_gdf.iterrows():
        date = str(row.get("Date", "unknown_date"))
        swath = str(row.get("SwathID", "IW"))
        burst_nr = int(row.get("BurstNr", idx + 1))
        simple_bid = f"{scene_id}_{swath}_B{burst_nr:03d}"
        prefix = f"{date}_{simple_bid}"

        out_dir = out_root / simple_bid / date
        out_dir.mkdir(parents=True, exist_ok=True)
        final_dir = out_dir / "final"
        final_dir.mkdir(exist_ok=True)
        pol_final = final_dir / f"{prefix}_pol_SWEREF99TM_10m.tif"

        # Skip if already done
        if pol_final.exists():
            final_tifs.append(str(pol_final))
            print(f"[{idx+1}/{n}] {simple_bid} — already done, skipping")
            continue

        burst = row.copy()
        burst["bid"] = simple_bid
        burst["file_location"] = scene_zip
        burst["master_prefix"] = prefix
        burst["out_directory"] = out_dir
        burst["slave_file"] = None
        burst["slave_prefix"] = None
        burst["slave_burst_nr"] = None

        print(f"[{idx+1}/{n}] {simple_bid}")

        try:
            bid, bdate, out_bs, out_ls, out_pol, out_coh, out_dprvi, err = \
                burst_to_ard(burst, str(config_file))

            if out_pol and Path(out_pol).exists():
                print(f"  ✓ H-A-Alpha")
                pol_tif = out_dir / f"{prefix}_pol.tif"
                dim_to_tif(Path(out_pol), pol_tif)
                reproject_to_sweref(pol_tif, pol_final)
                
                # NORMALIZE ENTROPY BAND (0-100 → 0-1)
                normalize_entropy_band(pol_final)
                
                final_tifs.append(str(pol_final))
                print(f"    → {pol_final.name}")
            elif err:
                print(f"  ⚠ {err}")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
        print()

    return final_tifs


def merge_by_date(out_root: Path):
    """
    Group per-scene mosaics by acquisition date and merge each group
    into a single dated mosaic: <YYYYMMDD>_pol_SWEREF99TM_10m.tif
    """
    merged_dir = out_root / "merged_SWEREF99TM_10m"
    merged_dir.mkdir(exist_ok=True)

    scene_tifs = [
        t for t in sorted(merged_dir.glob("*_pol_SWEREF99TM_10m.tif"))
        if "all_scenes" not in t.name
        and not re.match(r'^\d{8}_', t.name)
    ]

    if not scene_tifs:
        print("[INFO] No per-scene mosaics found to group by date")
        return

    def get_date(path: Path) -> str:
        return extract_date_from_scene_id(path.stem)

    print(f"\n{'=' * 60}")
    print("MERGING PER-SCENE MOSAICS BY DATE")
    print(f"{'=' * 60}")

    for date, group in groupby(sorted(scene_tifs, key=get_date), key=get_date):
        tifs = [str(t) for t in group]
        out_tif = merged_dir / f"{date}_pol_SWEREF99TM_10m.tif"

        if out_tif.exists():
            print(f"  [SKIP] {out_tif.name} already exists")
            continue

        print(f"  Merging {len(tifs)} scene(s) for date {date}")
        if merge_burst_tifs(tifs, out_tif):
            mb = out_tif.stat().st_size / (1024 * 1024)
            print(f"  ✓ {out_tif.name} ({mb:.1f} MB)")

    print(f"{'=' * 60}")


def main():
    ap = argparse.ArgumentParser(description="S1 SLC → H-A-Alpha → SWEREF99TM 10m (all zips)")
    ap.add_argument("--scene_zip", type=str, default=None)
    ap.add_argument("--scene_dir", type=str, default=None)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--temp_dir", type=str, default=None)
    ap.add_argument("--no_merge", action="store_true")
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--aoi", type=str, default=None)
    ap.add_argument("--product_type", type=str, default="RTC-gamma0")
    ap.add_argument("--resolution", type=int, default=20)
    ap.add_argument("--target_epsg", type=int, default=3006)
    ap.add_argument("--target_res", type=int, default=10)
    args = ap.parse_args()

    set_target(args.target_epsg, args.target_res)

    out_root = Path(wsl_unc_to_linux(args.out_root)).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    temp_dir = (Path(wsl_unc_to_linux(args.temp_dir)).expanduser().resolve()
                if args.temp_dir else out_root / "_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    if args.scene_zip:
        scene_zips = [Path(wsl_unc_to_linux(args.scene_zip))]
    elif args.scene_dir:
        scene_zips = sorted(Path(wsl_unc_to_linux(args.scene_dir)).rglob("*.zip"))
    else:
        print("ERROR: provide --scene_zip or --scene_dir")
        sys.exit(1)

    if not scene_zips:
        print(f"ERROR: No .zip files found in {args.scene_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(scene_zips)} scene(s) to process")

    all_scene_tifs = []

    for scene_zip in scene_zips:
        scene_id = scene_zip.stem

        print("=" * 60)
        print("SENTINEL-1 SLC → H-A-ALPHA DECOMPOSITION")
        print("=" * 60)
        print(f"Scene:    {scene_zip.name}")
        print(f"Product:  {args.product_type}")
        print(f"Retries:  {args.max_retries}")
        print(f"Target:   EPSG:{args.target_epsg} @ {args.target_res}m")
        print("=" * 60)

        config_dict = build_config(
            scene_zip, out_root, temp_dir,
            backscatter=False, haalpha=True, dprvi=False,
            product_type=args.product_type, resolution=args.resolution,
        )
        config_file = out_root / "config_haalpha.json"
        config_file.write_text(json.dumps(config_dict, indent=2))

        scene = S1Scene(scene_id)
        burst_gdf = extract_burst_inventory_from_zip(
            scene_zip, scene.scene_id, scene.rel_orbit, scene.start_date)
        if args.aoi:
            burst_gdf = filter_bursts_by_aoi(burst_gdf, args.aoi)
        n = len(burst_gdf)
        print(f"\n[PROCESSING] {n} bursts\n")

        if n == 0:
            print(f"[SKIP] No bursts overlap AOI for {scene_id}")
            continue

        for attempt in range(1 + args.max_retries):
            if attempt > 0:
                reset = reset_failed_bursts(out_root, PRODUCT_KEY, FINAL_PATTERN)
                if reset == 0:
                    break
                print(f"\n{'=' * 60}")
                print(f"RETRY {attempt}/{args.max_retries} — {reset} failed bursts reset")
                print(f"{'=' * 60}\n")

            final_tifs = process_bursts(burst_gdf, scene_id, scene_zip,
                                        out_root, config_file)
            if len(final_tifs) == n:
                break

        print(f"\n[RESULT] {len(final_tifs)}/{n} bursts completed")
        if len(final_tifs) < n:
            print(f"[WARN] {n - len(final_tifs)} bursts failed after {args.max_retries} retries")

        # Per-scene mosaic
        if not args.no_merge and final_tifs:
            merged_dir = out_root / "merged_SWEREF99TM_10m"
            merged_dir.mkdir(exist_ok=True)
            out_tif = merged_dir / f"{scene_id}_pol_SWEREF99TM_10m.tif"
            print(f"\n{'=' * 60}")
            print(f"MERGING {len(final_tifs)} bursts for {scene_id}")
            if merge_burst_tifs(final_tifs, out_tif):
                mb = out_tif.stat().st_size / (1024 * 1024)
                print(f"  ✓ H-A-Alpha: {out_tif.name} ({mb:.1f} MB)")
                all_scene_tifs.append(out_tif)
            print(f"{'=' * 60}")

        print(f"\nDone with scene: {scene_id}")

    # Per-date mosaic
    if not args.no_merge and all_scene_tifs:
        merge_by_date(out_root)

    print(f"\nAll done. Output: {out_root}")


if __name__ == "__main__":
    main()