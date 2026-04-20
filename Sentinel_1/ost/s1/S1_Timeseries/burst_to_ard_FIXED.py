#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
FIXED burst_to_ard.py — shared calibration for H-A-Alpha + DpRVI

Key: calibration to complex is done ONCE and shared. Then:
  - H-A-Alpha receives the calibrated (NOT debursted) product
    because OST's ha_alpha wrapper does deburst internally.
  - DpRVI receives the calibrated product, debursts it, then
    computes C2 matrix and DpRVI.
  - Backscatter has its own calibration to intensity (independent).

Returns 8 values: (bid, date, out_bs, out_ls, out_pol, out_coh, out_dprvi, error)
"""

import json
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Add OST to path (relative to script location)
script_dir = Path(__file__).parent.absolute()
ost_path = script_dir.parent.parent.parent  # Go up 3 levels to /home/johan/Thesis/Sentinel_1
sys.path.insert(0, str(ost_path))

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from ost.helpers import helpers as h
from ost.helpers.settings import GPT_FILE
from ost.s1 import slc_wrappers as slc
from ost.generic import common_wrappers as common
from ost.helpers import raster as ras
from ost.helpers.errors import GPTRuntimeError, NotValidFileError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DpRVI math
# ---------------------------------------------------------------------------

def compute_dprvi_from_c2(c11, c12_real, c12_imag, c22, window_size=3):
    """DpRVI = 1 - m*beta from C2 covariance matrix (Mandal et al. 2020)."""
    c11 = c11.astype(np.float64)
    c12_real = c12_real.astype(np.float64)
    c12_imag = c12_imag.astype(np.float64)
    c22 = c22.astype(np.float64)

    if window_size > 1:
        c11 = uniform_filter(c11, size=window_size)
        c12_real = uniform_filter(c12_real, size=window_size)
        c12_imag = uniform_filter(c12_imag, size=window_size)
        c22 = uniform_filter(c22, size=window_size)

    c12 = c12_real + 1j * c12_imag
    c21 = np.conjugate(c12)
    c2_det = c11 * c22 - c12 * c21
    c2_trace = c11 + c22

    trace_sq = np.power(c2_trace, 2)
    ratio = np.where(trace_sq > 0, 4.0 * np.real(c2_det) / np.real(trace_sq), 0.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    m = np.sqrt(1.0 - ratio)

    sqdiscr = np.sqrt(np.abs(c2_trace * c2_trace - 4.0 * c2_det))
    egv1 = np.real((c2_trace + sqdiscr) * 0.5)
    egv2 = np.real((c2_trace - sqdiscr) * 0.5)
    egv_sum = egv1 + egv2
    beta = np.where(egv_sum > 0, np.abs(egv1) / np.abs(egv_sum), 0.0)

    dprvi = np.abs(1.0 - m * beta)
    dprvi = np.where(np.isfinite(dprvi), np.clip(dprvi, 0.0, 1.0), np.nan)
    return dprvi.astype(np.float32)


# ---------------------------------------------------------------------------
# Shared: Calibrate to complex (NOT deburst — H-A-Alpha needs burst data)
# ---------------------------------------------------------------------------

def calibrate_complex(import_file, out_cal, out_dir, burst_prefix, config_dict):
    """
    Shared calibration to complex. Output is still in burst format.
    - H-A-Alpha can use this directly (its wrapper debursts internally)
    - DpRVI will deburst this before computing C2
    """
    cpus = config_dict["snap_cpu_parallelism"]
    cal_log = out_dir / f"{burst_prefix}_cal_complex.err_log"
    logger.info("Calibrating to complex (shared for H-A-Alpha + DpRVI)")

    try:
        command = (
            f"{GPT_FILE} Calibration -x -q {2*cpus} "
            f"-PoutputImageInComplex=true "
            f"-PoutputImageScaleInDb=false "
            f"-t '{str(out_cal)}' '{str(import_file)}'"
        )
        return_code = h.run_command(command, cal_log)
        if return_code != 0:
            raise GPTRuntimeError(
                f"Calibration failed ({return_code}). See {cal_log}"
            )
        return_code = h.check_out_dimap(out_cal)
        if return_code != 0:
            raise NotValidFileError(f"Cal check failed: {return_code}")
    except (GPTRuntimeError, NotValidFileError) as error:
        logger.info(error)
        return None, error

    return str(out_cal.with_suffix(".dim")), None


# ---------------------------------------------------------------------------
# H-A-Alpha (uses shared calibrated product — still in burst format)
# ---------------------------------------------------------------------------

def create_polarimetric_layers(cal_dim, out_dir, burst_prefix, config_dict):
    """
    H-A-Alpha from calibrated (NOT debursted) product.
    OST's ha_alpha wrapper handles deburst + pol speckle + decomposition.
    """
    with TemporaryDirectory(prefix=f"{config_dict['temp_dir']}/") as temp:
        temp = Path(temp)

        out_haa = temp / f"{burst_prefix}_h"
        haa_log = out_dir / f"{burst_prefix}_haa.err_log"
        try:
            slc.ha_alpha(str(cal_dim), out_haa, haa_log, config_dict)
        except (GPTRuntimeError, NotValidFileError) as error:
            logger.info(error)
            return None, error

        out_htc = temp / f"{burst_prefix}_pol"
        haa_tc_log = out_dir / f"{burst_prefix}_haa_tc.err_log"
        try:
            common.terrain_correction(
                out_haa.with_suffix(".dim"), out_htc, haa_tc_log, config_dict
            )
        except (GPTRuntimeError, NotValidFileError) as error:
            logger.info(error)
            return None, error

        try:
            ras.image_bounds(out_htc.with_suffix(".data"))
        except Exception as e:
            logger.info(f"Error creating outline: {e}")

        ard = config_dict["processing"]["single_ARD"]
        h.move_dimap(out_htc, out_dir / f"{burst_prefix}_pol", ard["to_tif"])

        with (out_dir / ".pol.processed").open("w+") as file:
            file.write("passed all tests \n")

        return str(out_dir / f"{burst_prefix}_pol.dim"), None


# ---------------------------------------------------------------------------
# DpRVI (uses shared calibrated product, debursts it first)
# ---------------------------------------------------------------------------

def create_dprvi_layers(cal_dim, out_dir, burst_prefix, config_dict):
    """
    DpRVI from calibrated product:
      1. TOPSAR-Deburst (needed for Polarimetric-Matrices)
      2. Polarimetric-Matrices → C2
      3. Optional pol speckle filter
      4. Terrain correction
      5. Compute DpRVI from C2 bands
    """
    ard = config_dict["processing"]["single_ARD"]
    cpus = config_dict["snap_cpu_parallelism"]

    with TemporaryDirectory(prefix=f"{config_dict['temp_dir']}/") as temp:
        temp = Path(temp)

        # 1. Deburst
        out_deburst = temp / f"{burst_prefix}_deburst_c2"
        deburst_log = out_dir / f"{burst_prefix}_deburst_c2.err_log"
        logger.info("Debursting for C2/DpRVI")

        try:
            command = (
                f"{GPT_FILE} TOPSAR-Deburst -x -q {2*cpus} "
                f"-t '{str(out_deburst)}' '{str(cal_dim)}'"
            )
            return_code = h.run_command(command, deburst_log)
            if return_code != 0:
                raise GPTRuntimeError(
                    f"Deburst failed ({return_code}). See {deburst_log}"
                )
            return_code = h.check_out_dimap(out_deburst)
            if return_code != 0:
                raise NotValidFileError(f"Deburst check failed: {return_code}")
        except (GPTRuntimeError, NotValidFileError) as error:
            logger.info(error)
            return None, error

        # 2. C2 covariance matrix
        out_c2 = temp / f"{burst_prefix}_c2"
        c2_log = out_dir / f"{burst_prefix}_c2.err_log"
        logger.info("Computing C2 covariance matrix")

        try:
            command = (
                f"{GPT_FILE} Polarimetric-Matrices -x -q {2*cpus} "
                f"-Pmatrix=C2 "
                f"-t '{str(out_c2)}' '{str(out_deburst.with_suffix('.dim'))}'"
            )
            return_code = h.run_command(command, c2_log)
            if return_code != 0:
                raise GPTRuntimeError(
                    f"C2 matrix failed ({return_code}). See {c2_log}"
                )
            return_code = h.check_out_dimap(out_c2)
            if return_code != 0:
                raise NotValidFileError(f"C2 check failed: {return_code}")
        except (GPTRuntimeError, NotValidFileError) as error:
            logger.info(error)
            return None, error

        h.delete_dimap(out_deburst)

        # 3. Polarimetric speckle filter (optional)
        if ard.get("remove_pol_speckle", False):
            out_c2_filt = temp / f"{burst_prefix}_c2_filt"
            c2_filt_log = out_dir / f"{burst_prefix}_c2_filt.err_log"
            logger.info("Polarimetric speckle filter on C2")

            pol_sf = ard.get("pol_speckle_filter", {})
            try:
                command = (
                    f"{GPT_FILE} Polarimetric-Speckle-Filter -x -q {2*cpus} "
                    f"-Pfilter='{pol_sf.get('polarimetric_filter', 'Refined Lee Filter')}' "
                    f"-PfilterSize={pol_sf.get('filter_size', 5)} "
                    f"-PnumLooksStr={pol_sf.get('num_of_looks', 1)} "
                    f"-t '{str(out_c2_filt)}' "
                    f"'{str(out_c2.with_suffix('.dim'))}'"
                )
                return_code = h.run_command(command, c2_filt_log)
                if return_code != 0:
                    raise GPTRuntimeError(
                        f"Pol speckle filter failed ({return_code}). See {c2_filt_log}"
                    )
                return_code = h.check_out_dimap(out_c2_filt)
                if return_code != 0:
                    raise NotValidFileError(f"C2 filt check failed: {return_code}")
                h.delete_dimap(out_c2)
                out_c2 = out_c2_filt
            except (GPTRuntimeError, NotValidFileError) as error:
                logger.info(f"Pol speckle filter failed, continuing without: {error}")

        # 4. Terrain correction
        out_c2_tc = temp / f"{burst_prefix}_c2_tc"
        c2_tc_log = out_dir / f"{burst_prefix}_c2_tc.err_log"
        logger.info("Terrain-correcting C2 matrix")

        try:
            common.terrain_correction(
                out_c2.with_suffix(".dim"), out_c2_tc, c2_tc_log, config_dict
            )
        except (GPTRuntimeError, NotValidFileError) as error:
            logger.info(error)
            return None, error

        h.delete_dimap(out_c2)

        # 5. Read C2 bands and compute DpRVI
        logger.info("Computing DpRVI from C2 bands")
        c2_data_dir = out_c2_tc.with_suffix(".data")
        if not c2_data_dir.is_dir():
            error = FileNotFoundError(f"C2 data dir not found: {c2_data_dir}")
            logger.error(str(error))
            return None, error

        all_imgs = sorted(c2_data_dir.glob("*.img"))
        logger.info(f"C2 bands available: {[f.stem for f in all_imgs]}")

        c2_bands = {}
        band_patterns = {
            "c11": ["C11", "c11"],
            "c12_real": ["C12_real", "c12_real", "C12_Real"],
            "c12_imag": ["C12_imag", "c12_imag", "C12_Imag"],
            "c22": ["C22", "c22"],
        }
        for band_key, patterns in band_patterns.items():
            for pattern in patterns:
                matches = [f for f in all_imgs if pattern in f.stem]
                if matches:
                    c2_bands[band_key] = matches[0]
                    break

        if len(c2_bands) < 4:
            missing = [k for k in band_patterns if k not in c2_bands]
            error = RuntimeError(
                f"Missing C2 bands: {missing}. "
                f"Available: {[f.stem for f in all_imgs]}"
            )
            logger.error(str(error))
            return None, error

        try:
            with rasterio.open(str(c2_bands["c11"])) as src:
                c11 = src.read(1).astype(np.float32)
                profile = src.profile.copy()
            with rasterio.open(str(c2_bands["c12_real"])) as src:
                c12_real = src.read(1).astype(np.float32)
            with rasterio.open(str(c2_bands["c12_imag"])) as src:
                c12_imag = src.read(1).astype(np.float32)
            with rasterio.open(str(c2_bands["c22"])) as src:
                c22 = src.read(1).astype(np.float32)
        except Exception as e:
            logger.error(f"Error reading C2 bands: {e}")
            return None, e

        dprvi_window = ard.get("dprvi_window_size", 3)

        # Mask SNAP background BEFORE spatial averaging smears the zeros
        snap_background = (c11 == 0) & (c22 == 0)

        dprvi = compute_dprvi_from_c2(c11, c12_real, c12_imag, c22,
                                       window_size=dprvi_window)

        # Expand mask to cover pixels contaminated by averaging window
        from scipy.ndimage import maximum_filter
        snap_background = maximum_filter(snap_background, size=dprvi_window)
        dprvi[snap_background] = np.nan

        dprvi_tif = out_dir / f"{burst_prefix}_dprvi.tif"
        profile.update(
            driver="GTiff", dtype="float32", count=1, nodata=np.nan,
            compress="deflate", tiled=True, blockxsize=256, blockysize=256,
        )
        with rasterio.open(str(dprvi_tif), "w", **profile) as dst:
            dst.write(dprvi, 1)
            dst.set_band_description(1, "DpRVI")

        h.delete_dimap(out_c2_tc)

        with (out_dir / ".dprvi.processed").open("w+") as file:
            file.write("passed all tests \n")

        logger.info(f"DpRVI → {dprvi_tif}")
        return str(dprvi_tif), None


# ---------------------------------------------------------------------------
# Backscatter (unchanged — own calibration to intensity)
# ---------------------------------------------------------------------------

def create_backscatter_layers(import_file, out_dir, burst_prefix, config_dict):
    ard = config_dict["processing"]["single_ARD"]

    with TemporaryDirectory(prefix=f"{config_dict['temp_dir']}/") as temp:
        temp = Path(temp)

        out_cal = temp / f"{burst_prefix}_cal"
        cal_log = out_dir / f"{burst_prefix}_cal.err_log"
        try:
            slc.calibration(import_file, out_cal, cal_log, config_dict)
        except (GPTRuntimeError, NotValidFileError) as error:
            logger.info(error)
            return None, None, error

        if ard["remove_speckle"]:
            speckle_import = temp / f"{burst_prefix}_speckle_import"
            speckle_log = out_dir / f"{burst_prefix}_speckle.err_log"
            try:
                common.speckle_filter(
                    out_cal.with_suffix(".dim"), speckle_import,
                    speckle_log, config_dict,
                )
            except (GPTRuntimeError, NotValidFileError) as error:
                logger.info(error)
                return None, None, error
            h.delete_dimap(out_cal)
            out_cal = speckle_import

        if ard["to_db"]:
            out_db = temp / f"{burst_prefix}_cal_db"
            db_log = out_dir / f"{burst_prefix}_cal_db.err_log"
            try:
                common.linear_to_db(
                    out_cal.with_suffix(".dim"), out_db, db_log, config_dict
                )
            except (GPTRuntimeError, NotValidFileError) as error:
                logger.info(error)
                return None, None, error
            h.delete_dimap(out_cal)
            out_cal = out_db

        out_tc = temp / f"{burst_prefix}_bs"
        tc_log = out_dir / f"{burst_prefix}_bs_tc.err_log"
        try:
            common.terrain_correction(
                out_cal.with_suffix(".dim"), out_tc, tc_log, config_dict
            )
        except (GPTRuntimeError, NotValidFileError) as error:
            logger.info(error)
            return None, None, error

        try:
            ras.image_bounds(out_tc.with_suffix(".data"))
        except Exception as e:
            logger.info(f"Error creating outline: {e}")

        out_ls = None
        if ard["create_ls_mask"] is True:
            ls_mask = temp / f"{burst_prefix}_ls_mask"
            logfile = out_dir / f"{burst_prefix}.ls_mask.errLog"
            try:
                common.ls_mask(
                    out_cal.with_suffix(".dim"), ls_mask, logfile, config_dict
                )
            except (GPTRuntimeError, NotValidFileError) as error:
                logger.info(error)
                return None, None, error
            ls_raster = list(ls_mask.with_suffix(".data").glob("*img"))[0]
            ras.polygonize_ls(ls_raster, ls_mask.with_suffix(".json"))
            out_ls = out_tc.with_suffix(".data").joinpath(
                ls_mask.name
            ).with_suffix(".json")
            ls_mask.with_suffix(".json").rename(out_ls)

        h.move_dimap(out_tc, out_dir / f"{burst_prefix}_bs", ard["to_tif"])

        with (out_dir / ".bs.processed").open("w+") as file:
            file.write("passed all tests \n")

        return (
            str((out_dir / f"{burst_prefix}_bs").with_suffix(".dim")),
            str(out_ls),
            None,
        )


# ---------------------------------------------------------------------------
# Coherence (unchanged)
# ---------------------------------------------------------------------------

def create_coherence_layers(master_import, slave_import, out_dir, master_prefix, config_dict):
    ard = config_dict["processing"]["single_ARD"]

    with TemporaryDirectory(prefix=f"{config_dict['temp_dir']}/") as temp:
        temp = Path(temp)

        out_coreg = temp / f"{master_prefix}_coreg"
        coreg_log = out_dir / f"{master_prefix}_coreg.err_log"
        try:
            slc.coreg(master_import, slave_import, out_coreg, coreg_log, config_dict)
        except (GPTRuntimeError, NotValidFileError) as error:
            logger.info(error)
            h.delete_dimap(out_coreg)
            h.delete_dimap(master_import)
            return None, error
        h.delete_dimap(master_import)
        h.delete_dimap(slave_import)

        out_coh = temp / f"{master_prefix}_coherence"
        coh_log = out_dir / f"{master_prefix}_coh.err_log"
        try:
            slc.coherence(out_coreg.with_suffix(".dim"), out_coh, coh_log, config_dict)
        except (GPTRuntimeError, NotValidFileError) as error:
            logger.info(error)
            return None, error
        h.delete_dimap(out_coreg)

        out_tc = temp / f"{master_prefix}_coh"
        tc_log = out_dir / f"{master_prefix}_coh_tc.err_log"
        try:
            common.terrain_correction(
                out_coh.with_suffix(".dim"), out_tc, tc_log, config_dict
            )
        except (GPTRuntimeError, NotValidFileError) as error:
            logger.info(error)
            return None, error
        h.delete_dimap(out_coh)

        ras.image_bounds(out_tc.with_suffix(".data"))
        h.move_dimap(out_tc, out_dir / f"{master_prefix}_coh", ard["to_tif"])

        with (out_dir / ".coh.processed").open("w+") as file:
            file.write("passed all tests \n")

        return str(out_dir / f"{master_prefix}_coh.dim"), None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def burst_to_ard(burst, config_file):
    """
    Returns: (bid, date, out_bs, out_ls, out_pol, out_coh, out_dprvi, error)
    """
    if isinstance(burst, tuple):
        i, burst = burst

    with open(config_file, "r") as file:
        config_dict = json.load(file)
        ard = config_dict["processing"]["single_ARD"]
        temp_dir = Path(config_dict["temp_dir"])

    out_dir = Path(burst.out_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    pol_file = (out_dir / ".pol.processed").exists()
    bs_file = (out_dir / ".bs.processed").exists()
    coh_file = (out_dir / ".coh.processed").exists()
    dprvi_file = (out_dir / ".dprvi.processed").exists()

    out_bs, out_ls, out_pol, out_coh, out_dprvi = None, None, None, None, None
    compute_dprvi = ard.get("DpRVI", False)

    if ard["coherence"]:
        coherence = True if burst.slave_file else False
    else:
        coherence = False

    master_prefix = burst["master_prefix"]
    master_file = burst["file_location"]
    master_burst_nr = burst["BurstNr"]
    swath = burst["SwathID"]

    logger.info(f"Processing burst {burst.bid} acquired at {burst.Date}")

    need_pol = ard["H-A-Alpha"] and not pol_file
    need_dprvi = compute_dprvi and not dprvi_file
    need_bs = ard["backscatter"] and not bs_file
    need_coh = coherence and not coh_file

    if need_pol or need_dprvi or need_bs or need_coh:

        # -----------------------------------------------------------------
        # 1. Master burst import
        # -----------------------------------------------------------------
        master_import = temp_dir / f"{master_prefix}_import"
        if not master_import.with_suffix(".dim").exists():
            import_log = out_dir / f"{master_prefix}_import.err_log"
            try:
                slc.burst_import(
                    master_file, master_import, import_log,
                    swath, master_burst_nr, config_dict,
                )
            except (GPTRuntimeError, NotValidFileError) as error:
                if master_import.with_suffix(".dim").exists():
                    h.delete_dimap(master_import)
                logger.info(error)
                return burst.bid, burst.Date, None, None, None, None, None, error

        # -----------------------------------------------------------------
        # 2. Shared calibration to complex (for H-A-Alpha and/or DpRVI)
        #    Output is still in burst format (NOT debursted).
        #    - H-A-Alpha uses this directly (its wrapper debursts internally)
        #    - DpRVI will deburst it in create_dprvi_layers()
        # -----------------------------------------------------------------
        cal_dim = None

        if need_pol or need_dprvi:
            out_cal = temp_dir / f"{master_prefix}_cal_complex_shared"

            if not out_cal.with_suffix(".dim").exists():
                cal_dim, error = calibrate_complex(
                    master_import.with_suffix(".dim"),
                    out_cal,
                    out_dir, master_prefix, config_dict,
                )
                if error:
                    logger.info(f"Shared calibration failed: {error}")
                    need_pol = False
                    need_dprvi = False
            else:
                cal_dim = str(out_cal.with_suffix(".dim"))

        # -----------------------------------------------------------------
        # 3. H-A-Alpha (from shared calibrated, NOT debursted product)
        # -----------------------------------------------------------------
        if need_pol and cal_dim:
            out_pol, error = create_polarimetric_layers(
                cal_dim, out_dir, master_prefix, config_dict
            )
        elif ard["H-A-Alpha"] and pol_file:
            out_pol = str(out_dir / f"{master_prefix}_pol.dim")

        # -----------------------------------------------------------------
        # 4. DpRVI (from shared calibrated product, debursts internally)
        # -----------------------------------------------------------------
        if need_dprvi and cal_dim:
            out_dprvi, error = create_dprvi_layers(
                cal_dim, out_dir, master_prefix, config_dict
            )
        elif compute_dprvi and dprvi_file:
            out_dprvi = str(out_dir / f"{master_prefix}_dprvi.tif")

        # Clean up shared calibrated product (both pipelines are done)
        if cal_dim:
            out_cal = temp_dir / f"{master_prefix}_cal_complex_shared"
            if out_cal.with_suffix(".dim").exists():
                h.delete_dimap(out_cal)

        # -----------------------------------------------------------------
        # 5. Backscatter (own calibration to intensity, independent)
        # -----------------------------------------------------------------
        if need_bs:
            out_bs, out_ls, error = create_backscatter_layers(
                master_import.with_suffix(".dim"), out_dir, master_prefix, config_dict
            )
        elif ard["backscatter"] and bs_file:
            out_bs = str(out_dir / f"{master_prefix}_bs.dim")
            if ard["create_ls_mask"] and bs_file:
                out_ls = str(out_dir / f"{master_prefix}_LS.dim")

        # -----------------------------------------------------------------
        # 6. Coherence
        # -----------------------------------------------------------------
        if need_coh:
            slave_prefix = burst["slave_prefix"]
            slave_file = burst["slave_file"]
            slave_burst_nr = burst["slave_burst_nr"]
            with TemporaryDirectory(prefix=f"{str(temp_dir)}/") as temp:
                temp = Path(temp)
                slave_import = temp / f"{slave_prefix}_import"
                import_log = out_dir / f"{slave_prefix}_import.err_log"
                try:
                    slc.burst_import(
                        slave_file, slave_import, import_log,
                        swath, slave_burst_nr, config_dict,
                    )
                except (GPTRuntimeError, NotValidFileError) as error:
                    if slave_import.with_suffix(".dim").exists():
                        h.delete_dimap(slave_import)
                    logger.info(error)
                    return burst.bid, burst.Date, None, None, None, None, None, error
                out_coh, error = create_coherence_layers(
                    master_import.with_suffix(".dim"),
                    slave_import.with_suffix(".dim"),
                    out_dir, master_prefix, config_dict,
                )
                h.delete_dimap(master_import)
        elif coherence and coh_file:
            out_coh = str(out_dir / f"{master_prefix}_coh.dim")
            h.delete_dimap(master_import)
        else:
            h.delete_dimap(master_import)

    else:
        # Everything already processed
        if ard["H-A-Alpha"] and pol_file:
            out_pol = str(out_dir / f"{master_prefix}_pol.dim")
        if ard["backscatter"] and bs_file:
            out_bs = str(out_dir / f"{master_prefix}_bs.dim")
        if ard["create_ls_mask"] and bs_file:
            out_ls = str(out_dir / f"{master_prefix}_LS.dim")
        if coherence and coh_file:
            out_coh = str(out_dir / f"{master_prefix}_coh.dim")
        if compute_dprvi and dprvi_file:
            out_dprvi = str(out_dir / f"{master_prefix}_dprvi.tif")

    return burst.bid, burst.Date, out_bs, out_ls, out_pol, out_coh, out_dprvi, None