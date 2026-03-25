import json
from pathlib import Path
import argparse
from tqdm import tqdm

import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import numpy as np

"""
run with:
python initialize_dataset.py \
    --fields_file /home/johan/Thesis/Sentinel_1/ost/s1/Example_Fields/example_fields.parquet \
    --dataset_path /home/johan/Thesis/Sentinel_1/ost/s1/S1_Timeseries/dataset \
    --id_field field_id \
    --crs EPSG:3006
"""


# Creates folder structure, auxilliary data for each field (mask, polygon, bbox)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields_file', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--id_field', type=str, required=False, default=None)
    parser.add_argument('--crs', type=str, required=False, default='EPSG:3006')
    parser.add_argument('--negative_buffer', type=int, default=-5, required=False)
    args = parser.parse_args()
    fields_file, dataset_path = Path(args.fields_file), Path(args.dataset_path)
    out_path = dataset_path / 'data'
    out_path.mkdir(exist_ok=True)

    grid_size = 10

    # Loading all fields
    fields = gpd.read_parquet(fields_file)
    fields = fields.to_crs(args.crs)

    samples = []

    # Iterating over fields
    for i, field in tqdm(enumerate(fields.itertuples()), total=len(fields)):
        field_id = str(getattr(field, args.id_field)) if args.id_field is not None else f'field{i}'
        field_path = out_path / field_id
        field_path.mkdir(exist_ok=True)

        # Extracting bounding box of field
        polygon = field.geometry
        polygon_buffered = polygon.buffer(args.negative_buffer)

        gdf_polygon = gpd.GeoDataFrame({'geometry': [polygon]}, crs=args.crs)
        gdf_polygon.to_parquet(field_path / f'polygon_{field_id}.parquet')

        # Get bounds of polygon
        xmin, ymin, xmax, ymax = polygon.bounds

        # Round bounds to grid alignment
        xmin_rounded = np.floor(xmin / grid_size) * grid_size
        ymin_rounded = np.floor(ymin / grid_size) * grid_size
        xmax_rounded = np.ceil(xmax / grid_size) * grid_size
        ymax_rounded = np.ceil(ymax / grid_size) * grid_size

        rounded_bounds = (xmin_rounded, ymin_rounded, xmax_rounded, ymax_rounded)

        gdf_bbox = gpd.GeoDataFrame({'geometry': [box(*rounded_bounds)]}, crs=args.crs)
        gdf_bbox.to_parquet(field_path / f'bbox_{field_id}.parquet')

        # Calculate grid dimensions
        n_cols = int((xmax_rounded - xmin_rounded) / grid_size)
        n_rows = int((ymax_rounded - ymin_rounded) / grid_size)

        # Create mask for grid cells completely covered by ORIGINAL polygon
        mask = np.zeros((n_rows, n_cols), dtype=bool)

        for i in range(n_rows):
            for j in range(n_cols):
                # Create grid cell
                cell_xmin = xmin_rounded + j * grid_size
                cell_ymin = ymin_rounded + i * grid_size
                cell_xmax = cell_xmin + grid_size
                cell_ymax = cell_ymin + grid_size

                cell = box(cell_xmin, cell_ymin, cell_xmax, cell_ymax)

                # Check if cell is completely within original polygon
                # A cell is completely covered if it's within the polygon
                mask[i, j] = polygon_buffered.contains(cell)

        # Convert boolean mask to uint8 (0 and 1)
        mask_uint8 = mask.astype(np.uint8)

        # Flip mask vertically because rasterio expects data from top to bottom
        mask_uint8 = np.flipud(mask_uint8)

        # Create affine transform for georeferencing
        # Rasterio uses top-left corner as origin
        transform = from_origin(xmin_rounded, ymax_rounded, grid_size, grid_size)

        # Write to GeoTIFF
        with rasterio.open(
                field_path / f'mask_{field_id}.tif',
                'w',
                driver='GTiff',
                height=mask.shape[0],
                width=mask.shape[1],
                count=1,
                dtype=mask_uint8.dtype,
                crs=args.crs,
                transform=transform,
                nodata=None,
                compress='lzw'  # Optional compression
        ) as dst:
            dst.write(mask_uint8, 1)