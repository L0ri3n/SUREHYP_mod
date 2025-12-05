"""
Enhanced DEM Download with Multi-Level Fallback Strategy

This module provides robust DEM downloading with multiple fallback options:
1. Primary GEE DEM source (SRTM, NASADEM, etc.)
2. Alternative GEE DEM sources
3. Local DEM file
4. Flat terrain assumption (handled by caller)

Author: Enhanced for Hyperion Processing
Date: 2025-12-05
"""

import os
import numpy as np
import ee
import requests
import rasterio
from pathlib import Path
from scipy.interpolate import griddata


def downloadDEMfromGEE_robust(UL_lon, UL_lat, UR_lon, UR_lat, LR_lon, LR_lat, LL_lon, LL_lat,
                               demID='USGS/SRTMGL1_003', elevationName='elevation', output_path=None,
                               fallback_dems=None, local_dem_path=None):
    """
    Download DEM from Google Earth Engine with multiple fallback strategies.

    Fallback strategy hierarchy:
    1. Primary DEM source (demID parameter)
    2. Alternative GEE DEM sources (ASTER, NASADEM, etc.)
    3. Local DEM file if provided
    4. Raise exception (caller can handle with flat terrain assumption)

    Parameters:
    -----------
    UL_lon, UL_lat, UR_lon, UR_lat, LR_lon, LR_lat, LL_lon, LL_lat : float
        Bounding box coordinates
    demID : str
        Primary GEE DEM asset ID (default: 'USGS/SRTMGL1_003')
    elevationName : str
        Band name containing elevation data
    output_path : str, optional
        Output directory for DEM file
    fallback_dems : list of dict, optional
        List of alternative DEM sources: [{'id': 'asset_id', 'band': 'band_name'}, ...]
    local_dem_path : str, optional
        Path to local DEM file to use as final fallback

    Returns:
    --------
    str : Path to output directory containing 'elev.tif'

    Raises:
    -------
    ValueError : If all DEM sources fail (including local fallback)
    """
    # Use default path in OUT folder if not specified
    if output_path is None:
        output_path = os.path.join(os.environ.get('HYPERION_OUT_PATH', '.'), 'elev/')

    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Define the bounding box with some buffer
    region = ee.Geometry.Polygon([
        [[UL_lon - 0.05, UL_lat + 0.05],
         [UR_lon + 0.05, UR_lat + 0.05],
         [LR_lon + 0.05, LR_lat - 0.05],
         [LL_lon - 0.05, LL_lat - 0.05],
         [UL_lon - 0.05, UL_lat + 0.05]]
    ], None, False)

    # Default fallback DEM sources (if not provided)
    if fallback_dems is None:
        fallback_dems = [
            {'id': 'NASA/NASADEM_HGT/001', 'band': 'elevation', 'name': 'NASADEM (improved SRTM)'},
            {'id': 'JAXA/ALOS/AW3D30/V4_1', 'band': 'DSM', 'name': 'ALOS World 3D 30m V4'},
            {'id': 'USGS/GTOPO30', 'band': 'elevation', 'name': 'GTOPO30 (global, lower res)'},
        ]

    # Create list of DEM sources to try (primary + fallbacks)
    primary_source = {'id': demID, 'band': elevationName, 'name': f'{demID} (primary)'}
    dem_sources = [primary_source] + fallback_dems

    # Try each DEM source in order
    print(f'\n    DEM Fallback Strategy: Trying {len(dem_sources)} GEE sources')
    for idx, dem_source in enumerate(dem_sources):
        source_id = dem_source['id']
        source_band = dem_source['band']
        source_name = dem_source.get('name', source_id)

        if idx == 0:
            print(f'    [{idx+1}/{len(dem_sources)}] Primary DEM: {source_name}')
        else:
            print(f'    [{idx+1}/{len(dem_sources)}] Fallback DEM: {source_name}')

        try:
            # Try downloading from this DEM source
            result = _download_dem_from_source(region, source_id, source_band, output_path)
            if result:
                print(f'    ✓ SUCCESS: Obtained DEM from {source_name}')
                return result
        except Exception as e:
            error_msg = str(e)[:150]  # Truncate long error messages
            print(f'    ✗ FAILED: {error_msg}')
            # Continue to next source
            continue

    # All GEE sources failed - try local DEM if provided
    if local_dem_path and os.path.exists(local_dem_path):
        print(f'\n    All GEE sources failed. Trying local DEM: {local_dem_path}')
        try:
            # Copy local DEM to output location
            import shutil
            output_file = os.path.join(output_path, 'elev.tif')
            shutil.copy2(local_dem_path, output_file)

            # Verify the file is valid
            with rasterio.open(output_file) as src:
                data = src.read(1)
                valid_data = data[(~np.isnan(data)) & (data != 0)]
                if len(valid_data) < 10:
                    raise ValueError("Local DEM contains insufficient valid data")

            print(f'    ✓ SUCCESS: Using local DEM file')
            return output_path
        except Exception as e:
            print(f'    ✗ FAILED: Local DEM error - {e}')

    # All sources failed - provide detailed error message
    error_msg = (
        f"\n{'='*70}\n"
        f"DEM ACQUISITION FAILED\n"
        f"{'='*70}\n"
        f"Failed to obtain DEM data from all {len(dem_sources)} sources.\n\n"
        f"Sources tried:\n"
    )
    for idx, src in enumerate(dem_sources, 1):
        error_msg += f"  {idx}. {src.get('name', src['id'])}\n"

    error_msg += (
        f"\nLocal DEM: {'Not provided or invalid' if not local_dem_path else local_dem_path}\n"
        f"Region: ({UL_lat:.4f}, {UL_lon:.4f}) to ({LR_lat:.4f}, {LR_lon:.4f})\n\n"
        f"Possible causes:\n"
        f"  • No DEM coverage for this region (e.g., ocean, polar areas)\n"
        f"  • Corrupted or missing data in Google Earth Engine\n"
        f"  • Network connectivity issues\n"
        f"  • GEE authentication problems\n\n"
        f"Solutions:\n"
        f"  1. Provide a local DEM file via local_dem_path parameter\n"
        f"  2. Disable topographic correction (set use_topo=False)\n"
        f"  3. Use flat terrain assumption (will be applied automatically)\n"
        f"{'='*70}\n"
    )

    raise ValueError(error_msg)


def _download_dem_from_source(region, demID, elevationName, output_path):
    """
    Internal function to download DEM from a specific GEE source.

    Parameters:
    -----------
    region : ee.Geometry
        Region of interest
    demID : str
        GEE asset ID
    elevationName : str
        Band name containing elevation data
    output_path : str
        Output directory

    Returns:
    --------
    str : Path to output directory on success

    Raises:
    -------
    Exception : On any download error
    """
    # Try loading as Image first, then as ImageCollection
    try:
        elev = ee.Image(demID)
        elev = elev.select(elevationName)
        # Test if it's valid
        info = elev.getInfo()
        if info is None:
            raise ValueError("Image info returned None")
    except Exception as e1:
        try:
            dem = ee.ImageCollection(demID)
            dem = dem.select(elevationName)
            elev = dem.mosaic()
            # Test if it's valid
            info = elev.getInfo()
            if info is None:
                raise ValueError("ImageCollection mosaic returned None")
        except Exception as e2:
            raise ValueError(f"Cannot load {demID}: {str(e1)[:50]} | {str(e2)[:50]}")

    # Clip to region
    elev = elev.clip(region)

    # Method 1: Try direct download via getDownloadURL
    try:
        url = elev.getDownloadURL({
            'name': 'elev',
            'scale': 30,
            'region': region,
            'format': 'GEO_TIFF'
        })

        response = requests.get(url, timeout=120)  # 2 minute timeout

        if response.status_code == 200:
            output_file = os.path.join(output_path, 'elev.tif')
            with open(output_file, 'wb') as f:
                f.write(response.content)

            # Verify the file is valid and contains sufficient data
            with rasterio.open(output_file) as src:
                data = src.read(1)
                # Check if we have valid data (not all zeros or NaN)
                valid_data = data[(~np.isnan(data)) & (data != 0)]
                if len(valid_data) < 100:  # Require at least 100 valid pixels
                    raise ValueError(f"DEM contains only {len(valid_data)} valid pixels (insufficient)")

                # Additional check: ensure reasonable elevation range
                if len(valid_data) > 0:
                    min_elev, max_elev = np.min(valid_data), np.max(valid_data)
                    if max_elev - min_elev < 0.1 and max_elev < 10000:
                        # Suspiciously flat or all same value
                        print(f"      Warning: DEM appears uniform (range: {min_elev:.1f} to {max_elev:.1f} m)")

            return output_path
        else:
            raise Exception(f"HTTP error {response.status_code}")

    except Exception as e:
        # Method 2: Try sampling method for smaller regions
        try:
            bounds = region.bounds().getInfo()['coordinates'][0]
            min_lon = min(p[0] for p in bounds)
            max_lon = max(p[0] for p in bounds)
            min_lat = min(p[1] for p in bounds)
            max_lat = max(p[1] for p in bounds)

            scale = 30  # meters

            # Sample the DEM with higher density
            samples = elev.sample(
                region=region,
                scale=scale,
                geometries=True,
                dropNulls=True  # Drop null/missing values
            ).getInfo()

            if not samples or not samples.get('features') or len(samples['features']) < 100:
                raise ValueError(f"Insufficient samples: {len(samples.get('features', []))} points")

            # Get coordinates and elevations
            points = []
            for f in samples['features']:
                coords = f['geometry']['coordinates']
                elev_val = f['properties'].get(elevationName, f['properties'].get('elevation'))
                # Only include valid elevation values
                if elev_val is not None and not np.isnan(elev_val):
                    points.append([coords[0], coords[1], elev_val])

            if len(points) < 100:
                raise ValueError(f"Only {len(points)} valid elevation samples (need 100+)")

            points = np.array(points)

            # Create a gridded DEM using interpolation
            # Define output grid (500x500 for good resolution)
            grid_x = np.linspace(min_lon, max_lon, 500)
            grid_y = np.linspace(min_lat, max_lat, 500)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

            # Interpolate using linear method first
            grid_z = griddata(points[:, :2], points[:, 2], (grid_xx, grid_yy), method='linear')

            # Fill NaN values (areas without nearby samples) using nearest neighbor
            if np.isnan(grid_z).any():
                nan_mask = np.isnan(grid_z)
                n_nans = np.sum(nan_mask)
                print(f"      Filling {n_nans} NaN pixels with nearest neighbor interpolation")
                grid_z[nan_mask] = griddata(points[:, :2], points[:, 2],
                                           (grid_xx[nan_mask], grid_yy[nan_mask]), method='nearest')

            grid_z = np.flipud(grid_z)  # Flip to match raster convention (north-up)

            # Save as GeoTIFF
            output_file = os.path.join(output_path, 'elev.tif')
            transform = rasterio.transform.from_bounds(min_lon, min_lat, max_lon, max_lat, 500, 500)

            with rasterio.open(
                output_file, 'w',
                driver='GTiff',
                height=500,
                width=500,
                count=1,
                dtype=grid_z.dtype,
                crs='EPSG:4326',
                transform=transform,
                compress='lzw'
            ) as dst:
                dst.write(grid_z, 1)

            print(f"      Created interpolated DEM from {len(points)} sample points")
            return output_path

        except Exception as e2:
            raise Exception(f"Both download methods failed: {str(e)[:80]} | {str(e2)[:80]}")


def apply_flat_terrain_assumption(image_shape, output_path, average_elevation_m=0):
    """
    Create a flat DEM for cases where no real DEM data is available.

    This is useful for:
    - Ocean/water areas where DEMs have no data
    - Testing without topographic correction
    - Fallback when all DEM sources fail

    Parameters:
    -----------
    image_shape : tuple
        (rows, cols) of the Hyperion image
    output_path : str
        Output directory for the flat DEM
    average_elevation_m : float
        Elevation to use (default: 0 = sea level)

    Returns:
    --------
    dict : Contains flat terrain parameters
        - elev: elevation array (all same value)
        - slope: slope array (all zeros)
        - aspect: aspect array (all zeros)
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)

    rows, cols = image_shape

    print(f"\n    {'='*70}")
    print(f"    APPLYING FLAT TERRAIN ASSUMPTION")
    print(f"    {'='*70}")
    print(f"    Creating synthetic flat DEM:")
    print(f"      - Dimensions: {rows} x {cols} pixels")
    print(f"      - Elevation: {average_elevation_m} m (constant)")
    print(f"      - Slope: 0° (flat terrain)")
    print(f"      - Aspect: 0° (undefined for flat terrain)")
    print(f"    ")
    print(f"    Note: Topographic correction will have minimal effect")
    print(f"          on flat terrain. Results will be similar to")
    print(f"          atmospheric correction without topo correction.")
    print(f"    {'='*70}\n")

    # Create flat terrain arrays
    elev = np.full((rows, cols), average_elevation_m, dtype=np.float32)
    slope = np.zeros((rows, cols), dtype=np.float32)
    aspect = np.zeros((rows, cols), dtype=np.float32)

    # Save as GeoTIFF for consistency
    output_file = os.path.join(output_path, 'elev_flat.tif')

    # Simple transform (we don't have real georeferencing, but create valid transform)
    from rasterio.transform import from_origin
    transform = from_origin(0, rows, 1, 1)  # 1x1 pixel size starting at origin

    with rasterio.open(
        output_file, 'w',
        driver='GTiff',
        height=rows,
        width=cols,
        count=1,
        dtype=elev.dtype,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw'
    ) as dst:
        dst.write(elev, 1)

    print(f"    Flat DEM saved to: {output_file}")

    return {
        'elev': elev,
        'slope': slope,
        'aspect': aspect,
        'is_flat': True,
        'dem_path': output_file
    }
