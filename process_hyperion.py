"""
Hyperion Image Processing Script
Image: EO1H0370412009263110KF
Processing: L1R/L1T to Surface Reflectance with Topographic Correction
"""

import numpy as np
import os
import sys
import ee
import pandas as pd
from scipy import interpolate
from zipfile import ZipFile
from pathlib import Path
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import subprocess
import requests
import time
import xml.etree.ElementTree as ET
import glob

import surehyp.preprocess
import surehyp.atmoCorrection
import re


def fix_envi_hdr_for_snap(hdr_path, wavelength_file=None, keep_wavelength=False):
    """
    Fix ENVI header files to be compatible with SNAP.

    SNAP automatically appends wavelength values in parentheses to band names,
    which causes "Undefined function" errors in SNAP's expression parser.

    Solution options:
    1. Remove wavelength field (default, safest for SNAP expressions)
    2. Keep wavelength field (keep_wavelength=True) for better SNAP visualization
    3. Load wavelengths from external file for custom spectral info

    Parameters:
    -----------
    hdr_path : str
        Path to the .hdr file to fix
    wavelength_file : str, optional
        Path to a .txt file containing wavelength values (one per line or comma-separated)
        Format: Can be CSV with wavelength,fwhm or plain text with one wavelength per line
    keep_wavelength : bool, optional
        If True, keep wavelength field in HDR (better for SNAP visualization but may
        cause issues with band math expressions). Default: False
    """
    if not os.path.exists(hdr_path):
        print(f"    Warning: HDR file not found: {hdr_path}")
        return

    with open(hdr_path, 'r') as f:
        content = f.read()

    original_content = content

    # Extract wavelengths before removing them
    wavelength_match = re.search(r'wavelength\s*=\s*\{([^}]*)\}', content, re.IGNORECASE)
    wavelengths = []
    if wavelength_match:
        wl_str = wavelength_match.group(1)
        wavelengths = [w.strip() for w in wl_str.split(',') if w.strip()]

    # If external wavelength file provided, load wavelengths from there
    if wavelength_file and os.path.exists(wavelength_file):
        print(f"    Loading wavelengths from external file: {wavelength_file}")
        wavelengths = []
        fwhm_values = []

        with open(wavelength_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Try comma-separated format first (band_name, wavelength, fwhm)
                if ',' in line:
                    parts = line.split(',')
                    # If format is "band_X, wavelength, fwhm", take parts[1] as wavelength
                    if len(parts) >= 2:
                        wavelengths.append(parts[1].strip())
                        if len(parts) >= 3:
                            fwhm_values.append(parts[2].strip())
                    else:
                        # Fallback: single value
                        wavelengths.append(parts[0].strip())
                else:
                    # Single wavelength per line
                    wavelengths.append(line)

        print(f"    Loaded {len(wavelengths)} wavelengths from file")
    else:
        # Extract fwhm before removing (only if not loading from file)
        fwhm_match = re.search(r'fwhm\s*=\s*\{([^}]*)\}', content, re.IGNORECASE)
        fwhm_values = []
        if fwhm_match:
            fwhm_str = fwhm_match.group(1)
            fwhm_values = [f.strip() for f in fwhm_str.split(',') if f.strip()]

    # Save wavelength and fwhm to a separate metadata file for reference
    if wavelengths:
        metadata_path = hdr_path.replace('.hdr', '_spectral_info.txt')
        with open(metadata_path, 'w') as f:
            f.write("# Spectral information (removed from HDR for SNAP compatibility)\n")
            f.write("# Band, Wavelength (nm), FWHM (nm)\n")
            for i, wl in enumerate(wavelengths):
                fwhm = fwhm_values[i] if i < len(fwhm_values) else "N/A"
                f.write(f"band_{i+1}, {wl}, {fwhm}\n")

    # Generate simple band names (band_1, band_2, etc.) - no wavelength to avoid SNAP issues
    if wavelengths:
        band_names = [f'band_{i+1}' for i in range(len(wavelengths))]
        new_band_names = ' , '.join(band_names)
        if re.search(r'band names\s*=', content, re.IGNORECASE):
            content = re.sub(r'band names\s*=\s*\{[^}]*\}', f'band names = {{ {new_band_names} }}', content, flags=re.IGNORECASE)
        else:
            content = re.sub(r'(bands\s*=\s*\d+)', f'\\1\nband names = {{ {new_band_names} }}', content)
    else:
        # Fallback: clean existing band names
        def fix_band_names(match):
            field_name = match.group(1)
            field_content = match.group(2)
            fixed_content = re.sub(r'\([^)]+\):[^,}]+', lambda m: m.group(0).split(':')[-1].strip(), field_content)
            fixed_content = re.sub(r'\([^)]*\)', '', fixed_content)
            parts = [p.strip().replace(' ', '_') for p in fixed_content.split(',')]
            fixed_content = ' , '.join(parts)
            return f'{field_name} = {{ {fixed_content} }}'
        content = re.sub(r'(band names)\s*=\s*\{([^}]*)\}', fix_band_names, content, flags=re.IGNORECASE)

    # Handle wavelength field based on keep_wavelength flag
    if keep_wavelength and wavelengths:
        # Keep/add wavelength field for SNAP (better visualization)
        print("    Keeping wavelength field in HDR for SNAP compatibility")
        wavelength_str = ' , '.join(wavelengths)

        if re.search(r'wavelength\s*=', content, re.IGNORECASE):
            # Update existing wavelength field
            content = re.sub(r'wavelength\s*=\s*\{[^}]*\}', f'wavelength = {{ {wavelength_str} }}', content, flags=re.IGNORECASE)
        else:
            # Add wavelength field after band names
            if re.search(r'band names\s*=', content, re.IGNORECASE):
                content = re.sub(r'(band names\s*=\s*\{[^}]*\})', f'\\1\nwavelength = {{ {wavelength_str} }}', content, flags=re.IGNORECASE)
            else:
                content = re.sub(r'(bands\s*=\s*\d+)', f'\\1\nwavelength = {{ {wavelength_str} }}', content)

        # Also add/update FWHM if available
        if fwhm_values:
            fwhm_str = ' , '.join(fwhm_values)
            if re.search(r'fwhm\s*=', content, re.IGNORECASE):
                content = re.sub(r'fwhm\s*=\s*\{[^}]*\}', f'fwhm = {{ {fwhm_str} }}', content, flags=re.IGNORECASE)
            else:
                content = re.sub(r'(wavelength\s*=\s*\{[^}]*\})', f'\\1\nfwhm = {{ {fwhm_str} }}', content, flags=re.IGNORECASE)
    else:
        # CRITICAL: Remove wavelength field - this is what causes SNAP to append (wavelength) to band names
        content = re.sub(r'\nwavelength\s*=\s*\{[^}]*\}', '', content, flags=re.IGNORECASE)

        # Also remove fwhm as it's not needed for SNAP viewing and can cause issues
        content = re.sub(r'\nfwhm\s*=\s*\{[^}]*\}', '', content, flags=re.IGNORECASE)

    # Clean up scale factor field formatting if present
    scale_match = re.search(r'(scale factor)\s*=\s*\{([^}]*)\}', content, re.IGNORECASE)
    if scale_match:
        field_name = scale_match.group(1)
        values = scale_match.group(2)
        values = re.sub(r'\s+', ' ', values).strip()
        content = re.sub(r'scale factor\s*=\s*\{[^}]*\}', f'{field_name} = {{ {values} }}', content, flags=re.IGNORECASE)

    # Write back only if changes were made
    if content != original_content:
        # Create backup
        backup_path = hdr_path + '.backup'
        if not os.path.exists(backup_path):
            with open(backup_path, 'w') as f:
                f.write(original_content)

        with open(hdr_path, 'w') as f:
            f.write(content)
        print(f"    Fixed HDR file for SNAP compatibility: {hdr_path}")
        if wavelengths:
            print(f"    Spectral info saved to: {hdr_path.replace('.hdr', '_spectral_info.txt')}")
        if keep_wavelength:
            print(f"    Wavelength field kept in HDR for visualization")
        else:
            print(f"    Wavelength field removed from HDR to avoid band math issues")
    else:
        print(f"    HDR file already SNAP-compatible: {hdr_path}")


def parse_hyperion_xml(xml_path):
    """
    Parse Hyperion XML metadata file from USGS EarthExplorer

    Parameters:
    -----------
    xml_path : str
        Path to the XML metadata file

    Returns:
    --------
    dict : Dictionary containing all metadata fields
    """
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Define namespace
    ns = {'eemetadata': 'http://earthexplorer.usgs.gov/eemetadata.xsd'}

    # Extract metadata fields
    metadata = {}
    for field in root.findall('.//eemetadata:metadataField', ns):
        field_name = field.get('name')
        field_value = field.find('eemetadata:metadataValue', ns).text
        metadata[field_name] = field_value

    return metadata


def convert_time_format(time_str):
    """
    Convert time from HH:MM:SS.SSS to HH:MM:SS format
    Example: 2013:284:10:09:05.020 → 2013:284:10:09:05
    """
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) >= 4:
            # Remove milliseconds from seconds
            seconds = parts[3].split('.')[0]
            return "{0}:{1}:{2}:{3}".format(parts[0], parts[1], parts[2], seconds)
    return time_str


def create_metadata_csv(xml_path, output_csv_path, append=True):
    """
    Create or append to metadata CSV file from XML

    Parameters:
    -----------
    xml_path : str
        Path to the XML metadata file
    output_csv_path : str
        Path for the output CSV file
    append : bool
        If True, append to existing CSV. If False, create new CSV.
    """
    # Parse XML
    metadata = parse_hyperion_xml(xml_path)

    # Extract Scene ID from Entity ID (remove _SG1_01 suffix)
    entity_id = metadata['Entity ID']
    scene_id = entity_id[:-7] if entity_id.endswith('_SG1_01') else entity_id

    # Create DataFrame with required columns (matching SUREHYP expectations)
    df_new = pd.DataFrame({
        'Entity ID': [metadata['Entity ID']],
        'Scene ID': [scene_id],
        'Acquisition Date': [metadata['Acquisition Date']],
        'Cloud Cover': [int(metadata['Cloud Cover'])],
        'Orbit Path': [int(metadata['Orbit Path'])],
        'Orbit Row': [int(metadata['Orbit Row'])],
        'Target Path': [int(metadata['Target Path'])],
        'Target Row': [int(metadata['Target Row'])],
        'Station': [metadata['Station']],
        'Processing Level': [metadata['Processing Level']],
        'Scene Start Time': [convert_time_format(metadata['Scene Start Time'])],
        'Start Time': [metadata['Scene Start Time'].split(':')[-1].split('.')[0] if ':' in metadata['Scene Start Time'] else 'N/A'],
        'Scene Stop Time': [convert_time_format(metadata['Scene Stop Time'])],
        'Stop Time': [metadata['Scene Stop Time'].split(':')[-1].split('.')[0] if ':' in metadata['Scene Stop Time'] else 'N/A'],
        'Sun Azimuth': [float(metadata['Sun Azimuth'])],
        'Sun Elevation': [float(metadata['Sun Elevation'])],
        'Satellite Inclination': [float(metadata['Satellite Inclination'])],
        'Look Angle': [float(metadata['Look Angle'])],
        'Date Entered': [metadata['Date Entered']],
        'Center Latitude': [metadata['Center Latitude']],
        'Center Longitude': [metadata['Center Longitude']],
        'Center Latitude dec': [float(metadata['Center Latitude dec'])],
        'Center Longtude dec': [float(metadata['Center Longtude dec'])],  # Note: typo 'Longtude' matches SUREHYP
        'NW Corner Lat': [metadata['NW Corner Lat']],
        'NW Corner Long': [metadata['NW Corner Long']],
        'NE Corner Lat': [metadata['NE Corner Lat']],
        'NE Corner Long': [metadata['NE Corner Long']],
        'SE Corner Lat': [metadata['SE Corner Lat']],
        'SE Corner Long': [metadata['SE Corner Long']],
        'SW Corner Lat': [metadata['SW Corner Lat']],
        'SW Corner Long': [metadata['SW Corner Long']],
        'NW Corner Lat dec': [float(metadata['NW Corner Lat dec'])],
        'NW Corner Long dec': [float(metadata['NW Corner Long dec'])],
        'NE Corner Lat dec': [float(metadata['NE Corner Lat dec'])],
        'NE Corner Long dec': [float(metadata['NE Corner Long dec'])],
        'SE Corner Lat dec': [float(metadata['SE Corner Lat dec'])],
        'SE Corner Long dec': [float(metadata['SE Corner Long dec'])],
        'SW Corner Lat dec': [float(metadata['SW Corner Lat dec'])],
        'SW Corner Long dec': [float(metadata['SW Corner Long dec'])],
        'Satellite Azimuth': [float(metadata.get('Satellite Azimuth', 0.0))]  # Not always in XML
    })

    # Append or create new CSV
    if append and os.path.exists(output_csv_path):
        # Read existing CSV
        df_existing = pd.read_csv(output_csv_path)

        # Check if this Entity ID already exists
        if entity_id in df_existing['Entity ID'].values:
            print("    Warning: {} already exists in metadata CSV".format(entity_id))
            print("    Skipping to avoid duplicates.")
            return

        # Append new row
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_csv_path, index=False)
        print("    Appended {} to {}".format(entity_id, output_csv_path))
    else:
        # Create new CSV
        df_new.to_csv(output_csv_path, index=False)
        print("    Created new CSV: {}".format(output_csv_path))

    # Print summary
    print("\n    Metadata Summary for {}:".format(scene_id))
    print("      Acquisition Date: {}".format(metadata['Acquisition Date']))
    print("      Sun Elevation: {}°".format(metadata['Sun Elevation']))
    print("      Sun Azimuth: {}°".format(metadata['Sun Azimuth']))
    print("      Look Angle: {}°".format(metadata['Look Angle']))
    print("      Center Coordinates: ({}, {})".format(metadata['Center Latitude dec'], metadata['Center Longtude dec']))
    print("      Cloud Cover: {}%".format(metadata['Cloud Cover']))


def getGEEdem_fixed(UL_lat, UL_lon, UR_lat, UR_lon, LL_lat, LL_lon, LR_lat, LR_lon,
                    demID='USGS/SRTMGL1_003', elevationName='elevation', numPixels=1000):
    """
    Fixed version of getGEEdem that handles SRTM as Image instead of ImageCollection.
    Gets the average elevation of the scene from Google Earth Engine.
    """
    # Define the bounding box
    coord = ee.Geometry.Polygon([
        [UL_lon, UL_lat],
        [UR_lon, UR_lat],
        [LR_lon, LR_lat],
        [LL_lon, LL_lat],
        [UL_lon, UL_lat]
    ])

    # SRTM is an Image, not an ImageCollection
    DEM = ee.Image(demID)

    # Sample the DEM and get mean elevation
    result = DEM.sample(region=coord, numPixels=numPixels, scale=1000).getInfo()

    if result['features']:
        elevations = [f['properties'][elevationName] for f in result['features'] if elevationName in f['properties']]
        if elevations:
            altit = np.mean(elevations) / 1000.0  # Convert to km
            return altit

    # Fallback: try reduceRegion
    mean_elev = DEM.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=coord,
        scale=1000,
        maxPixels=1e9
    ).getInfo()

    if elevationName in mean_elev and mean_elev[elevationName] is not None:
        return mean_elev[elevationName] / 1000.0  # Convert to km

    raise ValueError(f"Could not retrieve elevation data from {demID}")


def downloadDEMfromGEE(UL_lon, UL_lat, UR_lon, UR_lat, LR_lon, LR_lat, LL_lon, LL_lat,
                       demID='USGS/SRTMGL1_003', elevationName='elevation', output_path=None):
    """
    Download DEM from Google Earth Engine using the new API (compatible with geetools >= 1.0).
    Uses ee.Image.getDownloadURL() instead of geetools.batch.image.toLocal()
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

    # Try loading as Image first, then as ImageCollection
    try:
        elev = ee.Image(demID)
        elev = elev.select(elevationName)
        elev.getInfo()  # Will fail if not an image
    except:
        dem = ee.ImageCollection(demID)
        dem = dem.select(elevationName)
        elev = dem.mosaic()

    # Clip to region
    elev = elev.clip(region)

    # Get download URL
    try:
        url = elev.getDownloadURL({
            'name': 'elev',
            'scale': 30,
            'region': region,
            'format': 'GEO_TIFF'
        })

        print(f'    Downloading DEM from GEE...')
        response = requests.get(url)

        if response.status_code == 200:
            output_file = os.path.join(output_path, 'elev.tif')
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f'    DEM saved to: {output_file}')
            return output_path
        else:
            raise Exception(f"Download failed with status code: {response.status_code}")

    except Exception as e:
        print(f'    Warning: Direct download failed ({e}), trying alternative method...')

        # Alternative: Use computePixels for smaller regions
        try:
            # Get bounding box coordinates
            bounds = region.bounds().getInfo()['coordinates'][0]
            min_lon = min(p[0] for p in bounds)
            max_lon = max(p[0] for p in bounds)
            min_lat = min(p[1] for p in bounds)
            max_lat = max(p[1] for p in bounds)

            # Create a grid request
            url = elev.getThumbURL({
                'min': 0,
                'max': 5000,
                'dimensions': 1024,
                'region': region,
                'format': 'png'
            })

            # For actual DEM data, we'll sample at higher resolution
            scale = 30  # meters

            # Sample the DEM
            samples = elev.sample(
                region=region,
                scale=scale,
                geometries=True
            ).getInfo()

            if samples['features']:
                # Create a raster from samples
                print(f'    Creating DEM from {len(samples["features"])} sample points...')

                # Get coordinates and elevations
                points = []
                for f in samples['features']:
                    coords = f['geometry']['coordinates']
                    elev_val = f['properties'].get(elevationName, f['properties'].get('elevation', 0))
                    points.append([coords[0], coords[1], elev_val])

                points = np.array(points)

                # Create a simple gridded DEM using scipy
                from scipy.interpolate import griddata

                # Define output grid
                grid_x = np.linspace(min_lon, max_lon, 500)
                grid_y = np.linspace(min_lat, max_lat, 500)
                grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

                # Interpolate
                grid_z = griddata(points[:, :2], points[:, 2], (grid_xx, grid_yy), method='linear')
                grid_z = np.flipud(grid_z)  # Flip to match raster convention

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
                ) as dst:
                    dst.write(grid_z, 1)

                print(f'    DEM saved to: {output_file}')
                return output_path

        except Exception as e2:
            print(f'    Error in alternative method: {e2}')
            raise


def processImage_fixed(fname, pathToImages, pathToImagesFiltered):
    """
    Fixed version of processImage that doesn't use tiled output
    (fixes the "TileWidth must be multiple of 16" error)
    """
    Path(pathToImages + 'tmp/').mkdir(parents=True, exist_ok=True)
    Path(pathToImagesFiltered).mkdir(parents=True, exist_ok=True)

    fpath = pathToImages + "/" + fname + '_1T.ZIP'
    with ZipFile(fpath, 'r') as zip:
        namelist = zip.namelist()
        namelist = [string for string in namelist if 'TIF' in string]
        zip.extractall(path=pathToImages + 'tmp/')
    namelist = [string for string in namelist if 'TIF' in string]

    # Read all bands files separately
    arrays = []
    for name in namelist:
        tmpImg = pathToImages + 'tmp/' + name
        src = rasterio.open(tmpImg, driver='GTiff', dtype=rasterio.int16)
        arrays.append(src.read(1))

    # Write all bands to a new file (without tiling to avoid dimension issues)
    profile = src.profile
    profile.update(count=len(arrays), nodata=0, tiled=False)  # Changed: tiled=False

    with rasterio.open(pathToImagesFiltered + fname + '.TIF', 'w', compress='lzw', **profile) as dst:
        k = 1
        for array in arrays:
            dst.write(array.astype(rasterio.int16), k)
            k += 1

    # Clean up tmp folder
    import shutil
    shutil.rmtree(pathToImages + 'tmp/', ignore_errors=True)


def preprocess_radiance(fname, pathToL1Rmetadata, pathToL1Rimages, pathToL1Timages,
                        pathToL1TimagesFiltered, pathOut, nameOut,
                        destripingMethod='Pal', localDestriping=False,
                        smileCorrectionOrder=2, checkSmile=False):
    """
    Preprocess L1R radiance data: desmiling, destriping, georeferencing
    """
    print('=' * 60)
    print('STEP 1: PREPROCESSING RADIANCE')
    print('=' * 60)

    print('\n[1/10] Concatenate the L1T image')
    processImage_fixed(fname, pathToL1Timages, pathToL1TimagesFiltered)

    print('\n[2/10] Read the L1R image')
    arrayL1R = surehyp.preprocess.readL1R(pathToL1Rimages + fname + '/', fname)

    print('\n[3/10] Get the L1R image parameters')
    metadata, bands, fwhms = surehyp.preprocess.getImageMetadata(pathToL1Rimages + fname + '/', fname)

    print('\n[4/10] Separate VNIR and SWIR')
    VNIR, VNIRb, VNIRfwhm, SWIR, SWIRb, SWIRfwhm = surehyp.preprocess.separating(arrayL1R, bands, fwhms)

    print('\n[5/10] Convert DN to radiance')
    VNIR, SWIR = surehyp.preprocess.DN2Radiance(VNIR, SWIR)

    print('\n[6/10] Align VNIR and SWIR, part 1')
    VNIR, SWIR = surehyp.preprocess.alignSWIR2VNIRpart1(VNIR, SWIR)

    print('\n[7/10] Desmiling')
    VNIR = surehyp.preprocess.smileCorrectionAll(VNIR, smileCorrectionOrder, check=checkSmile)
    SWIR = surehyp.preprocess.smileCorrectionAll(SWIR, smileCorrectionOrder, check=checkSmile)

    if destripingMethod == 'Datt':
        print('\n[8/10] Destriping - Datt (2003)')
        VNIR = surehyp.preprocess.destriping(VNIR, 'VNIR', 0.11)
        SWIR = surehyp.preprocess.destriping(SWIR, 'SWIR', 0.11)
    elif destripingMethod == 'Pal':
        print('\n[8/10] Destriping - Pal et al. (2020)')
        VNIR, nc = surehyp.preprocess.destriping_quadratic(VNIR)
        if localDestriping:
            VNIR = surehyp.preprocess.destriping_local(VNIR, nc)
        SWIR, nc = surehyp.preprocess.destriping_quadratic(SWIR)
        if localDestriping:
            SWIR = surehyp.preprocess.destriping_local(SWIR, nc)
    else:
        print('\n[8/10] No destriping method selected -> skipping')

    print('\n[9/10] Assemble VNIR and SWIR')
    arrayL1R, wavelengths, fwhms = surehyp.preprocess.concatenateImages(VNIR, VNIRb, VNIRfwhm, SWIR, SWIRb, SWIRfwhm)

    print('\n[9b/10] Smooth the cirrus bands for later thin cirrus removal')
    arrayL1R = surehyp.preprocess.smoothCirrusBand(arrayL1R, wavelengths)

    print('\n[10/10] Georeference the corrected L1R data using L1T data')
    arrayL1Rgeoreferenced, metadataGeoreferenced = surehyp.preprocess.georeferencing(arrayL1R, pathToL1TimagesFiltered, fname)

    print('\nSaving the processed image as an ENVI file...')
    surehyp.preprocess.savePreprocessedL1R(arrayL1Rgeoreferenced, wavelengths, fwhms, metadataGeoreferenced,
                                            pathToL1Rimages, pathToL1Rmetadata, metadata, fname, pathOut + nameOut)

    # Fix HDR file for SNAP compatibility (uses parameters defined in main configuration)
    # Note: snap_wavelength_file and snap_keep_wavelength are passed from main()
    fix_envi_hdr_for_snap(pathOut + nameOut + '.hdr',
                          wavelength_file=None,  # Uses computed wavelengths
                          keep_wavelength=False)  # Default: remove for safety

    # Cleanup temporary files
    for f in os.listdir(pathOut):
        if (fname in f) and ('_tmp' in f):
            os.remove(os.path.join(pathOut, f))

    print('\n' + '=' * 60)
    print('PREPROCESSING COMPLETE!')
    print(f'Output saved to: {pathOut + nameOut}')
    print('=' * 60)

    return pathOut + nameOut


def atmospheric_correction(pathToRadianceImage, pathToOutImage, stepAltit=1, stepTilt=15,
                           stepWazim=30, demID='USGS/SRTMGL1_003', elevationName='elevation',
                           topo=True, smartsAlbedoFilePath=None,
                           snap_wavelength_file=None, snap_keep_wavelength=False):
    """
    Perform atmospheric correction with optional topographic correction
    """
    print('\n' + '=' * 60)
    print('STEP 2: ATMOSPHERIC CORRECTION')
    print('=' * 60)

    print('\n[1/12] Open processed radiance image')

    # Try to load the image with surehyp's function
    try:
        L, bands, fwhms, processing_metadata, metadata = surehyp.atmoCorrection.getImageAndParameters(pathToRadianceImage)
    except KeyError as e:
        if 'wavelength' in str(e):
            # If wavelength is missing from HDR, load it from the spectral info file
            print('    Wavelength field not found in HDR, loading from spectral info file...')

            # Determine HDR path
            hdr_path = pathToRadianceImage + '.hdr'

            # Load spectral info file
            spectral_info_path = pathToRadianceImage + '_spectral_info.txt'
            if os.path.exists(spectral_info_path):
                wavelengths = []
                fwhms_list = []
                with open(spectral_info_path, 'r') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            wavelengths.append(float(parts[1].strip()))
                        if len(parts) >= 3 and parts[2].strip() != 'N/A':
                            fwhms_list.append(float(parts[2].strip()))

                # Temporarily restore wavelength to HDR for compatibility with surehyp
                with open(hdr_path, 'r') as f:
                    hdr_content = f.read()

                # Add wavelength field
                wavelength_str = ' , '.join(str(w) for w in wavelengths)
                if 'band names' in hdr_content.lower():
                    hdr_content = re.sub(r'(band names\s*=\s*\{[^}]*\})',
                                        f'\\1\nwavelength = {{ {wavelength_str} }}',
                                        hdr_content, flags=re.IGNORECASE)
                else:
                    hdr_content = re.sub(r'(bands\s*=\s*\d+)',
                                        f'\\1\nwavelength = {{ {wavelength_str} }}',
                                        hdr_content)

                # Add FWHM field if available
                if fwhms_list:
                    fwhm_str = ' , '.join(str(f) for f in fwhms_list)
                    hdr_content = re.sub(r'(wavelength\s*=\s*\{[^}]*\})',
                                        f'\\1\nfwhm = {{ {fwhm_str} }}',
                                        hdr_content, flags=re.IGNORECASE)

                # Write back temporarily
                with open(hdr_path, 'w') as f:
                    f.write(hdr_content)

                print(f'    Restored wavelengths from: {spectral_info_path}')

                # Now try loading again
                L, bands, fwhms, processing_metadata, metadata = surehyp.atmoCorrection.getImageAndParameters(pathToRadianceImage)
            else:
                raise ValueError(f"Wavelength field missing from HDR and spectral info file not found: {spectral_info_path}")
        else:
            raise

    # Extract metadata for clearer visualization
    longit = processing_metadata['longit']
    latit = processing_metadata['latit']
    datestamp1 = processing_metadata['datestamp1']
    zenith = processing_metadata['zenith']
    azimuth = processing_metadata['azimuth']
    satelliteZenith = np.abs(processing_metadata['satelliteZenith'])
    satelliteAzimuth = processing_metadata['satelliteAzimuth']

    UL_lat = processing_metadata['UL_lat']
    UL_lon = processing_metadata['UL_lon']
    UR_lat = processing_metadata['UR_lat']
    UR_lon = processing_metadata['UR_lon']
    LL_lat = processing_metadata['LL_lat']
    LL_lon = processing_metadata['LL_lon']
    LR_lat = processing_metadata['LR_lat']
    LR_lon = processing_metadata['LR_lon']

    year = processing_metadata['year']
    doy = processing_metadata['doy']

    print(f'    Image center: ({latit:.4f}, {longit:.4f})')
    print(f'    Year: {year}, DOY: {doy}')
    print(f'    Sun zenith: {zenith:.2f}, Sun azimuth: {azimuth:.2f}')

    # Variables for topographic correction
    elev = None
    slope = None
    wazim = None

    if topo:
        print('\n[2/12] Download DEM images from GEE')
        try:
            # Set output path for DEM in the OUT folder
            dem_output_path = os.path.dirname(pathToOutImage) + '/elev/'

            # Try using our fixed download function
            path_to_dem = downloadDEMfromGEE(UL_lon, UL_lat, UR_lon, UR_lat,
                                              LR_lon, LR_lat, LL_lon, LL_lat,
                                              demID=demID, elevationName=elevationName,
                                              output_path=dem_output_path)

            print('\n[3/12] Reproject DEM images')
            # reprojectDEM expects a file path, not a folder
            dem_file_path = os.path.join(path_to_dem, 'elev.tif')
            reprojected_dem_path = os.path.join(path_to_dem, 'elev_reprojected.tif')
            path_to_reprojected_dem = surehyp.atmoCorrection.reprojectDEM(
                pathToRadianceImage,
                path_elev=dem_file_path,
                path_elev_out=reprojected_dem_path
            )

            print('\n[4/12] Resampling DEM')
            resampled_dem_path = os.path.join(dem_output_path, 'elev_resampled.tif')
            path_elev = surehyp.atmoCorrection.matchResolution(
                pathToRadianceImage,
                path_elev=path_to_reprojected_dem,
                path_out=resampled_dem_path
            )

            print("\n[5/12] Extract DEM data for Hyperion image pixels")
            elev, slope, wazim = surehyp.atmoCorrection.extractDEMdata(pathToRadianceImage, path_elev=path_elev)
        except Exception as e:
            print(f'\n    WARNING: Topographic correction failed: {e}')
            print('    Continuing without topographic correction...')
            topo = False
            slope = None
            wazim = None

    # Cloud/shadow detection
    print('\n[6/12] Cloud/shadow detection (skipped - using all pixels as clearview)')
    clearview = np.ones(L.shape[:2], dtype=np.uint8)
    print('    All pixels marked as clearview')

    print('\n[7/12] Get haze spectrum (dark object subtraction)')
    L, Lhaze = surehyp.atmoCorrection.darkObjectDehazing(L, bands)

    print('\n[8/12] Mask non-clearview pixels (skipped)')
    # L[clearview == 0] = 0  # Skipped since all pixels are clearview

    print('\n[9/12] Removal of thin cirrus')
    L, cirrus_cloudMask = surehyp.atmoCorrection.cirrusRemoval(bands, L, latit, doy, satelliteZenith, zenith, azimuth)

    print('\n[10/12] Get average elevation of the scene from GEE')
    # Use fixed version that handles SRTM as Image (not ImageCollection)
    altit = getGEEdem_fixed(UL_lat, UL_lon, UR_lat, UR_lon,
                            LL_lat, LL_lon, LR_lat, LR_lon,
                            demID=demID, elevationName=elevationName)
    print(f'    Average altitude: {altit*1000:.1f} m ({altit:.3f} km)')

    print('\n[11/12] Get atmospheric parameters')
    try:
        wv, o3, flag_no_o3 = surehyp.atmoCorrection.getAtmosphericParameters(bands, L, datestamp1, year, doy,
                                                                              longit, latit, altit,
                                                                              satelliteZenith, zenith, azimuth)
        print(f'    Water vapor: {wv:.2f} cm')
        print(f'    Ozone: {o3:.3f} atm-cm')
        if flag_no_o3:
            IO3 = 1
        else:
            IO3 = 0
    except (ValueError, Exception) as e:
        print(f'    WARNING: Could not compute atmospheric parameters from image: {e}')
        print('    Using default values for mid-latitude atmosphere')
        # Use typical mid-latitude winter values (Dec 24)
        # Water vapor ~ 1.5 cm for winter, Ozone ~ 0.35 atm-cm
        wv = 1.5  # precipitable water vapor in cm
        o3 = 0.35  # ozone column in atm-cm
        IO3 = 0  # Use the specified ozone value (not SMARTS default)
        print(f'    Water vapor (default): {wv:.2f} cm')
        print(f'    Ozone (default): {o3:.3f} atm-cm')

    print('\n[12/12] Obtain radiative transfer outputs from SMARTS')
    # Get atmosphere parameters for sun-ground section
    df = surehyp.atmoCorrection.runSMARTS(ALTIT=altit, LATIT=latit, IMASS=0, ZENITH=zenith,
                                           AZIM=azimuth, SUNCOR=surehyp.atmoCorrection.get_SUNCOR(doy),
                                           IH2O=0, WV=wv, IO3=IO3, IALT=0, AbO3=o3)

    # Get atmosphere parameters for ground-satellite section
    df_gs = surehyp.atmoCorrection.runSMARTS(ALTIT=altit, LATIT=0, LONGIT=0, IMASS=0,
                                              SUNCOR=surehyp.atmoCorrection.get_SUNCOR(doy),
                                              ITURB=5, ZENITH=satelliteZenith, AZIM=0,
                                              IH2O=0, WV=wv, IO3=IO3, IALT=0, AbO3=o3)

    print('\nComputing radiance to reflectance conversion...')
    R = surehyp.atmoCorrection.computeLtoR(L, bands, df, df_gs)

    if not topo:
        print('\nSaving the reflectance image (flat surface)...')
        surehyp.atmoCorrection.saveRimage(R, metadata, pathToOutImage)
        # Fix HDR file for SNAP compatibility
        fix_envi_hdr_for_snap(pathToOutImage + '.hdr',
                              wavelength_file=snap_wavelength_file,
                              keep_wavelength=snap_keep_wavelength)
    else:
        print('\n--- TOPOGRAPHIC CORRECTION ---')

        if smartsAlbedoFilePath is None:
            smartsAlbedoFilePath = os.environ['SMARTSPATH'] + 'Albedo/Albedo.txt'

        print('\nWriting Albedo.txt file for SMARTS')
        pathToAlbedoFile = surehyp.atmoCorrection.writeAlbedoFile(R, bands, pathOut=smartsAlbedoFilePath)

        print('\nGetting scene background reflectance')
        sp = pd.read_csv(pathToAlbedoFile, header=3, sep=r'\s+')
        w = sp.values[:, 0]
        r = sp.values[:, 1]
        f = interpolate.interp1d(w, r, bounds_error=False, fill_value='extrapolate')
        rho_background = f(df['Wvlgth'] * 1E-3)

        print('\nComputing LUT for rough terrain correction')

        # Check and clip terrain parameters to ensure they're within LUT bounds
        # The LUT is built with ranges: altit ± stepAltit, tilt: 0 to ~90, wazim: 0 to 360
        elev_km = elev / 1000.0  # Convert elevation from meters to km

        # Calculate LUT bounds based on scene average altitude and step sizes
        altit_min = max(0, altit - stepAltit)
        altit_max = altit + stepAltit

        # Check for NaN values and handle them
        elev_has_nan = np.isnan(elev_km).any()
        slope_has_nan = np.isnan(slope).any()
        wazim_has_nan = np.isnan(wazim).any()

        if elev_has_nan or slope_has_nan or wazim_has_nan:
            print(f'    WARNING: DEM data contains NaN values!')
            print(f'    Elevation NaN: {np.isnan(elev_km).sum()} pixels')
            print(f'    Slope NaN: {np.isnan(slope).sum()} pixels')
            print(f'    Aspect NaN: {np.isnan(wazim).sum()} pixels')
            print(f'    These are likely ocean/water pixels where DEM has no data')
            print(f'    Filling NaN values with sea level (0m) and flat terrain...')

            # Create DEM visualization before filling NaN values
            import matplotlib.pyplot as plt

            # Save DEM visualization
            dem_vis_path = os.path.dirname(pathToOutImage) + '/elev/'
            Path(dem_vis_path).mkdir(parents=True, exist_ok=True)

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # 1. Elevation map
            im1 = axes[0, 0].imshow(elev_km * 1000, cmap='terrain', vmin=0, vmax=np.nanmax(elev_km)*1000)
            axes[0, 0].set_title('Elevation (m above sea level)')
            axes[0, 0].axis('off')
            plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)', shrink=0.8)

            # 2. Elevation with NaN highlighted
            elev_display = elev_km.copy()
            elev_display[np.isnan(elev_display)] = -999  # Mark NaN as distinct value
            im2 = axes[0, 1].imshow(elev_display * 1000, cmap='terrain', vmin=-999, vmax=np.nanmax(elev_km)*1000)
            axes[0, 1].set_title('Elevation (NaN shown in blue/purple)')
            axes[0, 1].axis('off')
            plt.colorbar(im2, ax=axes[0, 1], label='Elevation (m)', shrink=0.8)

            # 3. Slope map
            im3 = axes[1, 0].imshow(slope, cmap='hot', vmin=0, vmax=90)
            axes[1, 0].set_title('Slope (degrees)')
            axes[1, 0].axis('off')
            plt.colorbar(im3, ax=axes[1, 0], label='Slope (°)', shrink=0.8)

            # 4. Aspect map
            im4 = axes[1, 1].imshow(wazim, cmap='hsv', vmin=0, vmax=360)
            axes[1, 1].set_title('Aspect (degrees from North)')
            axes[1, 1].axis('off')
            plt.colorbar(im4, ax=axes[1, 1], label='Aspect (°)', shrink=0.8)

            plt.tight_layout()
            dem_vis_file = dem_vis_path + 'DEM_visualization.png'
            plt.savefig(dem_vis_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'    DEM visualization saved to: {dem_vis_file}')

            # Fill NaN values with sea level and flat terrain
            # (appropriate for ocean/water pixels)
            if elev_has_nan:
                elev_km = np.nan_to_num(elev_km, nan=0.0)  # Sea level (0m)
            if slope_has_nan:
                slope = np.nan_to_num(slope, nan=0.0)  # Flat terrain
            if wazim_has_nan:
                wazim = np.nan_to_num(wazim, nan=0.0)  # North-facing

        # Print DEM statistics (using nanmin/nanmax to handle any remaining NaNs)
        print(f'    DEM elevation range: {np.nanmin(elev_km):.3f} to {np.nanmax(elev_km):.3f} km')
        print(f'    LUT altitude bounds: {altit_min:.3f} to {altit_max:.3f} km')
        print(f'    Slope range: {np.nanmin(slope):.1f}° to {np.nanmax(slope):.1f}°')
        print(f'    Aspect range: {np.nanmin(wazim):.1f}° to {np.nanmax(wazim):.1f}°')

        # Always clip to ensure values are within LUT bounds
        if np.nanmin(elev_km) < altit_min or np.nanmax(elev_km) > altit_max:
            print(f'    WARNING: Clipping elevation to LUT bounds...')
        elev_km = np.clip(elev_km, altit_min, altit_max)
        elev = elev_km * 1000.0  # Convert back to meters

        # Slope should be 0 to ~90 degrees
        tilt_max = 90
        if np.nanmin(slope) < 0 or np.nanmax(slope) > tilt_max:
            print(f'    WARNING: Clipping slope to 0-{tilt_max}°...')
        slope = np.clip(slope, 0, tilt_max)

        # Aspect (wazim) should be 0 to 360 degrees
        wazim = np.mod(wazim, 360)  # Normalize to 0-360
        # Handle any NaN that might result from mod operation
        wazim = np.nan_to_num(wazim, nan=0.0)

        print(f'    After processing - Elevation: {np.min(elev_km):.3f} to {np.max(elev_km):.3f} km')
        print(f'    After processing - Slope: {np.min(slope):.1f}° to {np.max(slope):.1f}°')
        print(f'    After processing - Aspect: {np.min(wazim):.1f}° to {np.max(wazim):.1f}°')

        R = surehyp.atmoCorrection.getDemReflectance(altitMap=elev, tiltMap=slope, wazimMap=wazim,
                                                      stepAltit=stepAltit, stepTilt=stepTilt,
                                                      stepWazim=stepWazim, latit=latit,
                                                      IH2O=0, WV=wv, IO3=IO3, IALT=0, AbO3=o3,
                                                      doy=doy, zenith=zenith, azimuth=azimuth,
                                                      satelliteZenith=satelliteZenith,
                                                      satelliteAzimuth=satelliteAzimuth,
                                                      L=L, bands=bands, IALBDX=1,
                                                      rho_background=rho_background)

        print('\nApplying Modified-Minnaert topography correction')
        R = surehyp.atmoCorrection.MM_topo_correction(R, bands, slope * np.pi / 180,
                                                       wazim * np.pi / 180, zenith * np.pi / 180,
                                                       azimuth * np.pi / 180)

        print('\nSaving the reflectance image (with topographic correction)...')
        surehyp.atmoCorrection.saveRimage(R, metadata, pathToOutImage)
        # Fix HDR file for SNAP compatibility
        fix_envi_hdr_for_snap(pathToOutImage + '.hdr',
                              wavelength_file=snap_wavelength_file,
                              keep_wavelength=snap_keep_wavelength)

    # Save masks
    pathOutDir = os.path.dirname(pathToOutImage) + '/'
    np.save(pathOutDir + os.path.basename(pathToOutImage) + '_clearview_mask.npy', clearview)
    np.save(pathOutDir + os.path.basename(pathToOutImage) + '_cirrus_mask.npy', cirrus_cloudMask)

    print('\n' + '=' * 60)
    print('ATMOSPHERIC CORRECTION COMPLETE!')
    print(f'Output saved to: {pathToOutImage}')
    print('=' * 60)

    return pathToOutImage, R, bands


def create_rgb_quicklook(R, bands, output_path, stretch_percentile=2):
    """
    Create an RGB quicklook image from reflectance data.

    Parameters:
    -----------
    R : numpy.ndarray
        Reflectance array (rows, cols, bands)
    bands : numpy.ndarray
        Wavelengths in nm
    output_path : str
        Path to save the RGB image
    stretch_percentile : float
        Percentile for contrast stretch (default 2%)
    """
    import matplotlib.pyplot as plt

    # Find RGB band indices (approximate wavelengths)
    red_idx = np.argmin(np.abs(bands - 660))    # ~660 nm
    green_idx = np.argmin(np.abs(bands - 550))  # ~550 nm
    blue_idx = np.argmin(np.abs(bands - 480))   # ~480 nm

    # Extract RGB bands
    red = R[:, :, red_idx].astype(float)
    green = R[:, :, green_idx].astype(float)
    blue = R[:, :, blue_idx].astype(float)

    # Apply contrast stretch
    def stretch(band, percentile):
        valid = band[band > 0]
        if len(valid) == 0:
            return band
        p_low = np.percentile(valid, percentile)
        p_high = np.percentile(valid, 100 - percentile)
        band = np.clip(band, p_low, p_high)
        band = (band - p_low) / (p_high - p_low) if p_high > p_low else band
        return band

    red = stretch(red, stretch_percentile)
    green = stretch(green, stretch_percentile)
    blue = stretch(blue, stretch_percentile)

    # Stack into RGB
    rgb = np.dstack([red, green, blue])
    rgb = np.clip(rgb, 0, 1)

    # Save image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.axis('off')
    plt.title('RGB Quicklook (R: 660nm, G: 550nm, B: 480nm)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'    RGB quicklook saved to: {output_path}')
    return rgb


def create_false_color_quicklook(R, bands, output_path, stretch_percentile=2):
    """
    Create a false color (NIR-R-G) quicklook image.

    Parameters:
    -----------
    R : numpy.ndarray
        Reflectance array (rows, cols, bands)
    bands : numpy.ndarray
        Wavelengths in nm
    output_path : str
        Path to save the image
    """
    import matplotlib.pyplot as plt

    # Find band indices
    nir_idx = np.argmin(np.abs(bands - 860))    # ~860 nm (NIR)
    red_idx = np.argmin(np.abs(bands - 660))    # ~660 nm
    green_idx = np.argmin(np.abs(bands - 550))  # ~550 nm

    # Extract bands
    nir = R[:, :, nir_idx].astype(float)
    red = R[:, :, red_idx].astype(float)
    green = R[:, :, green_idx].astype(float)

    # Apply contrast stretch
    def stretch(band, percentile):
        valid = band[band > 0]
        if len(valid) == 0:
            return band
        p_low = np.percentile(valid, percentile)
        p_high = np.percentile(valid, 100 - percentile)
        band = np.clip(band, p_low, p_high)
        band = (band - p_low) / (p_high - p_low) if p_high > p_low else band
        return band

    nir = stretch(nir, stretch_percentile)
    red = stretch(red, stretch_percentile)
    green = stretch(green, stretch_percentile)

    # Stack into RGB (NIR-R-G false color)
    rgb = np.dstack([nir, red, green])
    rgb = np.clip(rgb, 0, 1)

    # Save image
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.axis('off')
    plt.title('False Color Composite (NIR-R-G)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'    False color quicklook saved to: {output_path}')
    return rgb


def compute_ndvi(R, bands):
    """
    Compute NDVI from reflectance data.

    Parameters:
    -----------
    R : numpy.ndarray
        Reflectance array (rows, cols, bands)
    bands : numpy.ndarray
        Wavelengths in nm

    Returns:
    --------
    ndvi : numpy.ndarray
        NDVI image
    """
    # Find band indices
    nir_idx = np.argmin(np.abs(bands - 860))   # ~860 nm
    red_idx = np.argmin(np.abs(bands - 660))   # ~660 nm

    nir = R[:, :, nir_idx].astype(float)
    red = R[:, :, red_idx].astype(float)

    # Compute NDVI with division safety
    with np.errstate(invalid='ignore', divide='ignore'):
        ndvi = (nir - red) / (nir + red)

    # Mask invalid values
    ndvi = np.where(np.isfinite(ndvi), ndvi, 0)
    ndvi = np.clip(ndvi, -1, 1)

    return ndvi


def plot_sample_spectra(R, bands, output_path, n_samples=5):
    """
    Plot sample spectra from the reflectance image.

    Parameters:
    -----------
    R : numpy.ndarray
        Reflectance array (rows, cols, bands)
    bands : numpy.ndarray
        Wavelengths in nm
    output_path : str
        Path to save the plot
    n_samples : int
        Number of random spectra to plot
    """
    import matplotlib.pyplot as plt

    rows, cols, nbands = R.shape

    # Get random valid pixel locations
    np.random.seed(42)
    valid_mask = np.sum(R, axis=2) > 0
    valid_coords = np.argwhere(valid_mask)

    if len(valid_coords) < n_samples:
        print('    Warning: Not enough valid pixels for spectral plot')
        return

    sample_indices = np.random.choice(len(valid_coords), n_samples, replace=False)

    plt.figure(figsize=(12, 6))

    for idx in sample_indices:
        row, col = valid_coords[idx]
        spectrum = np.squeeze(R[row, col, :])  # Ensure 1D array
        plt.plot(bands, spectrum, label=f'Pixel ({row}, {col})', alpha=0.7)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Sample Spectra from Reflectance Image')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([400, 2500])
    plt.ylim([0, None])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'    Sample spectra plot saved to: {output_path}')


def post_processing(R, bands, pathOut, fname):
    """
    Generate post-processing outputs: quicklooks, NDVI, sample spectra.

    Parameters:
    -----------
    R : numpy.ndarray
        Reflectance array (rows, cols, bands)
    bands : numpy.ndarray
        Wavelengths in nm
    pathOut : str
        Output directory
    fname : str
        Base filename for outputs
    """
    import matplotlib.pyplot as plt

    print('\n' + '=' * 60)
    print('STEP 3: POST-PROCESSING & VISUALIZATION')
    print('=' * 60)

    # Create quicklooks directory
    quicklooks_dir = pathOut + 'quicklooks/'
    Path(quicklooks_dir).mkdir(parents=True, exist_ok=True)

    print('\n[1/5] Creating RGB quicklook...')
    create_rgb_quicklook(R, bands, quicklooks_dir + fname + '_RGB.png')

    print('\n[2/5] Creating false color quicklook...')
    create_false_color_quicklook(R, bands, quicklooks_dir + fname + '_FalseColor.png')

    print('\n[3/5] Computing NDVI...')
    ndvi = compute_ndvi(R, bands)

    # Save NDVI as numpy array
    np.save(pathOut + fname + '_NDVI.npy', ndvi)
    print(f'    NDVI array saved to: {pathOut + fname}_NDVI.npy')

    # Plot NDVI
    plt.figure(figsize=(10, 10))
    im = plt.imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    plt.colorbar(im, label='NDVI', shrink=0.8)
    plt.title('Normalized Difference Vegetation Index (NDVI)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(quicklooks_dir + fname + '_NDVI.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    NDVI plot saved to: {quicklooks_dir + fname}_NDVI.png')

    print('\n[4/5] Plotting sample spectra...')
    plot_sample_spectra(R, bands, quicklooks_dir + fname + '_spectra.png')

    print('\n[5/5] Computing statistics...')
    # Compute and print statistics
    valid_mask = np.sum(R, axis=2) > 0
    n_valid = np.sum(valid_mask)
    n_total = R.shape[0] * R.shape[1]

    print(f'    Image dimensions: {R.shape[0]} x {R.shape[1]} pixels, {R.shape[2]} bands')
    print(f'    Valid pixels: {n_valid} / {n_total} ({100*n_valid/n_total:.1f}%)')
    print(f'    Wavelength range: {bands.min():.1f} - {bands.max():.1f} nm')

    # NDVI statistics
    valid_ndvi = ndvi[valid_mask]
    print(f'    NDVI range: {valid_ndvi.min():.3f} to {valid_ndvi.max():.3f}')
    print(f'    NDVI mean: {valid_ndvi.mean():.3f}')

    # Save statistics to file
    with open(pathOut + fname + '_statistics.txt', 'w') as f:
        f.write(f'Hyperion Image Processing Statistics\n')
        f.write(f'=====================================\n\n')
        f.write(f'Image ID: {fname}\n')
        f.write(f'Image dimensions: {R.shape[0]} x {R.shape[1]} pixels\n')
        f.write(f'Number of bands: {R.shape[2]}\n')
        f.write(f'Wavelength range: {bands.min():.1f} - {bands.max():.1f} nm\n\n')
        f.write(f'Valid pixels: {n_valid} / {n_total} ({100*n_valid/n_total:.1f}%)\n\n')
        f.write(f'NDVI Statistics:\n')
        f.write(f'  Min: {valid_ndvi.min():.3f}\n')
        f.write(f'  Max: {valid_ndvi.max():.3f}\n')
        f.write(f'  Mean: {valid_ndvi.mean():.3f}\n')
        f.write(f'  Std: {valid_ndvi.std():.3f}\n')

    print(f'    Statistics saved to: {pathOut + fname}_statistics.txt')

    print('\n' + '=' * 60)
    print('POST-PROCESSING COMPLETE!')
    print('=' * 60)


if __name__ == '__main__':

    # ============================================================
    # CONFIGURATION
    # ============================================================

    # ============================================================
    # GOOGLE EARTH ENGINE PROJECT ID
    # ============================================================
    # IMPORTANT: You MUST set your GEE project ID below!
    #
    # To get a project ID:
    # 1. Go to https://code.earthengine.google.com/
    # 2. If prompted, register for Earth Engine
    # 3. Your project ID will be shown in the URL or in your account settings
    #    It typically looks like: 'ee-yourusername' or 'your-project-name'
    # 4. Replace 'YOUR_PROJECT_ID' below with your actual project ID
    # ============================================================

    GEE_PROJECT_ID = 'remote-sensing-478802'  # Your GEE Project ID

    print('Initializing Google Earth Engine...')

    if GEE_PROJECT_ID == 'YOUR_PROJECT_ID':
        print('\n' + '=' * 60)
        print('ERROR: GEE Project ID not configured!')
        print('=' * 60)
        print('\nPlease edit process_hyperion.py and set your GEE project ID.')
        print('Look for the line: GEE_PROJECT_ID = "YOUR_PROJECT_ID"')
        print('\nTo get a project ID:')
        print('1. Go to https://code.earthengine.google.com/')
        print('2. Register/sign in with your Google account')
        print('3. Find your project ID (e.g., "ee-yourusername")')
        print('=' * 60)
        raise ValueError('GEE Project ID not configured. Please edit the script.')

    ee.Initialize(project=GEE_PROJECT_ID)
    print(f'GEE initialized with project: {GEE_PROJECT_ID}')

    # SMARTS configuration
    smartsPath = 'C:/Program Files/SMARTS_295_PC/'
    os.environ['SMARTSPATH'] = smartsPath

    # Add SMARTS to PATH so the executable can be found
    if smartsPath not in os.environ['PATH']:
        os.environ['PATH'] = smartsPath + os.pathsep + os.environ['PATH']

    surehyp.atmoCorrection.smartsVersion = 'smarts295'
    surehyp.atmoCorrection.smartsExecutable = 'smarts295bat.exe'
    print(f"SMARTS path: {smartsPath}")

    # ============================================================
    # PATHS CONFIGURATION
    # ============================================================

    basePath = 'C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/'

    # Path to USGS Hyperion metadata (optional, can be empty file)
    pathToL1Rmetadata = basePath + 'METADATA/METADATA.csv'

    # Path to L1R images (uncompressed folders)
    pathToL1Rimages = basePath + 'L1R/'

    # Path to L1T images (TIF files)
    pathToL1Timages = basePath + 'L1T/'

    # Path for filtered/processed L1T images
    pathToL1TimagesFiltered = basePath + 'L1T/filteredImages/'

    # Output folder
    pathOut = basePath + 'OUT/'

    # ============================================================
    # IMAGE CONFIGURATION
    # ============================================================

    # Hyperion image ID (folder name)
    fname = 'EO1H0370412009266110PF'

    # Output names
    nameOut_radiance = fname + '_preprocessed'
    nameOut_reflectance = fname + '_reflectance'

    # ============================================================
    # PROCESSING OPTIONS
    # ============================================================

    # Destriping method: 'Pal' (recommended) or 'Datt'
    destripingMethod = 'Pal'

    # Local destriping refinement (slower but better results)
    localDestriping = False

    # Topographic correction
    # Set to True to enable (requires DEM download from GEE)
    use_topo = True

    # DEM source from Google Earth Engine
    # Options: 'USGS/SRTMGL1_003' (global), 'NRCan/CDEM' (Canada), etc.
    demID = 'USGS/SRTMGL1_003'
    elevationName = 'elevation'

    # Post-processing: generate quicklooks and statistics
    run_postprocessing = True

    # ============================================================
    # SNAP WAVELENGTH COMPATIBILITY OPTIONS
    # ============================================================
    # Configure how spectral wavelengths are written to ENVI HDR files
    # for compatibility with SNAP software

    # Option 1: Load wavelengths from external file (e.g., custom spectral calibration)
    # Set to None to use wavelengths computed during processing
    snap_wavelength_file = "C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/OUT/EO1H2020342016359110KF_reflectance_spectral_info.txt"
    # Example: snap_wavelength_file = basePath + 'OUT/EO1H2020342016359110KF_reflectance_spectral_info.txt'

    # Option 2: Keep wavelength field in HDR file
    # - True: Better for SNAP visualization (wavelength labels show in plots)
    # - False: Safer for SNAP band math expressions (avoids "Undefined function" errors)
    snap_keep_wavelength = True  # Set to True to include wavelengths in HDR for SNAP

    # ============================================================
    # METADATA XML CONFIGURATION (OPTIONAL)
    # ============================================================
    # Automatically find XML metadata file from fname
    # The script searches for any file matching: eo1_hyp_pub_{fname}_*.xml
    # This handles all USGS naming variations (SG1_01, SGS_01, etc.)
    #
    # To download XML metadata:
    # 1. Go to https://earthexplorer.usgs.gov/
    # 2. Find your image (e.g., EO1H0370412009263110KF)
    # 3. Click 'Metadata' → 'Export Metadata'
    # 4. Save in METADATA folder (naming is automatic from USGS)
    #
    # Set process_xml_metadata to False to skip metadata conversion
    process_xml_metadata = True  # Set to False to skip XML metadata conversion

    # Find XML file automatically using glob pattern
    if process_xml_metadata:
        xml_pattern = basePath + 'METADATA/eo1_hyp_pub_' + fname + '_*.xml'
        matching_files = glob.glob(xml_pattern)

        if matching_files:
            xml_metadata_path = matching_files[0]  # Use first match
            if len(matching_files) > 1:
                print(f'    Note: Found {len(matching_files)} XML files for {fname}, using: {os.path.basename(xml_metadata_path)}')
        else:
            # No file found - set to expected pattern for error message
            xml_metadata_path = basePath + 'METADATA/eo1_hyp_pub_' + fname + '_*.xml'
    else:
        xml_metadata_path = None

    # ============================================================
    # RUN PROCESSING
    # ============================================================

    print('\n' + '=' * 60)
    print('HYPERION IMAGE PROCESSING')
    print(f'Image ID: {fname}')
    print('=' * 60)

    # STEP 0 (Optional): Convert XML metadata to CSV
    if xml_metadata_path is not None:
        print('\n' + '=' * 60)
        print('STEP 0: METADATA CONVERSION (XML TO CSV)')
        print('=' * 60)

        if os.path.exists(xml_metadata_path):
            print(f'\nInput XML: {xml_metadata_path}')
            print(f'Output CSV: {pathToL1Rmetadata}')

            try:
                create_metadata_csv(xml_metadata_path, pathToL1Rmetadata, append=True)
                print('\n' + '=' * 60)
                print('METADATA CONVERSION COMPLETE!')
                print('=' * 60)
            except Exception as e:
                print(f'\nWarning: Metadata conversion failed: {e}')
                print('Continuing with image processing...')
        else:
            print(f'\nWarning: XML metadata file not found: {xml_metadata_path}')
            print('Skipping metadata conversion...')
            print('\nTo download XML metadata:')
            print('1. Go to https://earthexplorer.usgs.gov/')
            print(f'2. Find your image ({fname})')
            print('3. Click "Metadata" → "Export Metadata"')
            print('4. Save and update xml_metadata_path in this script')

    # STEP 1: Preprocess radiance
    # Check if preprocessed file already exists
    pathToRadianceImage = pathOut + nameOut_radiance
    if os.path.exists(pathToRadianceImage + '.bip') or os.path.exists(pathToRadianceImage + '.img'):
        print('\nPreprocessed radiance file already exists, skipping Step 1...')
        print(f'Using: {pathToRadianceImage}')
    else:
        pathToRadianceImage = preprocess_radiance(
            fname,
            pathToL1Rmetadata,
            pathToL1Rimages,
            pathToL1Timages,
            pathToL1TimagesFiltered,
            pathOut,
            nameOut_radiance,
            destripingMethod=destripingMethod,
            localDestriping=localDestriping,
            checkSmile=False
        )

    # STEP 2: Atmospheric correction with optional topographic correction
    pathToReflectanceImage = pathOut + nameOut_reflectance

    # Check if reflectance file already exists
    if os.path.exists(pathToReflectanceImage + '.bip') or os.path.exists(pathToReflectanceImage + '.img'):
        print('\nReflectance file already exists, skipping Step 2...')
        print(f'Using: {pathToReflectanceImage}')

        # Load existing reflectance for post-processing
        import spectral.io.envi as envi
        if os.path.exists(pathToReflectanceImage + '.img'):
            hdr_path = pathToReflectanceImage + '.hdr'
            img_path = pathToReflectanceImage + '.img'
        else:
            hdr_path = pathToReflectanceImage + '.hdr'
            img_path = pathToReflectanceImage + '.bip'

        img = envi.open(hdr_path, img_path)
        R = img.load()

        # Try to get wavelengths from HDR, or fall back to spectral_info.txt
        if 'wavelength' in img.metadata:
            bands = np.array([float(w) for w in img.metadata['wavelength']])
        else:
            # Load wavelengths from the spectral info file (created by fix_envi_hdr_for_snap)
            spectral_info_path = pathToReflectanceImage + '_spectral_info.txt'
            if os.path.exists(spectral_info_path):
                bands = []
                with open(spectral_info_path, 'r') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            bands.append(float(parts[1].strip()))
                bands = np.array(bands)
                print(f'    Loaded wavelengths from: {spectral_info_path}')
            else:
                raise ValueError(f"No wavelength data found in HDR or spectral info file: {spectral_info_path}")
    else:
        pathToReflectanceImage, R, bands = atmospheric_correction(
            pathToRadianceImage,
            pathToReflectanceImage,
            topo=use_topo,
            demID=demID,
            elevationName=elevationName,
            smartsAlbedoFilePath=os.environ['SMARTSPATH'] + 'Albedo/Albedo.txt',
            snap_wavelength_file=snap_wavelength_file,
            snap_keep_wavelength=snap_keep_wavelength
        )

    # STEP 3: Post-processing (visualization and statistics)
    if run_postprocessing:
        post_processing(R, bands, pathOut, fname)

    print('\n' + '=' * 60)
    print('ALL PROCESSING COMPLETE!')
    print('=' * 60)
    print(f'\nOutputs:')
    print(f'  - Preprocessed radiance: {pathOut + nameOut_radiance}.img')
    print(f'  - Surface reflectance:   {pathOut + nameOut_reflectance}.img')
    print(f'  - Clearview mask:        {pathOut + nameOut_reflectance}_clearview_mask.npy')
    print(f'  - Cirrus mask:           {pathOut + nameOut_reflectance}_cirrus_mask.npy')
    print(f'  - NDVI:                  {pathOut + fname}_NDVI.npy')
    print(f'  - Statistics:            {pathOut + fname}_statistics.txt')
    print(f'  - Quicklooks:            {pathOut}quicklooks/')
