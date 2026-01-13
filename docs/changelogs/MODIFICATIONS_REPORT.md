# SUREHYP Modifications Report

## Overview

This document details all modifications made to the SUREHYP package (Surface Reflectance from Hyperion) to make the processing script runnable with current software versions. The original package by Thomas Miraglio required several fixes to address compatibility issues with SNAP, Google Earth Engine API changes, and rasterio.

**Date:** November 21-22, 2025
**Modified by:** Lorien Crespo (with Roger in Kreuzgasse)
**Original package:** SUREHYP v1.0.1.2 by Thomas Miraglio

---

## 1. SNAP Compatibility Fix (ENVI HDR File Issue)

### Problem

SNAP automatically appends wavelength values in parentheses to band names when loading ENVI files. For example, a band named `band_1` becomes `band_1 (436.99)`. This causes **"Undefined function" errors** in SNAP's expression parser when performing band math operations, because the parentheses and decimal points are interpreted as function calls.

### Solution

Created a new function `fix_envi_hdr_for_snap()` that modifies ENVI header files after saving:

```python
def fix_envi_hdr_for_snap(hdr_path):
    """
    Fix ENVI header files to be compatible with SNAP.

    - Removes 'wavelength' and 'fwhm' fields from HDR
    - Generates simple band names (band_1, band_2, etc.)
    - Saves spectral info to separate _spectral_info.txt file
    - Creates backup of original HDR file
    """
```

**Key changes:**
- Removes `wavelength = { ... }` field from HDR (this is what triggers SNAP's band renaming)
- Removes `fwhm = { ... }` field (not needed for SNAP viewing)
- Replaces complex band names with simple `band_1, band_2, ...` format
- Preserves spectral information in a separate `_spectral_info.txt` file
- Creates `.hdr.backup` before modifying

**Location in code:** Lines 26-122 in `process_hyperion.py`

---

## 2. Google Earth Engine API Compatibility

### Problem 2a: SRTM Treated as ImageCollection Instead of Image

The original SUREHYP code treated SRTM DEM (`USGS/SRTMGL1_003`) as an `ImageCollection`, but Google Earth Engine now provides it as a single `Image`. This caused errors when trying to use `.mosaic()` or iterate over what was expected to be a collection.

### Solution

Created `getGEEdem_fixed()` function that correctly handles SRTM as an Image:

```python
def getGEEdem_fixed(UL_lat, UL_lon, UR_lat, UR_lon, LL_lat, LL_lon, LR_lat, LR_lon,
                    demID='USGS/SRTMGL1_003', elevationName='elevation', numPixels=1000):
    """
    Fixed version of getGEEdem that handles SRTM as Image instead of ImageCollection.
    """
    # SRTM is an Image, not an ImageCollection
    DEM = ee.Image(demID)  # Changed from ee.ImageCollection(demID)

    # Use sample() or reduceRegion() to get elevation
    result = DEM.sample(region=coord, numPixels=numPixels, scale=1000).getInfo()
```

**Location in code:** Lines 125-163 in `process_hyperion.py`

---

### Problem 2b: Deprecated geetools Download Function

The original code used `geetools.batch.image.toLocal()` for downloading DEM data, which is deprecated in geetools >= 1.0.

### Solution

Created `downloadDEMfromGEE()` function using the new GEE API:

```python
def downloadDEMfromGEE(UL_lon, UL_lat, UR_lon, UR_lat, LR_lon, LR_lat, LL_lon, LL_lat,
                       demID='USGS/SRTMGL1_003', elevationName='elevation', output_path=None):
    """
    Download DEM from Google Earth Engine using the new API.
    Uses ee.Image.getDownloadURL() instead of deprecated geetools.batch.image.toLocal()
    """
    # Primary method: getDownloadURL
    url = elev.getDownloadURL({
        'name': 'elev',
        'scale': 30,
        'region': region,
        'format': 'GEO_TIFF'
    })
    response = requests.get(url)

    # Fallback method: sample points and interpolate
    # (implemented if primary method fails)
```

**Key changes:**
- Uses `ee.Image.getDownloadURL()` as primary download method
- Implements fallback using `ee.Image.sample()` with scipy interpolation
- Handles both Image and ImageCollection DEM sources with try-except
- Creates proper GeoTIFF output using rasterio

**Location in code:** Lines 166-298 in `process_hyperion.py`

---

## 3. Rasterio Tiling Issue

### Problem

The original `processImage()` function in `preprocess.py` caused a **"TileWidth must be multiple of 16"** error when writing multi-band GeoTIFF files. This occurred because rasterio's tiled output mode has strict requirements on image dimensions.

### Solution

Created `processImage_fixed()` function that disables tiling:

```python
def processImage_fixed(fname, pathToImages, pathToImagesFiltered):
    """
    Fixed version of processImage that doesn't use tiled output
    (fixes the "TileWidth must be multiple of 16" error)
    """
    # ... read bands ...

    # Write all bands to a new file (without tiling to avoid dimension issues)
    profile.update(count=len(arrays), nodata=0, tiled=False)  # KEY CHANGE: tiled=False

    with rasterio.open(path, 'w', compress='lzw', **profile) as dst:
        # write bands...
```

**Location in code:** Lines 301-335 in `process_hyperion.py`

---

## 4. Atmospheric Parameter Fallback Values

### Problem

The `getAtmosphericParameters()` function from the original SUREHYP can fail when Google Earth Engine data extraction doesn't return valid values for water vapor and ozone. This causes the entire processing chain to fail.

### Solution

Added try-except block with sensible default values for mid-latitude atmospheres:

```python
try:
    wv, o3, flag_no_o3 = surehyp.atmoCorrection.getAtmosphericParameters(
        bands, L, datestamp1, year, doy, longit, latit, altit,
        satelliteZenith, zenith, azimuth
    )
except (ValueError, Exception) as e:
    print(f'WARNING: Could not compute atmospheric parameters: {e}')
    print('Using default values for mid-latitude atmosphere')

    # Default values for mid-latitude winter
    wv = 1.5   # precipitable water vapor in cm
    o3 = 0.35  # ozone column in atm-cm
    IO3 = 0    # Use the specified ozone value (not SMARTS default)
```

**Location in code:** Lines 519-538 in `process_hyperion.py`

---

## 5. Topographic Correction Error Handling

### Problem

The topographic correction workflow could fail at various stages (DEM download, reprojection, resampling) and would crash the entire script.

### Solution

Wrapped the topographic correction section in try-except to allow processing to continue without topographic correction if it fails:

```python
if topo:
    try:
        # Download DEM
        path_to_dem = downloadDEMfromGEE(...)

        # Reproject and resample DEM
        path_to_reprojected_dem = surehyp.atmoCorrection.reprojectDEM(...)
        path_elev = surehyp.atmoCorrection.matchResolution(...)

        # Extract DEM data
        elev, slope, wazim = surehyp.atmoCorrection.extractDEMdata(...)

    except Exception as e:
        print(f'WARNING: Topographic correction failed: {e}')
        print('Continuing without topographic correction...')
        topo = False  # Disable topo correction and continue
```

**Location in code:** Lines 458-495 in `process_hyperion.py`

---

## 6. Google Earth Engine Project Initialization

### Problem

Newer versions of the Earth Engine Python API require explicit project initialization with a project ID.

### Solution

Added proper GEE initialization with project ID and clear error message if not configured:

```python
GEE_PROJECT_ID = 'remote-sensing-478802'  # User must set their project ID

if GEE_PROJECT_ID == 'YOUR_PROJECT_ID':
    print('ERROR: GEE Project ID not configured!')
    print('Please edit process_hyperion.py and set your GEE project ID.')
    raise ValueError('GEE Project ID not configured.')

ee.Initialize(project=GEE_PROJECT_ID)
```

**Location in code:** Lines 906-937 in `process_hyperion.py`

---

## 7. SMARTS Path Configuration

### Problem

The SMARTS radiative transfer model executable needs to be in the system PATH and its directory needs to be set as an environment variable.

### Solution

Added proper SMARTS configuration at script startup:

```python
# SMARTS configuration
smartsPath = 'C:/Program Files/SMARTS_295_PC/'
os.environ['SMARTSPATH'] = smartsPath

# Add SMARTS to PATH so the executable can be found
if smartsPath not in os.environ['PATH']:
    os.environ['PATH'] = smartsPath + os.pathsep + os.environ['PATH']

surehyp.atmoCorrection.smartsVersion = 'smarts295'
surehyp.atmoCorrection.smartsExecutable = 'smarts295bat.exe'
```

**Location in code:** Lines 940-949 in `process_hyperion.py`

---

## 8. Result Caching (Skip Existing Outputs)

### Problem

The original workflow would reprocess everything from scratch each time, even if intermediate results already existed. This was time-consuming and wasteful.

### Solution

Added checks to skip processing steps if output files already exist:

```python
# Check for existing preprocessed radiance
pathToRadianceImage = pathOut + nameOut_radiance
if os.path.exists(pathToRadianceImage + '.bip') or os.path.exists(pathToRadianceImage + '.img'):
    print('Preprocessed radiance file already exists, skipping Step 1...')
else:
    pathToRadianceImage = preprocess_radiance(...)

# Check for existing reflectance
if os.path.exists(pathToReflectanceImage + '.bip') or os.path.exists(pathToReflectanceImage + '.img'):
    print('Reflectance file already exists, skipping Step 2...')
    # Load existing file instead
    img = envi.open(hdr_path, img_path)
    R = img.load()
else:
    pathToReflectanceImage, R, bands = atmospheric_correction(...)
```

**Location in code:** Lines 1016-1081 in `process_hyperion.py`

---

## Summary of All Modifications

| # | Issue | Root Cause | Solution |
|---|-------|-----------|----------|
| 1 | SNAP HDR compatibility | SNAP appends wavelength to band names | Remove wavelength/fwhm from HDR, use simple band names |
| 2a | SRTM DEM access | GEE changed SRTM from ImageCollection to Image | Use `ee.Image()` instead of `ee.ImageCollection()` |
| 2b | DEM download | `geetools.batch.image.toLocal()` deprecated | Use `ee.Image.getDownloadURL()` |
| 3 | GeoTIFF writing | Tiled output requires specific dimensions | Disable tiling with `tiled=False` |
| 4 | Atmospheric params | GEE extraction can fail | Add fallback default values |
| 5 | Topo correction | Various failure points | Wrap in try-except, continue without topo if fails |
| 6 | GEE initialization | New API requires project ID | Add `ee.Initialize(project=ID)` |
| 7 | SMARTS path | Executable not found | Set PATH and SMARTSPATH environment variables |
| 8 | Reprocessing | No caching of results | Skip steps if output files exist |

---

## New Functions Added

| Function | Purpose | Lines |
|----------|---------|-------|
| `fix_envi_hdr_for_snap()` | Fix ENVI headers for SNAP compatibility | 26-122 |
| `getGEEdem_fixed()` | Get mean elevation handling SRTM as Image | 125-163 |
| `downloadDEMfromGEE()` | Download DEM using new GEE API | 166-298 |
| `processImage_fixed()` | Process L1T without tiling issues | 301-335 |
| `preprocess_radiance()` | Enhanced preprocessing with logging | 338-412 |
| `atmospheric_correction()` | Enhanced atmo correction with error handling | 415-607 |
| `create_rgb_quicklook()` | Generate RGB visualization | 610-666 |
| `create_false_color_quicklook()` | Generate NIR-R-G false color | 669-723 |
| `compute_ndvi()` | Calculate NDVI | 726-757 |
| `plot_sample_spectra()` | Plot sample pixel spectra | 760-808 |
| `post_processing()` | Generate all visualization outputs | 811-897 |

---

## Prerequisites for Running

1. **Google Earth Engine account** with a project ID
2. **SMARTS v2.9.5** installed (radiative transfer model)
3. **Python environment** with dependencies:
   - numpy, scipy, pandas, matplotlib
   - rasterio, spectral, earthengine-api
   - requests, tqdm, scikit-image

4. **Input data:**
   - L1R Hyperion image (HDF format in folder)
   - L1T Hyperion image (ZIP file with TIF bands)

---

## Usage

1. Edit `process_hyperion.py` and set your paths and GEE project ID
2. Run: `python process_hyperion.py`
3. Outputs will be saved to the configured `pathOut` directory

---

*Last updated: November 2025*
