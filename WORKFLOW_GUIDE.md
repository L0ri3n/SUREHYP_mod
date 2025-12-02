# Hyperion Image Processing Workflow Guide

## Complete Step-by-Step Guide for Processing EO1H2020342013284110KF

---

## Table of Contents
1. [Overview](#overview)
2. [Required Input Data](#required-input-data)
3. [Directory Structure](#directory-structure)
4. [Step-by-Step Instructions](#step-by-step-instructions)
5. [Metadata Conversion Details](#metadata-conversion-details)
6. [Processing Pipeline](#processing-pipeline)
7. [Output Files](#output-files)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This workflow processes Hyperion hyperspectral satellite imagery from Level 1R (radiance) to surface reflectance with atmospheric and topographic corrections.

**Processing Chain:**
```
L1R (Radiance) + L1T (Georeferenced) + Metadata (XML)
    ↓
Preprocessing (Desmiling, Destriping, Georeferencing)
    ↓
Atmospheric Correction (SMARTS radiative transfer)
    ↓
Topographic Correction (optional, using DEM from GEE)
    ↓
Surface Reflectance + Quicklooks + Statistics
```

---

## Required Input Data

For processing image **EO1H2020342013284110KF**, you need 3 files from [USGS EarthExplorer](https://earthexplorer.usgs.gov/):

| File | Description | Format | Size (approx) |
|------|-------------|--------|---------------|
| `EO1H2020342013284110KF_1R.ZIP` | Level 1R radiance data (uncalibrated) | HDF4 + ENVI | ~400 MB |
| `EO1H2020342013284110KF_1T.ZIP` | Level 1T georeferenced TIF bands | GeoTIFF | ~200 MB |
| `eo1_hyp_pub_EO1H2020342013284110KF_SG1_01.xml` | Metadata (sun angles, coordinates, etc.) | XML | ~5 KB |

### How to Download from USGS EarthExplorer

1. Go to https://earthexplorer.usgs.gov/
2. Search for scene: `EO1H2020342013284110KF`
3. Download:
   - **L1R product**: Click "Download Options" → Select "Level 1R" → Download ZIP
   - **L1T product**: Click "Download Options" → Select "Level 1T" → Download ZIP
   - **Metadata**: Click "Metadata" button → "Export Metadata" → Save XML file

---

## Directory Structure

### Required Folder Layout

```
Remote_Sensing/
│
├── L1R/                                  ← L1R radiance data (EXTRACTED)
│   └── EO1H2020342013284110KF/          ← Extract _1R.ZIP here
│       ├── EO1H2020342013284110KF.L1R   ← HDF4 radiance file
│       ├── EO1H2020342013284110KF.hdr   ← ENVI header
│       ├── EO1H2020342013284110KF.MET   ← Corner coordinates
│       ├── EO1H2020342013284110KF.AUX   ← Auxiliary data
│       └── README.txt
│
├── L1T/                                  ← L1T georeferenced data (ZIPPED)
│   └── EO1H2020342013284110KF_1T.ZIP    ← Keep as ZIP! Do NOT extract
│
├── METADATA/                             ← Metadata CSV
│   └── METADATA.csv                      ← Generated from XML (multiple images)
│
├── Project/                              ← Store XML files here
│   └── eo1_hyp_pub_EO1H2020342013284110KF_SG1_01.xml
│
├── SUREHYP/                              ← Processing scripts
│   ├── process_hyperion.py              ← Main processing script
│   ├── create_metadata_EO1H2020342013284110KF.py  ← XML→CSV converter
│   └── ...
│
└── OUT/                                  ← Output folder (auto-created)
    ├── EO1H2020342013284110KF_preprocessed.img
    ├── EO1H2020342013284110KF_reflectance.img
    └── quicklooks/
```

### ⚠️ Important Notes

- **L1R data must be EXTRACTED** (the code reads individual files from the folder)
- **L1T data must stay ZIPPED** (the code uses `ZipFile` to read it directly)
- **METADATA.csv** can contain multiple images (one row per image)

---

## Step-by-Step Instructions

### Step 1: Place Input Files

✅ **Already done** - You have:
- ✓ L1R extracted: `L1R/EO1H2020342013284110KF/`
- ✓ L1T zipped: `L1T/EO1H2020342013284110KF_1T.ZIP`
- ✓ XML metadata: `Project/eo1_hyp_pub_EO1H2020342013284110KF_SG1_01.xml`

### Step 2: Convert XML Metadata to CSV

Run the metadata conversion script:

```bash
cd C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP
python create_metadata_EO1H2020342013284110KF.py
```

**What this does:**
- Parses the XML file
- Extracts sun angles, look angles, coordinates, acquisition time
- Appends a new row to `METADATA/METADATA.csv` (or creates it if it doesn't exist)
- Checks for duplicates (won't add the same image twice)

**Expected Output:**
```
✓ Appended EO1H2020342013284110KF_SG1_01 to METADATA/METADATA.csv

Metadata Summary for EO1H2020342013284110KF:
  Acquisition Date: 2013/10/11
  Sun Elevation: 36.752901°
  Sun Azimuth: 140.392107°
  Look Angle: -25.682°
  Center Coordinates: (37.555186, -6.580288)
  Cloud Cover: 0%
```

### Step 3: Update Processing Script

Edit `process_hyperion.py` at **line 1051**:

```python
# OLD (for 2016 image):
fname = 'EO1H2020342016359110KF'

# NEW (for 2013 image):
fname = 'EO1H2020342013284110KF'
```

### Step 4: Run Processing

```bash
cd C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP
python process_hyperion.py
```

**Processing time:** ~10-30 minutes depending on:
- Topographic correction (enabled/disabled)
- DEM download speed from Google Earth Engine
- Computer performance

---

## Metadata Conversion Details

### Why is Metadata Needed?

The `METADATA.csv` file provides critical information that is **NOT** in the L1R/L1T files:

| Field | Used For | Required? |
|-------|----------|-----------|
| Sun Elevation | Calculating sun zenith angle for atmospheric correction | ✅ Yes |
| Sun Azimuth | Solar geometry for topographic correction | ✅ Yes |
| Look Angle | Satellite zenith angle | ✅ Yes |
| Satellite Inclination | Satellite azimuth angle | ✅ Yes |
| Center Lat/Lon | Atmospheric parameters (water vapor, ozone) | ✅ Yes |
| Scene Start/Stop Time | Precise acquisition time for solar position | ✅ Yes |
| Acquisition Date | Day-of-year for solar distance correction | ✅ Yes |

### Metadata CSV Format

The CSV must have these exact column names (case-sensitive):

```csv
Entity ID,Sun Elevation,Sun Azimuth,Look Angle,Satellite Inclination,Center Longtude dec,Center Latitude dec,Acquisition Date,Scene Start Time,Scene Stop Time
EO1H2020342013284110KF_SG1_01,36.752901,140.392107,-25.682,98.0,-6.580288,37.555186,2013/10/11,2013:284:10:09:05,2013:284:10:09:19
```

⚠️ **Note:** The typo `Longtude` (instead of `Longitude`) is intentional - SUREHYP expects this spelling!

### Manual Metadata Entry (Alternative)

If you don't have the XML file, you can manually add a row to `METADATA.csv`:

1. Get values from the `.MET` file in the L1R folder
2. Calculate sun zenith = 90 - Sun Elevation
3. Add to CSV with proper formatting

---

## Processing Pipeline

### Stage 1: Preprocessing Radiance (15-20 minutes)

**Input:** L1R (radiance) + L1T (georeferenced)

**Steps:**
1. **Concatenate L1T image** - Merge 242 individual TIF bands
2. **Read L1R image** - Load HDF4 radiance data
3. **Get metadata** - Load wavelengths, FWHM, acquisition parameters
4. **Separate VNIR/SWIR** - Split into two detector arrays
5. **DN to radiance** - Apply calibration coefficients (40 for VNIR, 80 for SWIR)
6. **Align VNIR/SWIR** - Register spatial offset between detectors
7. **Desmiling** - Correct spectral smile (wavelength shift across track)
8. **Destriping** - Remove detector striping artifacts
9. **Smooth cirrus bands** - Prepare for thin cirrus removal
10. **Georeference** - Apply L1T geolocation to corrected L1R data

**Output:** `EO1H2020342013284110KF_preprocessed.img` (georeferenced radiance)

### Stage 2: Atmospheric Correction (10-15 minutes)

**Input:** Preprocessed radiance

**Steps:**
1. **Open radiance image** - Load preprocessed data
2. **Download DEM** - Get SRTM elevation data from Google Earth Engine
3. **Reproject DEM** - Match image coordinate system
4. **Resample DEM** - Match image resolution (30m)
5. **Extract DEM data** - Get elevation, slope, aspect for each pixel
6. **Cloud/shadow detection** - Identify clear-view pixels (currently skipped)
7. **Dark object subtraction** - Estimate haze spectrum
8. **Cirrus removal** - Remove thin cirrus using 1.38μm band
9. **Get average elevation** - Scene mean altitude for atmospheric model
10. **Get atmospheric parameters** - Estimate water vapor and ozone
11. **Run SMARTS** - Radiative transfer modeling (sun→ground→satellite)
12. **Compute reflectance** - Convert radiance to reflectance using SMARTS outputs

**With topographic correction:**
- **Write albedo file** - Scene-average reflectance for terrain modeling
- **Compute LUT** - Look-up table for terrain illumination effects
- **Apply Modified-Minnaert correction** - Normalize for topographic effects

**Output:** `EO1H2020342013284110KF_reflectance.img` (surface reflectance)

### Stage 3: Post-processing (2-5 minutes)

**Steps:**
1. **RGB quicklook** - True color composite (R:660nm, G:550nm, B:480nm)
2. **False color quicklook** - NIR-R-G composite for vegetation
3. **Compute NDVI** - Normalized Difference Vegetation Index
4. **Plot sample spectra** - Show spectral signatures from random pixels
5. **Compute statistics** - Image stats, NDVI range, valid pixels

**Outputs:** PNG images, NPY arrays, TXT statistics

---

## Output Files

### Main Products

| File | Description | Format | Usage |
|------|-------------|--------|-------|
| `EO1H2020342013284110KF_preprocessed.img` | Preprocessed radiance | ENVI | Intermediate (can delete after processing) |
| `EO1H2020342013284110KF_reflectance.img` | Surface reflectance | ENVI | **Main product** - use for analysis |
| `EO1H2020342013284110KF_reflectance.hdr` | ENVI header | Text | Contains wavelengths, metadata |

### Masks & Auxiliary

| File | Description |
|------|-------------|
| `EO1H2020342013284110KF_clearview_mask.npy` | Clear-view pixel mask (all 1s if cloud detection skipped) |
| `EO1H2020342013284110KF_cirrus_mask.npy` | Cirrus cloud mask |
| `EO1H2020342013284110KF_NDVI.npy` | NDVI array (NumPy format) |
| `EO1H2020342013284110KF_statistics.txt` | Summary statistics |

### Quicklooks (in `OUT/quicklooks/`)

| File | Description |
|------|-------------|
| `EO1H2020342013284110KF_RGB.png` | True color RGB composite |
| `EO1H2020342013284110KF_FalseColor.png` | NIR-R-G false color |
| `EO1H2020342013284110KF_NDVI.png` | NDVI visualization |
| `EO1H2020342013284110KF_spectra.png` | Sample spectral signatures |

### Opening Outputs

**ENVI files** (`.img` + `.hdr`):
- Open in ENVI, SNAP, QGIS (with GDAL), Python (spectral.io.envi)

**NumPy files** (`.npy`):
```python
import numpy as np
ndvi = np.load('EO1H2020342013284110KF_NDVI.npy')
```

---

## Troubleshooting

### Common Issues

#### 1. "Entity ID not found in METADATA.csv"

**Cause:** Metadata CSV doesn't have entry for your image

**Solution:**
```bash
python create_metadata_EO1H2020342013284110KF.py
```

#### 2. "File not found: L1R folder"

**Cause:** L1R ZIP not extracted

**Solution:**
- Extract `EO1H2020342013284110KF_1R.ZIP` to `L1R/EO1H2020342013284110KF/`

#### 3. "TileWidth must be multiple of 16"

**Cause:** Rasterio tiling issue

**Solution:** Code already uses fixed version (`processImage_fixed`) that disables tiling

#### 4. "Google Earth Engine authentication failed"

**Cause:** GEE not initialized or project ID not set

**Solution:**
1. Check line 993 in `process_hyperion.py`:
   ```python
   GEE_PROJECT_ID = 'remote-sensing-478802'  # Your project ID
   ```
2. Run `earthengine authenticate` in terminal if needed

#### 5. "SMARTS executable not found"

**Cause:** SMARTS path not configured

**Solution:** Check line 1014 in `process_hyperion.py`:
```python
smartsPath = 'C:/Program Files/SMARTS_295_PC/'
```

#### 6. "Wavelength field causes SNAP errors"

**Cause:** SNAP appends wavelengths to band names in expressions

**Solution:** Already handled by `fix_envi_hdr_for_snap()` function. Configure at lines 1087-1093:
```python
snap_keep_wavelength = False  # Set to False to avoid band math issues
```

---

## Processing Checklist

Use this checklist to ensure everything is ready:

- [ ] L1R extracted to `L1R/EO1H2020342013284110KF/`
- [ ] L1T ZIP in `L1T/EO1H2020342013284110KF_1T.ZIP` (keep zipped)
- [ ] XML metadata downloaded
- [ ] Run `create_metadata_EO1H2020342013284110KF.py`
- [ ] Verify row added to `METADATA/METADATA.csv`
- [ ] Update `fname` in `process_hyperion.py` line 1051
- [ ] Google Earth Engine authenticated
- [ ] SMARTS installed and path configured
- [ ] Run `python process_hyperion.py`
- [ ] Check outputs in `OUT/` folder

---

## Additional Notes

### Comparison with Previous Image

| Parameter | EO1H2020342016359110KF (2016) | EO1H2020342013284110KF (2013) |
|-----------|-------------------------------|-------------------------------|
| Acquisition Date | 2016-12-24 (Winter) | 2013-10-11 (Autumn) |
| Sun Elevation | 8.49° (low, winter) | 36.75° (higher, autumn) |
| Look Angle | -7.39° | -25.68° (more off-nadir) |
| Scene Center | 37.62°N, -6.59°W | 37.56°N, -6.58°W |
| Area | Same region (Spain) | Same region (Spain) |

**Implications:**
- 2013 image has higher sun elevation → better illumination
- 2013 image has larger look angle → more atmospheric path
- Seasonal differences in vegetation (autumn vs winter)

---

## References

- **SUREHYP**: Hyperspectral preprocessing package
- **SMARTS**: Simple Model of the Atmospheric Radiative Transfer of Sunshine
- **Google Earth Engine**: Cloud-based geospatial analysis platform
- **USGS EarthExplorer**: https://earthexplorer.usgs.gov/

---

**Last Updated:** 2024-12-02
**Author:** Automated workflow guide for Hyperion processing
