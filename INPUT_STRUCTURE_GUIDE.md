# Input Structure Guide for Inundation Mapping

## Overview

This document describes the complete input structure and workflow for the Hyperion inundation mapping system with NDWI endmember.

---

## 📁 Directory Structure

```
C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/
│
├── L1R/                                    # Raw Hyperion L1R data (uncompressed)
│   ├── EO1H2020342013284110KF/            # Image 1 folder (2013)
│   │   ├── EO1H2020342013284110KF_L1R.TIF
│   │   └── EO1H2020342013284110KF_MTL.txt
│   └── EO1H2020342016359110KF/            # Image 2 folder (2016)
│       ├── EO1H2020342016359110KF_L1R.TIF
│       └── EO1H2020342016359110KF_MTL.txt
│
├── L1T/                                    # Georeferenced L1T data (optional)
│   └── filteredImages/                    # Filtered images
│
├── METADATA/                               # Metadata files
│   └── METADATA.csv                       # USGS metadata (optional)
│
├── OUT/                                    # All outputs go here
│   ├── EO1H2020342013284110KF_preprocessed.img/.hdr    # Radiance (Step 1)
│   ├── EO1H2020342013284110KF_reflectance.img/.hdr     # Reflectance (Step 2)
│   ├── EO1H2020342013284110KF_NDVI.npy                 # NDVI output
│   ├── EO1H2020342013284110KF_NDWI.npy                 # NDWI output (NEW!)
│   ├── EO1H2020342016359110KF_preprocessed.img/.hdr    # Image 2 radiance
│   ├── EO1H2020342016359110KF_reflectance.img/.hdr     # Image 2 reflectance
│   ├── EO1H2020342016359110KF_NDVI.npy                 # Image 2 NDVI
│   ├── EO1H2020342016359110KF_NDWI.npy                 # Image 2 NDWI (NEW!)
│   │
│   ├── inundation/                         # Inundation analysis outputs (NEW!)
│   │   ├── *_inundation_map.tif/.npy
│   │   ├── *_ndwi_change.tif/.npy
│   │   ├── *_ndwi_early.tif/.npy
│   │   ├── *_ndwi_late.tif/.npy
│   │   ├── *_comparison.png
│   │   ├── *_overlay.png
│   │   ├── *_histogram.png
│   │   └── *_statistics.json
│   │
│   ├── quicklooks/                         # Visualization outputs
│   │   ├── EO1H2020342013284110KF_RGB.png
│   │   ├── EO1H2020342013284110KF_NDVI.png
│   │   ├── EO1H2020342013284110KF_NDWI.png    # (NEW!)
│   │   ├── EO1H2020342016359110KF_RGB.png
│   │   ├── EO1H2020342016359110KF_NDVI.png
│   │   └── EO1H2020342016359110KF_NDWI.png    # (NEW!)
│   │
│   └── elev/                               # DEM data (downloaded from GEE)
│       └── DEM files per image
│
└── SUREHYP/                                # Processing scripts
    ├── process_hyperion.py                 # Main processing script (MODIFIED)
    ├── inundation_mapping.py               # Inundation analysis (NEW!)
    ├── test_ndwi_postprocessing.py         # Test script (NEW!)
    ├── INUNDATION_MAPPING_README.md        # Documentation (NEW!)
    └── INPUT_STRUCTURE_GUIDE.md            # This file (NEW!)
```

---

## 🔄 Complete Workflow

### **WORKFLOW 1: Single Image Processing (with NDWI)**

#### Step 1: Prepare Input Data

**Required files in `L1R/` folder:**
```
L1R/EO1H2020342013284110KF/
├── EO1H2020342013284110KF_L1R.TIF    # Hyperion Level 1R radiance image
└── EO1H2020342013284110KF_MTL.txt    # Metadata file
```

#### Step 2: Configure `process_hyperion.py`

Edit the configuration section (lines 1733-1870):

```python
# ============================================================
# CONFIGURATION PARAMETERS
# ============================================================

# 1. GOOGLE EARTH ENGINE PROJECT ID
GEE_PROJECT_ID = 'remote-sensing-478802'  # Your actual GEE project ID

# 2. PATHS CONFIGURATION
basePath = 'C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/'
pathToL1Rmetadata = basePath + 'METADATA/METADATA.csv'
pathToL1Rimages = basePath + 'L1R/'
pathToL1Timages = basePath + 'L1T/'
pathToL1TimagesFiltered = basePath + 'L1T/filteredImages/'
pathOut = basePath + 'OUT/'

# 3. IMAGE CONFIGURATION
fname = 'EO1H2020342013284110KF'  # Image ID to process
nameOut_radiance = fname + '_preprocessed'
nameOut_reflectance = fname + '_reflectance'

# 4. PROCESSING OPTIONS
destripingMethod = 'Pal'        # 'Pal' or 'Datt'
localDestriping = False         # True for slower but better destriping
use_topo = True                 # Topographic correction (requires DEM)
demID = 'NASA/NASADEM_HGT/001'  # DEM source from GEE
use_flat_dem = False            # True to skip DEM download (use flat terrain)

# 5. POST-PROCESSING
run_postprocessing = True       # Generate quicklooks, NDVI, NDWI
```

#### Step 3: Run Processing

```bash
conda activate Hyperion_roger
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
python process_hyperion.py
```

#### Step 4: Outputs Generated

**Primary outputs:**
- `OUT/EO1H2020342013284110KF_preprocessed.img/.hdr` - Preprocessed radiance
- `OUT/EO1H2020342013284110KF_reflectance.img/.hdr` - Surface reflectance
- `OUT/EO1H2020342013284110KF_NDVI.npy` - NDVI array
- `OUT/EO1H2020342013284110KF_NDWI.npy` - **NDWI array (NEW!)**

**Visualizations:**
- `OUT/quicklooks/EO1H2020342013284110KF_RGB.png`
- `OUT/quicklooks/EO1H2020342013284110KF_FalseColor.png`
- `OUT/quicklooks/EO1H2020342013284110KF_NDVI.png`
- `OUT/quicklooks/EO1H2020342013284110KF_NDWI.png` - **(NEW!)**
- `OUT/quicklooks/EO1H2020342013284110KF_spectra.png`
- `OUT/quicklooks/EO1H2020342013284110KF_valid_pixels.png`

**Masks & Statistics:**
- `OUT/EO1H2020342013284110KF_reflectance_clearview_mask.npy`
- `OUT/EO1H2020342013284110KF_reflectance_cirrus_mask.npy`
- `OUT/EO1H2020342013284110KF_valid_pixels_mask.npy`
- `OUT/EO1H2020342013284110KF_statistics.txt` - Includes NDVI & NDWI stats

---

### **WORKFLOW 2: Multi-Temporal Inundation Mapping**

#### Prerequisites

**You must have already processed both images** using Workflow 1 above, so these files exist:
- `OUT/EO1H2020342013284110KF_reflectance.img/.hdr`
- `OUT/EO1H2020342016359110KF_reflectance.img/.hdr`

#### Step 1: Configure `inundation_mapping.py`

Edit the configuration section (lines 648-663):

```python
# ============================================================
# CONFIGURATION
# ============================================================

base_path = 'C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/'
output_path = base_path + 'OUT/'

# Image IDs (chronological order: earlier first, later second)
image1_id = 'EO1H2020342013284110KF'  # Earlier date (2013)
image2_id = 'EO1H2020342016359110KF'  # Later date (2016)

# Processing options
run_preprocessing = False  # Always False (use existing reflectance files)
inundation_threshold = 0.1  # NDWI increase threshold

# Threshold options:
# 0.05 = Very sensitive (detects small changes)
# 0.10 = Moderate (default, balanced)
# 0.15 = Conservative (only large changes)
# 0.20 = Very conservative (permanent water bodies only)
```

#### Step 2: Run Inundation Mapping

```bash
conda activate Hyperion_roger
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
python inundation_mapping.py
```

#### Step 3: Outputs Generated

All outputs are saved to: `OUT/inundation/`

**GeoTIFF Files (Georeferenced for GIS):**
- `EO1H2020342013284110KF_to_EO1H2020342016359110KF_inundation_inundation_map.tif`
  - Binary map: 0 = no inundation, 1 = inundated

- `*_ndwi_change.tif`
  - Continuous NDWI change (Late - Early)
  - Range: typically -1 to +1

- `*_ndwi_early.tif`
  - NDWI from earlier date (2013), aligned

- `*_ndwi_late.tif`
  - NDWI from later date (2016), aligned

**Numpy Arrays (For Python analysis):**
- `*_inundation_map.npy`
- `*_ndwi_change.npy`
- `*_ndwi_early.npy`
- `*_ndwi_late.npy`

**Visualizations (Publication-ready):**
- `*_comparison.png` - Four-panel view:
  - Top-left: Early NDWI
  - Top-right: Late NDWI
  - Bottom-left: NDWI change
  - Bottom-right: Binary inundation map

- `*_overlay.png` - Inundation overlay:
  - Grayscale background: Late NDWI
  - Red overlay: Inundated areas

- `*_histogram.png` - NDWI change distribution:
  - Blue histogram: Frequency of changes
  - Black dashed line: Zero change
  - Red dashed line: Inundation threshold

**Statistics (JSON format):**
- `*_statistics.json` - Contains:
  ```json
  {
    "image1_id": "EO1H2020342013284110KF",
    "image2_id": "EO1H2020342016359110KF",
    "date1": "2013 (DOY 284)",
    "date2": "2016 (DOY 359)",
    "aligned_shape": [3120, 830],
    "n_valid_pixels": 2589600,
    "n_inundated_pixels": 1660,
    "percent_inundated": 0.064,
    "ndwi_change_mean": -0.092,
    "ndwi_change_std": 0.195,
    "ndwi_change_min": -0.628,
    "ndwi_change_max": 0.653,
    "threshold": 0.1
  }
  ```

---

## 🆕 What's New in This Version

### 1. **NDWI Endmember Added** ⭐

**In `process_hyperion.py`:**
- New function: `compute_ndwi(R, bands)` (lines 1248-1279)
- Formula: `NDWI = (Green - NIR) / (Green + NIR)`
- Hyperion bands used:
  - Green: ~550 nm (automatically selected)
  - NIR: ~860 nm (automatically selected)

**Post-processing updates:**
- NDWI computed automatically alongside NDVI
- NDWI visualization saved to quicklooks
- NDWI statistics added to `*_statistics.txt`

### 2. **Inundation Mapping System** ⭐

**New script: `inundation_mapping.py`**

Key capabilities:
- **Dual-image processing:** Efficiently handles two time periods
- **Spatial alignment:** Automatic geographic alignment using geotransforms
- **Subset extraction:** Processes only overlapping regions
- **Inundation detection:** Based on NDWI increase threshold
- **Multiple outputs:** GeoTIFFs, arrays, visualizations, statistics

### 3. **Bug Fix** 🐛

**Fixed in `process_hyperion.py` (lines 1334-1354):**
- `plot_sample_spectra()` function now handles spectral library memmap objects
- Uses coordinate iteration instead of boolean indexing
- Prevents `IndexError: too many indices for array`

---

## 📊 Input Data Requirements

### For Single Image Processing

**Minimum required:**
1. L1R folder: `L1R/EO1H2020342013284110KF/`
2. L1R TIF file inside folder
3. MTL metadata file inside folder
4. GEE Project ID configured
5. SMARTS installed at: `C:/Program Files/SMARTS_295_PC/`

**Optional:**
- L1T georeferenced data (if available)
- Custom DEM (otherwise downloaded from GEE)
- METADATA.csv (not critical)

### For Inundation Mapping

**Required (must exist before running):**
1. `OUT/image1_reflectance.img/.hdr` - Image 1 reflectance
2. `OUT/image2_reflectance.img/.hdr` - Image 2 reflectance
3. Geospatial metadata in HDR files (geotransform, projection)

**Optional (improves results):**
- `OUT/image1_reflectance_clearview_mask.npy` - Valid pixel mask
- `OUT/image2_reflectance_clearview_mask.npy` - Valid pixel mask
- `OUT/image1_reflectance_spectral_info.txt` - Wavelength metadata
- `OUT/image2_reflectance_spectral_info.txt` - Wavelength metadata

---

## 🎯 Key Configuration Parameters

### `process_hyperion.py` Configuration

| Parameter | Location | Options | Description |
|-----------|----------|---------|-------------|
| `GEE_PROJECT_ID` | Line 1752 | Your GEE ID | Google Earth Engine project |
| `basePath` | Line 1788 | Path string | Base directory for all data |
| `fname` | Line 1810 | Image ID | Hyperion scene to process |
| `destripingMethod` | Line 1821 | 'Pal', 'Datt' | Destriping algorithm |
| `localDestriping` | Line 1824 | True/False | Local refinement (slower) |
| `use_topo` | Line 1828 | True/False | Topographic correction |
| `demID` | Line 1832 | GEE dataset ID | DEM source for topo correction |
| `use_flat_dem` | Line 1842 | True/False | Skip DEM download (flat terrain) |
| `run_postprocessing` | Line 1879 | True/False | Generate NDVI, NDWI, quicklooks |

### `inundation_mapping.py` Configuration

| Parameter | Location | Type | Description |
|-----------|----------|------|-------------|
| `base_path` | Line 654 | String | Base directory |
| `output_path` | Line 655 | String | Output directory (usually `base_path + 'OUT/'`) |
| `image1_id` | Line 658 | String | Earlier image ID |
| `image2_id` | Line 659 | String | Later image ID |
| `run_preprocessing` | Line 662 | Boolean | Always False (not implemented) |
| `inundation_threshold` | Line 663 | Float (0.0-1.0) | NDWI increase threshold |

**Threshold Selection Guide:**
- **0.05:** Very sensitive - detects all water-related changes (may include noise)
- **0.10:** Default - balanced sensitivity for moderate flooding events
- **0.15:** Conservative - only significant water increases
- **0.20:** Very conservative - permanent water body changes only

---

## ⚙️ System Requirements

### Software Dependencies

**Conda Environment: `Hyperion_roger`**

Required packages:
```
- Python 3.9+
- numpy
- scipy
- matplotlib
- gdal (osgeo)
- spectral (Python Spectral Library)
- earthengine-api
- surehyp (custom package)
```

**External Software:**
- SMARTS 2.9.5 (Atmospheric radiative transfer)
- Google Earth Engine account (for DEM download)

### Hardware Requirements

**Minimum:**
- 8 GB RAM
- 10 GB free disk space per image
- Windows/Linux/Mac

**Recommended:**
- 16 GB RAM
- SSD for faster I/O
- Multi-core processor for faster processing

---

## 🔍 Verification Checklist

Before running inundation mapping, verify:

### ✅ For Each Image:

- [ ] Reflectance file exists: `OUT/{image_id}_reflectance.img`
- [ ] HDR file exists: `OUT/{image_id}_reflectance.hdr`
- [ ] NDWI file generated: `OUT/{image_id}_NDWI.npy`
- [ ] Wavelength info available (in HDR or `*_spectral_info.txt`)
- [ ] Geospatial metadata present (check HDR for `map info` field)

### ✅ System Configuration:

- [ ] Conda environment `Hyperion_roger` activated
- [ ] GEE Project ID configured in script
- [ ] Base paths match your directory structure
- [ ] SMARTS installed (if running full processing)

### ✅ Image Compatibility:

- [ ] Both images are from same geographic area
- [ ] Images have spatial overlap (script will validate)
- [ ] Images are from different time periods
- [ ] Both images processed with same configuration

---

## 📈 Expected Processing Times

### Single Image Processing (`process_hyperion.py`)

| Step | Time | Notes |
|------|------|-------|
| Preprocessing (destriping) | 2-5 min | Depends on `localDestriping` |
| DEM download (if needed) | 1-3 min | From Google Earth Engine |
| Atmospheric correction | 5-10 min | SMARTS radiative transfer |
| Post-processing (NDVI/NDWI) | 1-2 min | Includes visualizations |
| **Total** | **10-20 min** | Per image |

### Inundation Mapping (`inundation_mapping.py`)

| Step | Time | Notes |
|------|------|-------|
| Loading reflectance | 10-20 sec | ENVI memmap loading |
| Computing NDWI (×2) | 5-10 sec | Per image |
| Spatial alignment | 5 sec | Geotransform calculations |
| Inundation detection | 2-5 sec | Threshold application |
| Saving outputs | 10-15 sec | GeoTIFFs + arrays |
| Visualizations | 10-15 sec | PNG generation |
| **Total** | **1-2 min** | Uses existing reflectance |

---

## 🚨 Common Issues & Solutions

### Issue 1: "Reflectance file not found"
**Cause:** Inundation script run before processing images
**Solution:** Run `process_hyperion.py` for both images first

### Issue 2: "No spatial overlap between images"
**Cause:** Images from different geographic locations
**Solution:** Ensure both images cover the same area (check path/row)

### Issue 3: "IndexError: too many indices for array"
**Cause:** Old version of `plot_sample_spectra()` function
**Solution:** Use updated version (lines 1334-1354 fixed)

### Issue 4: Very low inundation percentage (< 0.1%)
**Cause:** May be correct for your images, or threshold too high
**Solution:**
- Check NDWI change histogram
- Try lower threshold (e.g., 0.05)
- Verify images are from different flood states

### Issue 5: GEE authentication error
**Cause:** Google Earth Engine not authenticated
**Solution:**
```bash
earthengine authenticate
```
Follow prompts to authenticate with your Google account

---

## 📚 Additional Resources

**Documentation Files:**
- `INUNDATION_MAPPING_README.md` - Comprehensive technical documentation
- `INPUT_STRUCTURE_GUIDE.md` - This file (input/output structure)
- Code comments in `process_hyperion.py` and `inundation_mapping.py`

**Test Scripts:**
- `test_ndwi_postprocessing.py` - Test NDWI computation for single image

**Output Examples:**
- Check `OUT/inundation/` for example outputs from test run

---

## 🎓 Quick Start Guide

**First Time Setup (5 minutes):**

1. Ensure raw L1R data in correct folders
2. Configure GEE Project ID in `process_hyperion.py`
3. Verify paths match your directory structure
4. Activate conda environment

**Processing First Image Pair (30 minutes):**

```bash
# Step 1: Activate environment
conda activate Hyperion_roger
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"

# Step 2: Process first image (edit fname in script)
python process_hyperion.py  # Set fname = 'EO1H2020342013284110KF'

# Step 3: Process second image (edit fname in script)
python process_hyperion.py  # Set fname = 'EO1H2020342016359110KF'

# Step 4: Run inundation mapping
python inundation_mapping.py

# Step 5: Check outputs
cd ../OUT/inundation/
ls -lh
```

**Subsequent Analyses (2 minutes):**

Once images are preprocessed, you can:
- Adjust threshold in `inundation_mapping.py`
- Re-run `python inundation_mapping.py` instantly
- Compare different thresholds quickly

---

**Last Updated:** 2025-12-18
**Version:** 1.0
**Compatible With:** SUREHYP v1.0, Python 3.9+, GDAL 3.x
