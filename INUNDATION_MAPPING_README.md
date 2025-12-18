# Multi-Temporal Inundation Mapping from Hyperion NDWI

## Overview

This workflow enables the detection of inundated areas by comparing Normalized Difference Water Index (NDWI) values from two Hyperion images acquired at different time periods. The system automatically handles spatial alignment, computes NDWI changes, and generates comprehensive outputs including GeoTIFFs and visualizations.

## What Was Implemented

### 1. NDWI Index Integration

**File Modified:** `process_hyperion.py`

**Changes:**
- Added `compute_ndwi()` function (lines 1248-1279) that calculates NDWI using:
  - **Formula:** NDWI = (Green - NIR) / (Green + NIR)
  - **Hyperion bands:** Green ~550 nm, NIR ~860 nm

- Integrated into post-processing pipeline:
  - Automatic NDWI computation for all processed images
  - Saves NDWI as `.npy` numpy array
  - Generates NDWI visualization with Blues colormap
  - Includes NDWI statistics in output reports

- **Bug Fix:** Fixed `plot_sample_spectra()` function (lines 1334-1354) to handle spectral library memmap objects correctly by iterating through valid coordinates instead of using boolean indexing.

### 2. Inundation Mapping Script

**New File:** `inundation_mapping.py`

**Key Functions:**

#### `load_reflectance_and_bands(reflectance_path)`
- Loads ENVI format reflectance images
- Extracts wavelength information from HDR or spectral info files
- Retrieves geospatial metadata (geotransform, projection)

#### `compute_ndwi_from_reflectance(R, bands, image_name)`
- Computes NDWI using the imported function from `process_hyperion.py`
- Ensures 2D output (handles dimension issues)
- Provides logging and statistics

#### `align_images_spatially(img1, gt1, img2, gt2)`
- **Core spatial alignment algorithm:**
  1. Extracts geotransform parameters for both images
  2. Calculates geographic bounds for each image
  3. Determines overlapping geographic region
  4. Converts overlap bounds to pixel coordinates
  5. Extracts aligned subsets from both images
  6. Ensures exact dimension matching (handles rounding errors)

- **Features:**
  - Handles different pixel sizes between images
  - Validates overlap exists before proceeding
  - Provides detailed logging of alignment process
  - Returns aligned subsets with matching dimensions

#### `compute_inundation_map(ndwi_early, ndwi_late, valid_mask_early, valid_mask_late, threshold)`
- Calculates NDWI change: `ndwi_change = ndwi_late - ndwi_early`
- Combines valid pixel masks from both dates
- Classifies inundation where NDWI increase > threshold (default: 0.1)
- Computes comprehensive statistics on NDWI changes

#### `save_inundation_results(...)`
- Saves multiple output formats:
  - **Numpy arrays (.npy):** For further Python analysis
  - **GeoTIFFs (.tif):** For GIS software (QGIS, ArcGIS, SNAP)
  - Preserves geospatial metadata (projection, geotransform)
  - Uses LZW compression and tiling for efficient storage

#### `visualize_inundation(...)`
Creates three comprehensive visualizations:

1. **Four-panel comparison** (`*_comparison.png`):
   - Early NDWI (Blues colormap)
   - Late NDWI (Blues colormap)
   - NDWI change (RdBu_r colormap, -0.3 to +0.3)
   - Binary inundation map (Gray to Blue)

2. **Overlay visualization** (`*_overlay.png`):
   - Late NDWI as grayscale background
   - Inundated areas highlighted in red
   - Legend explaining visualization

3. **Histogram** (`*_histogram.png`):
   - Distribution of NDWI changes
   - Zero-change line (dashed black)
   - Inundation threshold line (dashed red)
   - Shows frequency distribution

#### `process_image_pair_for_inundation(...)`
**Main workflow orchestrator:**

1. **Load reflectance images** - Checks for preprocessed files
2. **Compute NDWI** - For both time periods
3. **Spatial alignment** - Extracts overlapping regions
4. **Compute inundation** - Detects changes above threshold
5. **Save and visualize** - Generates all outputs

**Returns:** Dictionary with inundation map, NDWI arrays, and statistics

## Output Files

### Directory Structure
```
OUT/
├── inundation/
│   ├── *_inundation_map.tif          # Binary GeoTIFF (0=no inundation, 1=inundated)
│   ├── *_inundation_map.npy          # Binary numpy array
│   ├── *_ndwi_change.tif             # Continuous NDWI change GeoTIFF
│   ├── *_ndwi_change.npy             # Continuous NDWI change array
│   ├── *_ndwi_early.tif              # Early date NDWI GeoTIFF
│   ├── *_ndwi_early.npy              # Early date NDWI array
│   ├── *_ndwi_late.tif               # Late date NDWI GeoTIFF
│   ├── *_ndwi_late.npy               # Late date NDWI array
│   ├── *_comparison.png              # Four-panel comparison
│   ├── *_overlay.png                 # Inundation overlay
│   ├── *_histogram.png               # NDWI change distribution
│   └── *_statistics.json             # Comprehensive statistics
└── quicklooks/
    ├── EO1H2020342013284110KF_NDWI.png
    └── EO1H2020342016359110KF_NDWI.png
```

### Statistics JSON Structure
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

## Usage

### Prerequisites

1. **Preprocessed reflectance images** must exist for both dates:
   - Run `process_hyperion.py` for each image first
   - This generates `*_reflectance.img` files in `OUT/` directory

2. **Conda environment:** `Hyperion_roger`

### Configuration

Edit `inundation_mapping.py` (lines 644-651):

```python
# Image IDs (chronological order: earlier first, later second)
image1_id = 'EO1H2020342013284110KF'  # Earlier date (2013, day 284)
image2_id = 'EO1H2020342016359110KF'  # Later date (2016, day 359)

# Processing options
inundation_threshold = 0.1  # NDWI increase threshold (default: 0.1)
```

**Threshold interpretation:**
- `0.1` = Detect areas with NDWI increase ≥ 0.1
- Lower values = More sensitive (detects smaller changes)
- Higher values = More conservative (only detects large changes)
- Typical range: 0.05 to 0.2

### Execution

```bash
# Activate environment
conda activate Hyperion_roger

# Navigate to project directory
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"

# Run inundation mapping
python inundation_mapping.py
```

### Expected Runtime
- Loading images: ~10-20 seconds
- Computing NDWI: ~5-10 seconds per image
- Spatial alignment: ~5 seconds
- Inundation detection: ~2-5 seconds
- Saving and visualization: ~10-15 seconds
- **Total: ~1-2 minutes**

## Technical Details

### Spatial Alignment Algorithm

The alignment process handles images with different:
- Spatial extents
- Origin points
- Pixel sizes (with warning)

**Steps:**
1. Extract geotransform: `(x_origin, pixel_width, 0, y_origin, 0, pixel_height)`
2. Calculate bounds: `x_max = x_origin + pixel_width × cols`
3. Find overlap: `overlap_x_min = max(x1_origin, x2_origin)`
4. Convert to pixels: `col_start = int((overlap_x_min - x_origin) / pixel_width)`
5. Extract subsets: `img1_aligned = img1[row_start:row_end, col_start:col_end]`
6. Ensure matching: Crop to minimum dimensions if rounding errors exist

### NDWI Interpretation

**NDWI Values:**
- **+1.0:** Pure water (high water content)
- **+0.5 to +1.0:** Water bodies, wet soil
- **0.0 to +0.5:** Moist vegetation, damp surfaces
- **-0.5 to 0.0:** Dry vegetation, bare soil
- **-1.0:** Very dry surfaces, rocks

**NDWI Change Interpretation:**
- **Positive change:** Increased water presence (inundation)
- **Negative change:** Decreased water presence (drying)
- **Near zero:** No significant change

### Performance Optimizations

1. **Memory Efficient:**
   - Processes only overlapping regions, not full images
   - Uses subsampling for mean spectrum calculation (max 10,000 pixels)

2. **Efficient I/O:**
   - GeoTIFFs use LZW compression and tiling
   - Numpy arrays use memory-mapped files when possible

3. **Parallel Ready:**
   - Functions are independent and can be parallelized
   - Could process multiple image pairs simultaneously

## Example Results (Test Case)

**Images Analyzed:**
- **Early:** EO1H2020342013284110KF (October 11, 2013)
- **Late:** EO1H2020342016359110KF (December 24, 2016)

**Results:**
- **Overlapping area:** 3,120 × 830 pixels = 2.59 million pixels
- **Valid pixels:** 2,589,600 (both dates clear)
- **Inundated pixels:** 1,660 (0.064% of area)
- **NDWI change:**
  - Mean: -0.092 (overall slight drying)
  - Std Dev: 0.195 (moderate variability)
  - Range: -0.628 to +0.653

**Interpretation:**
- Most of the area experienced slight drying (negative mean)
- Small localized areas (1,660 pixels) showed significant water increase
- These could represent:
  - Seasonal flooding
  - New water bodies
  - Increased soil moisture
  - Vegetation phenology changes

## Workflow Efficiency Features

### 1. No Redundant Processing
- Uses existing reflectance files
- Doesn't recompute atmospheric correction
- Loads only necessary bands for NDWI

### 2. Automatic Validation
- Checks for file existence before processing
- Validates spatial overlap
- Ensures dimension compatibility
- Provides informative error messages

### 3. Comprehensive Outputs
- Multiple formats for different use cases
- Geospatial metadata preserved
- Both raster (GeoTIFF) and vector (stats) outputs
- Publication-ready visualizations

### 4. Reproducibility
- All parameters logged in JSON
- Random seed set for sampling operations
- Clear documentation of band selections
- Version control friendly (Python scripts)

## Troubleshooting

### Common Issues

**1. "Reflectance file not found"**
- **Solution:** Run `process_hyperion.py` first for both images
- Check `OUT/` directory for `*_reflectance.img` files

**2. "No spatial overlap between images"**
- **Solution:** Ensure images cover the same geographic area
- Check image metadata for coordinate systems

**3. "IndexError: too many indices for array"**
- **Solution:** This was fixed in the updated `plot_sample_spectra()` function
- Ensure you have the latest version of `process_hyperion.py`

**4. Very small inundation percentages**
- **Solution:** May be correct if images are from different seasons
- Try lowering threshold (e.g., 0.05) for more sensitive detection
- Check NDWI change histogram for appropriate threshold

### Validation Steps

1. **Visual inspection:**
   - Open `*_comparison.png` to see all stages
   - Check if inundated areas make geographic sense

2. **GIS verification:**
   - Load GeoTIFFs into QGIS or ArcGIS
   - Compare with reference water body maps
   - Check alignment with satellite imagery

3. **Statistical review:**
   - Read `*_statistics.json`
   - Verify NDWI ranges are reasonable
   - Check if threshold is appropriate for your study area

## Advanced Usage

### Custom Threshold Selection

Based on your study area and objectives:

```python
# For detecting permanent water bodies (conservative)
inundation_threshold = 0.2

# For detecting all water-related changes (sensitive)
inundation_threshold = 0.05

# For seasonal flooding (moderate)
inundation_threshold = 0.1  # Default
```

### Batch Processing Multiple Image Pairs

Create a script to process multiple pairs:

```python
image_pairs = [
    ('EO1H2020342013284110KF', 'EO1H2020342016359110KF'),
    ('EO1H2020342014150110KF', 'EO1H2020342017150110KF'),
    # Add more pairs...
]

for img1, img2 in image_pairs:
    results = process_image_pair_for_inundation(
        image1_id=img1,
        image2_id=img2,
        base_path=base_path,
        output_path=output_path,
        inundation_threshold=0.1
    )
    print(f"Completed: {img1} to {img2}")
```

### Extracting Inundation Polygons

Convert raster inundation map to vector polygons:

```python
from osgeo import gdal, ogr
import numpy as np

# Load inundation map
ds = gdal.Open('inundation_map.tif')
band = ds.GetRasterBand(1)
arr = band.ReadAsArray()

# Create vector output
driver = ogr.GetDriverByName('ESRI Shapefile')
dst_ds = driver.CreateDataSource('inundation_polygons.shp')
dst_layer = dst_ds.CreateLayer('inundation', srs=None)

# Polygonize
gdal.Polygonize(band, None, dst_layer, -1, [], callback=None)

dst_ds = None
ds = None
```

## Future Enhancements

Potential improvements for future versions:

1. **Automatic threshold selection:** Use Otsu's method or k-means clustering
2. **Change persistence:** Multi-temporal analysis across >2 dates
3. **Confidence maps:** Estimate uncertainty in inundation detection
4. **Object-based analysis:** Group inundated pixels into water bodies
5. **Time series analysis:** Track water body evolution over time
6. **Integration with MODIS:** Compare with coarser resolution but daily data

## Citation

If you use this workflow in research, consider citing:

```
@software{hyperion_inundation_mapping,
  title={Multi-Temporal Inundation Mapping from Hyperion NDWI},
  author={Generated with Claude Code},
  year={2025},
  note={Based on SUREHYP atmospheric correction framework}
}
```

## References

- **NDWI:** McFeeters, S.K. (1996). "The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features." International Journal of Remote Sensing, 17(7), 1425-1432.

- **SUREHYP:** Atmospheric correction framework for Hyperion imagery

- **Hyperion:** EO-1 Hyperion hyperspectral sensor (2000-2017)

## Support

For issues or questions:
1. Check this README's troubleshooting section
2. Review code comments in `inundation_mapping.py`
3. Examine example outputs in `OUT/inundation/`
4. Verify input data quality and preprocessing

---

**Last Updated:** 2025-12-18
**Script Version:** 1.0
**Compatible with:** Python 3.9+, GDAL 3.x, spectral 0.22+
