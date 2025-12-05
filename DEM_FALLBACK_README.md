# Enhanced DEM Fallback System for Hyperion Processing

## Overview

This enhancement adds a robust multi-level fallback strategy for Digital Elevation Model (DEM) acquisition in Hyperion image processing. It automatically handles cases where the primary DEM source fails due to missing data, corruption, or network issues.

## Problem Statement

The original implementation would fail when:
- DEM data doesn't exist for the region (e.g., ocean areas, polar regions)
- Data is corrupted in Google Earth Engine
- Network or authentication issues occur
- The primary DEM source has gaps or missing tiles

This resulted in processing failures or complete disabling of topographic correction.

## Solution: Multi-Level Fallback Strategy

The enhanced system tries multiple approaches in order:

```
1. Primary GEE DEM (e.g., SRTM)
   ↓ (if fails)
2. Fallback GEE DEMs (NASADEM, ALOS, GTOPO30)
   ↓ (if all fail)
3. Local DEM file (if provided)
   ↓ (if fails/not provided)
4. Flat terrain assumption (0° slope, constant elevation)
```

## Files Added

### 1. `dem_fallback.py`
New module containing:
- `downloadDEMfromGEE_robust()`: Enhanced DEM download with multiple fallbacks
- `apply_flat_terrain_assumption()`: Creates synthetic flat DEM when all sources fail
- `_download_dem_from_source()`: Internal function to try individual DEM sources

### 2. `process_hyperion.py` (Modified)
Enhanced `atmospheric_correction()` function with:
- New parameters: `local_dem_path`, `fallback_dems`, `use_flat_terrain_on_failure`
- Automatic DEM fallback logic
- Comprehensive error reporting

## Configuration Options

### In `process_hyperion.py` (lines ~1694-1716):

```python
# ============================================================
# DEM FALLBACK CONFIGURATION
# ============================================================

# Option 1: Local DEM backup file
local_dem_backup = None
# Example: local_dem_backup = basePath + 'DEM/my_local_dem.tif'

# Option 2: Alternative GEE DEM sources (None = use defaults)
fallback_dem_sources = None
# Example custom:
# fallback_dem_sources = [
#     {'id': 'NASA/NASADEM_HGT/001', 'band': 'elevation', 'name': 'NASADEM'},
#     {'id': 'JAXA/ALOS/AW3D30/V3_2', 'band': 'DSM', 'name': 'ALOS World 3D'},
# ]

# Option 3: Use flat terrain if all sources fail
use_flat_terrain_fallback = True  # Recommended: True
```

## Default Fallback DEMs

When `fallback_dem_sources = None`, the system automatically tries:

1. **NASA NASADEM** (`NASA/NASADEM_HGT/001`)
   - Improved version of SRTM
   - 30m resolution
   - Coverage: 60°N to 56°S

2. **ALOS World 3D** (`JAXA/ALOS/AW3D30/V3_2`)
   - 30m resolution
   - Near-global coverage
   - Good for areas where SRTM fails

3. **GTOPO30** (`USGS/GTOPO30`)
   - Lower resolution (~1km)
   - Global coverage
   - Last resort before local/flat terrain

## Usage Examples

### Example 1: Default Configuration (Recommended)
```python
# In process_hyperion.py
use_topo = True
demID = 'USGS/SRTMGL1_003'
local_dem_backup = None
fallback_dem_sources = None
use_flat_terrain_fallback = True
```

**Result**: Automatically tries SRTM → NASADEM → ALOS → GTOPO30 → Flat Terrain

### Example 2: With Local DEM Backup
```python
local_dem_backup = basePath + 'DEM/caribbean_dem.tif'
use_flat_terrain_fallback = True
```

**Result**: Tries GEE sources first, then uses your local DEM if all fail

### Example 3: Custom Fallback Order
```python
fallback_dem_sources = [
    {'id': 'JAXA/ALOS/AW3D30/V3_2', 'band': 'DSM', 'name': 'ALOS (first fallback)'},
    {'id': 'NASA/NASADEM_HGT/001', 'band': 'elevation', 'name': 'NASADEM (second)'},
]
```

**Result**: Uses your custom order instead of defaults

### Example 4: Strict Mode (No Flat Terrain)
```python
use_flat_terrain_fallback = False
```

**Result**: Processing stops if no valid DEM can be obtained (useful for research requiring real topography)

## Output and Logging

The system provides detailed logging:

```
[2/12] Download DEM images from GEE (with fallback support)
    Using enhanced DEM fallback system...

    DEM Fallback Strategy: Trying 4 GEE sources
    [1/4] Primary DEM: USGS/SRTMGL1_003 (primary)
    ✗ FAILED: HTTP error 400
    [2/4] Fallback DEM: NASADEM (improved SRTM)
    ✓ SUCCESS: Obtained DEM from NASADEM (improved SRTM)
```

Or if all fail:
```
    [4/4] Fallback DEM: GTOPO30 (global, lower res)
    ✗ FAILED: No samples returned from GEE

    All GEE sources failed. Trying local DEM: C:/path/to/local.tif
    ✗ FAILED: Local DEM error - File not found

    Applying flat terrain assumption as final fallback...

    ======================================================================
    APPLYING FLAT TERRAIN ASSUMPTION
    ======================================================================
    Creating synthetic flat DEM:
      - Dimensions: 256 x 3176 pixels
      - Elevation: 0 m (constant)
      - Slope: 0° (flat terrain)
      - Aspect: 0° (undefined for flat terrain)

    Note: Topographic correction will have minimal effect
          on flat terrain. Results will be similar to
          atmospheric correction without topo correction.
    ======================================================================
```

## When to Use Each Option

### Use Default Settings When:
- Processing diverse datasets from different regions
- You want maximum robustness without manual intervention
- Ocean/coastal areas might be included in scenes

### Use Local DEM When:
- You have high-quality local DEM data
- Working in regions with poor GEE coverage
- Processing the same region repeatedly (faster)

### Disable Flat Terrain Fallback When:
- Research requires real topographic data
- You want processing to fail if no valid DEM exists
- Quality control is more important than completion rate

## Technical Details

### DEM Validation

Each downloaded DEM is validated:
- Minimum 100 valid pixels required
- Checks for all-zero or all-NaN data
- Verifies reasonable elevation ranges
- Warns about suspiciously uniform data

### Flat Terrain Implementation

When flat terrain is applied:
1. Uses average scene elevation if available (from `getGEEdem_fixed`)
2. Falls back to sea level (0m) if average cannot be computed
3. Creates constant elevation, zero slope, zero aspect arrays
4. Topographic correction runs but has minimal effect

### Memory and Performance

- DEM download timeout: 120 seconds per source
- Interpolation uses 500x500 grid for sampled DEMs
- Flat terrain has zero overhead (simple array creation)

## Troubleshooting

### Issue: "All DEM sources failed"
**Solutions:**
1. Check internet connectivity
2. Verify GEE authentication: `ee.Initialize(project='your-project')`
3. Try providing a local DEM file
4. Enable flat terrain fallback: `use_flat_terrain_fallback = True`

### Issue: "Local DEM contains insufficient valid data"
**Solutions:**
1. Verify DEM file is not corrupted: `gdalinfo your_dem.tif`
2. Check if DEM covers your scene area
3. Ensure DEM is in a supported format (GeoTIFF recommended)

### Issue: "DEM appears uniform"
**Explanation:** The DEM has very little elevation variation. This is a warning, not an error. Processing continues but verify the DEM is correct for your region.

### Issue: Processing slow with fallbacks
**Solutions:**
1. Use local DEM to skip GEE entirely for known-good regions
2. Reduce number of fallback sources in `fallback_dem_sources`
3. Cache successful DEMs for reuse

## Testing

### Test with conda environment:
```bash
conda activate hyperion_roger
cd C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/SUREHYP/
python process_hyperion.py
```

### Test scenarios:

1. **Test fallback cascade:**
   - Set `demID` to invalid source: `'INVALID/DEM'`
   - Observe automatic fallback to NASADEM/ALOS/GTOPO30

2. **Test local DEM:**
   - Download a DEM for your region
   - Set `local_dem_backup = 'path/to/dem.tif'`
   - Disconnect internet and verify it uses local DEM

3. **Test flat terrain:**
   - Set all sources to invalid
   - Set `use_flat_terrain_fallback = True`
   - Verify processing completes with flat terrain

## Performance Comparison

| Scenario | Original System | Enhanced System |
|----------|----------------|-----------------|
| Valid primary DEM | ✓ Success | ✓ Success |
| Invalid primary DEM | ✗ Failure | ✓ Success (fallback) |
| No GEE coverage | ✗ Failure | ✓ Success (local/flat) |
| Ocean area | ✗ Failure | ✓ Success (flat terrain) |
| Network issues | ✗ Failure | ✓ Success (retry/fallback) |

## Future Enhancements

Possible improvements:
- [ ] DEM caching system to avoid re-downloading
- [ ] Automatic region detection to suggest best DEM source
- [ ] Support for additional DEM formats (NetCDF, HDF)
- [ ] Parallel DEM source testing for faster fallback
- [ ] DEM quality metrics and automatic source selection

## References

- SRTM: NASA Shuttle Radar Topography Mission
- NASADEM: https://lpdaac.usgs.gov/products/nasadem_hgtv001/
- ALOS World 3D: https://www.eorc.jaxa.jp/ALOS/en/aw3d30/
- GTOPO30: https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-global-30-arc-second-elevation-gtopo30

## Contact

For issues or questions about the DEM fallback system:
- Check the detailed error messages in console output
- Review this README for configuration options
- Verify all dependencies are installed in hyperion_roger environment

---

**Version:** 1.0
**Date:** 2025-12-05
**Compatibility:** Python 3.x, Google Earth Engine API, SUREHYP package
