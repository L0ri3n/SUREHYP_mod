# Fixes for Dataset Processing Issues

## Fix 1: KeyError: 'wavelength' Issue

### Problem
The script was failing with `KeyError: 'wavelength'` when processing new datasets during the atmospheric correction step (Step 2).

### Root Cause
1. During preprocessing (Step 1), the `fix_envi_hdr_for_snap()` function removes the `wavelength` field from the HDR file to avoid SNAP band math issues
2. The wavelength data is saved to a separate `_spectral_info.txt` file for reference
3. However, during atmospheric correction (Step 2), `surehyp.atmoCorrection.getImageAndParameters()` expects the wavelength field to be present in the HDR file
4. This causes a `KeyError: 'wavelength'` when the function tries to access `img.metadata['wavelength']`

### Solution
The fix adds error handling in the `atmospheric_correction()` function to:

1. **Detect the KeyError**: Catch when the wavelength field is missing from the HDR
2. **Load from spectral info file**: Read wavelength and FWHM data from the `_spectral_info.txt` file that was saved during preprocessing
3. **Temporarily restore the data**: Write the wavelength and FWHM fields back into the HDR file
4. **Retry loading**: Call `surehyp.atmoCorrection.getImageAndParameters()` again, which now succeeds

### Code Changes
The fix is implemented in the `atmospheric_correction()` function (starting at line 636):

```python
# Try to load the image with surehyp's function
try:
    L, bands, fwhms, processing_metadata, metadata = surehyp.atmoCorrection.getImageAndParameters(pathToRadianceImage)
except KeyError as e:
    if 'wavelength' in str(e):
        # If wavelength is missing from HDR, load it from the spectral info file
        # Load spectral info file and restore wavelength/FWHM to HDR
        # Then retry loading
```

---

## Fix 2: ValueError: Out of Bounds in Topographic Correction (NaN in DEM Data)

### Problem
The script was failing with `ValueError: One of the requested xi is out of bounds in dimension 0` during topographic correction in Step 2.

### Root Cause
1. **Primary cause**: DEM data contains NaN (Not a Number) values for ocean/water pixels where elevation data is unavailable
2. The topographic correction uses a lookup table (LUT) that cannot handle NaN values
3. This commonly happens when:
   - The Hyperion scene includes coastal areas or water bodies
   - SRTM DEM has no data over oceans
   - The scene boundary extends beyond land coverage

### Solution
The fix detects and handles NaN values in DEM data:

1. **Detect NaN values**: Check for NaN in elevation, slope, and aspect data
2. **Create DEM visualization**: Generate a 4-panel figure showing:
   - Elevation map with terrain colors
   - Elevation with NaN pixels highlighted (ocean/water areas)
   - Slope map
   - Aspect map
3. **Fill NaN with sea level**: Replace NaN elevations with 0m (sea level), appropriate for ocean pixels
4. **Fill terrain parameters**: Replace NaN slopes with 0° (flat) and aspects with 0° (north-facing)
5. **Clip to LUT bounds**: Ensure all values are within acceptable ranges for the lookup table

### Code Changes
The fix is implemented in the `atmospheric_correction()` function (line 857-920):

```python
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

    # Create DEM visualization (4-panel figure)
    # ... visualization code ...

    # Fill NaN values with sea level and flat terrain
    if elev_has_nan:
        elev_km = np.nan_to_num(elev_km, nan=0.0)  # Sea level (0m)
    if slope_has_nan:
        slope = np.nan_to_num(slope, nan=0.0)  # Flat terrain
    if wazim_has_nan:
        wazim = np.nan_to_num(wazim, nan=0.0)  # North-facing

# Clip to ensure values are within LUT bounds
elev_km = np.clip(elev_km, altit_min, altit_max)
slope = np.clip(slope, 0, 90)
wazim = np.mod(wazim, 360)
```

### Impact
- **Appropriate handling of ocean pixels**: NaN values filled with sea level (0m) - physically correct for water
- **DEM visualization created**: 4-panel figure saved to `OUT/elev/DEM_visualization.png` showing:
  - Elevation distribution
  - Location of NaN pixels (ocean/water areas)
  - Slope and aspect maps
- **Robust processing**: Script can now handle coastal scenes with mixed land/ocean coverage
- **Minimal accuracy impact**: Ocean pixels get appropriate sea-level values

### DEM Visualization Output
The fix automatically creates a visualization file that shows:
- **Top-left**: Elevation map with terrain colors (green=low, brown=high)
- **Top-right**: Same elevation with NaN pixels highlighted in blue/purple (shows ocean extent)
- **Bottom-left**: Slope map in degrees (shows terrain roughness)
- **Bottom-right**: Aspect map 0-360° (shows slope orientation)

**Location**: `C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/OUT/elev/DEM_visualization.png`

---

## Combined Benefits
- **Automatic error handling**: Both fixes work automatically without user intervention
- **Robust processing**: Scripts can handle various datasets and terrain conditions
- **Maintains compatibility**: SNAP integration and data quality are preserved
- **Informative output**: Users are warned when adjustments are made

## Testing
To verify both fixes work:
1. Run the script with a new dataset: `python process_hyperion.py`
2. Check that Step 1 completes and creates `*_spectral_info.txt`
3. Verify Step 2 starts and restores wavelengths if needed
4. Confirm topographic correction completes without interpolation errors
5. Check for any clipping warnings in the output

## Related Configuration
The script has configuration options for SNAP compatibility (lines 1280-1292):
- `snap_wavelength_file`: Optional external wavelength file
- `snap_keep_wavelength`: Whether to keep wavelengths in HDR (True for visualization, False for band math safety)

And topographic correction options (line ~1360):
- `stepAltit`: Altitude step size for LUT (default: 1 km)
- `stepTilt`: Slope step size for LUT (default: 15°)
- `stepWazim`: Aspect step size for LUT (default: 30°)
- `use_topo`: Enable/disable topographic correction
