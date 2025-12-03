# Hyperion Processing Troubleshooting Guide

## Quick Reference for Common Issues

### Issue 1: KeyError: 'wavelength'
**Error message:**
```
KeyError: 'wavelength'
```

**Status:** ✅ FIXED AUTOMATICALLY

**What happens now:**
- Script detects missing wavelength field
- Automatically loads from `*_spectral_info.txt` file
- Restores wavelengths temporarily and continues processing

**User action required:** None - fix is automatic

---

### Issue 2: ValueError: Out of bounds in dimension
**Error message:**
```
ValueError: One of the requested xi is out of bounds in dimension 0
```

**Status:** ✅ FIXED AUTOMATICALLY

**What happens now:**
- Script checks terrain parameters against LUT bounds
- Automatically clips out-of-range values
- Prints warnings if clipping occurs
- Continues with topographic correction

**User action required:** None - fix is automatic

**Optional improvements:**
If you see many clipping warnings, you can increase the LUT range by editing the configuration:

```python
# Around line 1419 in process_hyperion.py
pathToReflectanceImage, R, bands = atmospheric_correction(
    pathToRadianceImage,
    pathToReflectanceImage,
    stepAltit=2,  # Increase from 1 to 2 km for more extreme terrain
    # ...
)
```

---

### Issue 3: DEM Download Fails
**Error message:**
```
Error: Could not retrieve elevation data
```

**Possible causes:**
- No internet connection
- Google Earth Engine authentication expired
- DEM not available for this region

**Solutions:**
1. Check internet connection
2. Re-authenticate GEE: Run `earthengine authenticate` in terminal
3. Disable topographic correction:
   ```python
   use_topo = False  # Line ~1268
   ```

---

### Issue 4: SMARTS Executable Not Found
**Error message:**
```
FileNotFoundError: smarts295bat.exe
```

**Solutions:**
1. Verify SMARTS installation path in configuration:
   ```python
   smartsPath = 'C:/Program Files/SMARTS_295_PC/'  # Line ~1153
   ```
2. Download SMARTS if not installed from: http://www.nrel.gov/rredc/smarts/

---

### Issue 5: Missing L1R or L1T Files
**Error message:**
```
FileNotFoundError: [file path]
```

**Solutions:**
1. Check that L1R folder exists: `C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/L1R/`
2. Check that L1T ZIP exists: `C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/L1T/[fname]_1T.ZIP`
3. Verify `fname` variable matches your image ID (line ~1250)

---

### Issue 6: XML Metadata File Not Found
**Warning message:**
```
Warning: XML metadata file not found
```

**Impact:** Non-critical - processing continues without metadata conversion

**Solutions:**
1. Download XML from USGS EarthExplorer:
   - Go to https://earthexplorer.usgs.gov/
   - Search for your image ID
   - Click "Metadata" → "Export Metadata"
   - Save to `METADATA` folder
2. OR disable metadata processing:
   ```python
   process_xml_metadata = False  # Line ~1308
   ```

---

## Configuration Checklist

Before running `process_hyperion.py`, verify these settings:

### Required Settings
- [ ] **GEE_PROJECT_ID** (line ~1132): Your Google Earth Engine project ID
- [ ] **smartsPath** (line ~1153): Path to SMARTS installation
- [ ] **basePath** (line ~1228): Base path to your data folders
- [ ] **fname** (line ~1250): Hyperion image ID (e.g., 'EO1H0370412009263110KF')

### Data Folders Required
- [ ] `L1R/[fname]/` - L1R radiance data (uncompressed)
- [ ] `L1T/[fname]_1T.ZIP` - L1T georeferenced TIF files
- [ ] `METADATA/METADATA.csv` - Metadata file (created automatically from XML)

### Optional Settings
- [ ] **use_topo** (line ~1268): Enable/disable topographic correction
- [ ] **destripingMethod** (line ~1261): 'Pal' (recommended) or 'Datt'
- [ ] **snap_keep_wavelength** (line ~1292): Keep wavelengths in HDR for SNAP
- [ ] **run_postprocessing** (line ~1276): Generate quicklooks and statistics

---

## Output Files

After successful processing, you should see:

### Radiance Files (Step 1)
- `OUT/[fname]_preprocessed.img` - Preprocessed radiance image
- `OUT/[fname]_preprocessed.hdr` - ENVI header file
- `OUT/[fname]_preprocessed_spectral_info.txt` - Wavelength reference

### Reflectance Files (Step 2)
- `OUT/[fname]_reflectance.img` - Surface reflectance image
- `OUT/[fname]_reflectance.hdr` - ENVI header file
- `OUT/[fname]_reflectance_clearview_mask.npy` - Cloud/shadow mask
- `OUT/[fname]_reflectance_cirrus_mask.npy` - Cirrus cloud mask

### Post-Processing Files (Step 3)
- `OUT/[fname]_NDVI.npy` - NDVI array
- `OUT/[fname]_statistics.txt` - Processing statistics
- `OUT/quicklooks/[fname]_RGB.png` - True color composite
- `OUT/quicklooks/[fname]_FalseColor.png` - False color (NIR-R-G)
- `OUT/quicklooks/[fname]_NDVI.png` - NDVI visualization
- `OUT/quicklooks/[fname]_spectra.png` - Sample spectra plot

### DEM Files (if topographic correction enabled)
- `OUT/elev/elev.tif` - Downloaded DEM
- `OUT/elev/elev_reprojected.tif` - Reprojected DEM
- `OUT/elev/elev_resampled.tif` - Resampled DEM

---

## Getting Help

### Check the Logs
The script provides detailed output for each processing step. Look for:
- `WARNING:` messages - Non-critical issues that were handled
- `ERROR:` messages - Critical failures that stop processing

### Documentation Files
- `FIX_DOCUMENTATION.md` - Detailed explanation of all fixes
- `TROUBLESHOOTING.md` - This file
- `METADATA_INTEGRATION_README.md` - Metadata conversion guide
- `QUICK_REFERENCE.md` - Quick start guide

### Common Warning Messages (Safe to Ignore)
- Python version warnings (Python 3.9 end of life)
- pkg_resources deprecation warnings
- "Could not compute atmospheric parameters" - Uses defaults

### Environment Information
- **Conda environment:** hyperion_roger
- **Python version:** 3.9.23 (upgrade to 3.10+ recommended)
- **Key dependencies:** surehyp, spectral, rasterio, richdem, google earth engine

---

## Quick Start

1. **Activate environment:**
   ```bash
   conda activate hyperion_roger
   ```

2. **Verify configuration:**
   - Edit `process_hyperion.py`
   - Set GEE_PROJECT_ID, basePath, fname
   - Check that data folders exist

3. **Run processing:**
   ```bash
   python process_hyperion.py
   ```

4. **Check outputs:**
   - Look in `OUT/` folder for reflectance image
   - Look in `OUT/quicklooks/` for visualizations

5. **For new datasets:**
   - Update `fname` variable only
   - All other settings remain the same
   - Script automatically handles new data

---

## Performance Tips

### Speed up processing:
- Disable local destriping: `localDestriping = False`
- Disable topographic correction: `use_topo = False`
- Disable post-processing: `run_postprocessing = False`

### Increase accuracy:
- Enable local destriping: `localDestriping = True`
- Increase LUT step sizes: `stepAltit=2, stepTilt=10, stepWazim=15`

### Save disk space:
- Delete temporary DEM files after processing
- Compress output images to `.zip` format
- Keep only final reflectance files
