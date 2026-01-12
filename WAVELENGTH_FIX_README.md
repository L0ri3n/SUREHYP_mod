# Wavelength Field Fix - README

## Problem
The `fix_envi_hdr_for_snap()` function was removing the wavelength field from HDR files to ensure SNAP compatibility. However, this caused the atmospheric correction step to fail because it needs to read wavelengths from the preprocessed radiance file.

**Error:**
```
KeyError: 'wavelength'
```

## Solution Implemented

### Two-Part Fix:

#### 1. Keep Wavelengths in Radiance File (Primary Fix)
**File**: `process_hyperion.py` (line ~467)

The preprocessing step now **keeps** the wavelength field in the radiance HDR file:
```python
# IMPORTANT: Keep wavelength field in preprocessed radiance file for atmospheric correction
# It will be removed/kept in the final reflectance file based on user settings
print('    Keeping wavelength field in radiance HDR for atmospheric correction...')
# Don't call fix_envi_hdr_for_snap here - wavelengths needed for next step
```

**Result**: The radiance file retains wavelengths → atmospheric correction can read them.

#### 2. Fallback Loader (Backup Fix)
**File**: `process_hyperion.py` (lines 483-527, 692-733)

Added `load_wavelengths_from_spectral_info()` function that reads wavelengths from the `_spectral_info.txt` file if they're missing from the HDR.

The atmospheric correction now has a try-except block:
```python
try:
    L, bands, fwhms, processing_metadata, metadata = surehyp.atmoCorrection.getImageAndParameters(pathToRadianceImage)
except KeyError as e:
    if 'wavelength' in str(e):
        # Fall back to loading from spectral_info.txt
        bands, fwhms = load_wavelengths_from_spectral_info(hdr_path)
        # ... manually extract metadata
```

## Workflow Now

```
Step 1: Preprocessing (Radiance)
  ├─ Process L1R → Radiance
  ├─ Save with wavelength field (DON'T remove)
  └─ Output: {image_id}_preprocessed.img + .hdr (WITH wavelengths)

Step 2: Atmospheric Correction
  ├─ Read radiance file (needs wavelengths ✓)
  ├─ Apply atmospheric correction
  ├─ Apply preprocessing fixes
  ├─ Save reflectance
  └─ Fix HDR for SNAP (remove/keep wavelengths based on config)

Step 3: Post-processing
  └─ Use reflectance file (wavelengths optional)
```

## Files Generated

### Radiance File (Intermediate):
```
OUT/{image_id}_preprocessed.img
OUT/{image_id}_preprocessed.hdr  ← Contains wavelength field
```

### Reflectance File (Final):
```
OUT/{image_id}_reflectance.img
OUT/{image_id}_reflectance.hdr  ← Wavelength field removed/kept per config
OUT/{image_id}_reflectance_spectral_info.txt  ← Wavelengths backup
```

## Configuration Options

In `process_hyperion.py` (lines ~1440-1455):

```python
# Option 1: Load wavelengths from external file
snap_wavelength_file = None  # or path to custom wavelength file

# Option 2: Keep wavelength in final reflectance HDR
snap_keep_wavelength = True  # True = keep, False = remove

# Current settings:
snap_keep_wavelength = True  # Keep wavelengths for SNAP visualization
```

**Recommendation**: Keep `snap_keep_wavelength = True` to retain wavelengths in the final reflectance file. This helps with SNAP visualization and future analysis.

## If You Still Get the Error

### Scenario 1: Radiance file exists from previous run
**Problem**: Old radiance file has wavelengths removed
**Solution**: Delete and re-run preprocessing
```bash
rm OUT/EO1H2020342016359110KF_preprocessed.*
python process_hyperion.py
```

### Scenario 2: Manual HDR modification
**Problem**: Someone manually removed wavelengths from radiance HDR
**Solution**: The fallback loader will automatically use `_spectral_info.txt`

### Scenario 3: spectral_info.txt missing
**Problem**: Both HDR and spectral_info.txt lack wavelength data
**Solution**: Re-run from scratch (delete all outputs)

## Verification

Check that radiance file has wavelengths:
```bash
grep -i "wavelength" OUT/EO1H2020342016359110KF_preprocessed.hdr
```

Should see:
```
wavelength = { 427.55, 437.66, ... }
```

If empty → re-run preprocessing

## Summary

✅ **Radiance file**: KEEPS wavelength field (needed for atmospheric correction)
✅ **Reflectance file**: Wavelengths kept/removed based on `snap_keep_wavelength` setting
✅ **Fallback**: Loads from `_spectral_info.txt` if wavelength field missing
✅ **Error fixed**: Atmospheric correction can now read wavelengths

---

**Status**: ✅ FIXED
**Last Updated**: 2025-12-18
