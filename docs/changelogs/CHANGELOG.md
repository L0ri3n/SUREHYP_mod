# Hyperion Image Processing Script - Bug Fixes Summary

**Date:** 2026-01-13
**File Modified:** `process_hyperion.py`
**Issue:** Script crashed during Step 3 (Post-processing) with `ValueError: zero-size array to reduction operation minimum`

---

## 🐛 Problems Identified

### 1. **Zero Valid Pixels Detection Error**
- **Symptom:** Script reported "0 valid pixels" despite output images looking correct
- **Error:** `ValueError: zero-size array to reduction operation minimum which has no identity`
- **Location:** Line 1433 in `post_processing()` function

### 2. **Missing Wavelength Data**
- **Symptom:** `ValueError: No wavelength data found in HDR or spectral info file`
- **Issue:** Reflectance file's spectral info file didn't exist when loading existing processed data

---

## 🔍 Root Cause Analysis

### Primary Issue: NaN Values in Water Vapor Bands

The script masks unreliable atmospheric water vapor absorption bands (1350-1450 nm and 1800-1950 nm) by setting them to `np.nan`:

```python
# Line 589: mask_water_vapor_bands()
R[:, :, bad_bands_idx] = np.nan  # Sets water bands to NaN
```

**The Problem:**
- Valid pixel detection used `np.sum(R, axis=2) > 0`
- When ANY band contains NaN, `np.sum()` returns NaN
- `NaN > 0` evaluates to `False`
- Result: **ALL pixels incorrectly marked as invalid**

**Why Images Still Looked Correct:**
- RGB/NDVI functions use specific band indices (480, 550, 660, 850 nm)
- These bands are outside water absorption regions
- They contain valid data despite the statistical check failing

---

## ✅ Fixes Applied

### Fix 1: Handle NaN Values in Valid Pixel Detection

**Changed:** Replace `np.sum()` with `np.nansum()` to ignore NaN values

#### Location 1: `visualize_valid_pixels()` function
**Line 1198-1199**

**Before:**
```python
# Define valid pixels based on reflectance data
valid_reflectance = np.sum(R, axis=2) > 0
```

**After:**
```python
# Define valid pixels based on reflectance data
# Use nansum to handle NaN values in masked water vapor bands
valid_reflectance = np.nansum(R, axis=2) > 0
```

---

#### Location 2: `plot_sample_spectra()` function
**Line 1141**

**Before:**
```python
valid_mask = np.sum(R, axis=2) > 0
```

**After:**
```python
valid_mask = np.nansum(R, axis=2) > 0
```

---

#### Location 3: `normalize_vnir_swir_transition()` function
**Line 678-679**

**Before:**
```python
# Find valid pixels (non-zero reflectance)
valid_pixels = np.sum(R, axis=2) > 0
```

**After:**
```python
# Find valid pixels (non-zero reflectance)
# Use nansum to handle NaN values in masked water vapor bands
valid_pixels = np.nansum(R, axis=2) > 0
```

---

### Fix 2: Handle Empty Valid Pixel Arrays

**Added:** Safety checks before computing statistics on potentially empty arrays

#### Location: `post_processing()` function
**Line 1433-1439**

**Before:**
```python
# NDVI statistics
valid_ndvi = ndvi[valid_mask]
print(f'    NDVI range: {valid_ndvi.min():.3f} to {valid_ndvi.max():.3f}')
print(f'    NDVI mean: {valid_ndvi.mean():.3f}')
```

**After:**
```python
# NDVI statistics
valid_ndvi = ndvi[valid_mask]
if len(valid_ndvi) > 0:
    print(f'    NDVI range: {valid_ndvi.min():.3f} to {valid_ndvi.max():.3f}')
    print(f'    NDVI mean: {valid_ndvi.mean():.3f}')
else:
    print(f'    WARNING: No valid pixels found - cannot compute NDVI statistics')
    print(f'    NDVI range: N/A')
    print(f'    NDVI mean: N/A')
```

---

#### Location: Statistics file writing
**Line 1455-1461**

**Before:**
```python
f.write(f'NDVI Statistics (valid pixels only):\n')
f.write(f'  Min: {valid_ndvi.min():.3f}\n')
f.write(f'  Max: {valid_ndvi.max():.3f}\n')
f.write(f'  Mean: {valid_ndvi.mean():.3f}\n')
f.write(f'  Std: {valid_ndvi.std():.3f}\n')
```

**After:**
```python
f.write(f'NDVI Statistics (valid pixels only):\n')
if len(valid_ndvi) > 0:
    f.write(f'  Min: {valid_ndvi.min():.3f}\n')
    f.write(f'  Max: {valid_ndvi.max():.3f}\n')
    f.write(f'  Mean: {valid_ndvi.mean():.3f}\n')
    f.write(f'  Std: {valid_ndvi.std():.3f}\n')
else:
    f.write(f'  WARNING: No valid pixels found - cannot compute statistics\n')
```

---

### Fix 3: Improved Wavelength Loading with Fallback

**Added:** Multiple fallback options for loading wavelength data when reflectance file already exists

#### Location: Main execution block (loading existing reflectance)
**Line 1647-1679**

**Before:**
```python
# Try to get wavelengths from HDR, or fall back to spectral_info.txt
if 'wavelength' in img.metadata:
    bands = np.array([float(w) for w in img.metadata['wavelength']])
else:
    # Load wavelengths from the spectral info file
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
```

**After:**
```python
# Try to get wavelengths from HDR, or fall back to spectral_info.txt
if 'wavelength' in img.metadata:
    bands = np.array([float(w) for w in img.metadata['wavelength']])
    print(f'    Loaded wavelengths from HDR metadata')
else:
    # Load wavelengths from the spectral info file (created by fix_envi_hdr_for_snap)
    # Try reflectance spectral info first, then fall back to preprocessed spectral info
    spectral_info_paths = [
        pathToReflectanceImage + '_spectral_info.txt',
        pathToRadianceImage + '_spectral_info.txt'
    ]

    bands = None
    for spectral_info_path in spectral_info_paths:
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
            break

    if bands is None:
        raise ValueError(
            f"No wavelength data found in HDR or spectral info files.\n"
            f"Tried: {spectral_info_paths}\n"
            f"The reflectance file may have been processed without wavelength field removal."
        )
```

**Improvement:** Now tries to load wavelengths from:
1. HDR metadata (if `wavelength` field exists)
2. Reflectance spectral_info.txt file
3. **Preprocessed (radiance) spectral_info.txt file** ← New fallback option

---

## 📊 Impact Summary

| Change | Lines Modified | Functions Affected |
|--------|----------------|-------------------|
| NaN-aware valid pixel detection | 3 locations | `visualize_valid_pixels()`, `plot_sample_spectra()`, `normalize_vnir_swir_transition()` |
| Empty array safety checks | 2 locations | `post_processing()` |
| Wavelength loading fallback | 1 location | Main execution block |

---

## 🎯 Expected Behavior After Fixes

### Before Fixes:
- ❌ Crashed with `ValueError: zero-size array to reduction operation`
- ❌ Reported "0 valid pixels" despite correct images
- ❌ Failed to load wavelengths when reflectance spectral_info.txt missing

### After Fixes:
- ✅ Completes Step 3 (Post-processing) successfully
- ✅ Correctly identifies valid pixels (typically 50-100% depending on cloud cover)
- ✅ Computes NDVI statistics for valid pixels
- ✅ Loads wavelengths from multiple fallback sources
- ✅ Generates all quicklooks and statistics files
- ✅ Handles edge cases gracefully with informative warnings

---

## 🔧 Technical Details

### Why `np.nansum()` Works:

```python
# Example with water vapor bands masked as NaN
pixel_spectrum = [0.05, 0.12, 0.15, np.nan, np.nan, 0.20, 0.18]

# Old method (fails):
np.sum(pixel_spectrum)  # Returns: nan
nan > 0  # Returns: False ❌

# New method (works):
np.nansum(pixel_spectrum)  # Returns: 0.70 (ignores NaN)
0.70 > 0  # Returns: True ✅
```

### Wavelength Loading Priority:

1. **HDR metadata** - Fastest, used if wavelength field wasn't removed
2. **Reflectance spectral_info.txt** - Created when wavelength field removed from reflectance HDR
3. **Preprocessed spectral_info.txt** - Fallback when reflectance file processed without HDR modification

---

## 📝 Testing Recommendations

Run the script with existing processed files:
```bash
python process_hyperion.py
```

Expected output in Step 3:
```
[6/7] Computing statistics...
    Image dimensions: 3311 x 891 pixels, 195 bands
    Valid pixels: XXXX / 2950101 (XX.X%)  # Should be > 0%
    Wavelength range: 426.8 - 2385.4 nm
    NDVI range: -0.XXX to 0.XXX  # Should show actual values
    NDVI mean: 0.XXX
```

---

## 🚀 Files Generated After Successful Run

- ✅ `EO1H2020342016359110KF_reflectance.img` - Surface reflectance image
- ✅ `EO1H2020342016359110KF_reflectance.hdr` - ENVI header
- ✅ `EO1H2020342016359110KF_NDVI.npy` - NDVI array
- ✅ `EO1H2020342016359110KF_statistics.txt` - Image statistics
- ✅ `EO1H2020342016359110KF_valid_pixels_mask.npy` - Valid pixel mask
- ✅ `quicklooks/EO1H2020342016359110KF_RGB.png` - RGB composite
- ✅ `quicklooks/EO1H2020342016359110KF_FalseColor.png` - NIR-R-G composite
- ✅ `quicklooks/EO1H2020342016359110KF_NDVI.png` - NDVI visualization
- ✅ `quicklooks/EO1H2020342016359110KF_spectra.png` - Sample spectra plot
- ✅ `quicklooks/EO1H2020342016359110KF_valid_pixels.png` - Pixel distribution map
- ✅ `quicklooks/EO1H2020342016359110KF_valid_pixels_stats.png` - Detailed statistics

---

## 📚 Additional Notes

### Water Vapor Band Masking
The script masks these spectral regions as they're unreliable even after atmospheric correction:
- **1350-1450 nm** - 1.4 μm water vapor absorption
- **1800-1950 nm** - 1.9 μm water vapor absorption

These bands are set to `np.nan` and excluded from analysis, but other bands remain valid.

### SNAP Compatibility
The script includes optional SNAP (Sentinel Application Platform) compatibility features:
- Removes wavelength fields from HDR files to avoid SNAP expression parser errors
- Saves spectral info to separate `.txt` files for reference
- Controlled by `snap_keep_wavelength` parameter (line 1586)

---

## 👤 Author Notes

These fixes address critical issues in the post-processing pipeline while maintaining compatibility with the existing preprocessing and atmospheric correction steps. The changes are backward-compatible and add defensive programming to handle edge cases gracefully.

**Key Principle:** Use NaN-aware functions (`np.nansum`, `np.nanmean`, etc.) when working with hyperspectral data that may have masked bands.
