# Instructions for Reprocessing with Reflectance Fixes

## What Was Fixed

I've implemented **TWO critical fixes**:

### Fix 1: Enhanced Bad Band Masking (in `clip_reflectance_outliers`)
**Location:** [process_hyperion.py:600-703](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\process_hyperion.py#L600-L703)

**What it does:**
- Identifies bands with extreme max values (> 2.0 unscaled or > 20000 scaled)
- **Masks entire bands** that show bad calibration
- Clips remaining pixel outliers
- Validates final result

**Why this fixes your problem:**
- Your spectra showed peaks at ~750nm (~3.5M) and ~2000nm (~1M)
- These are **bad bands** that slipped through preprocessing
- Now they will be **completely masked** (set to NaN)
- Result: Clean spectra like your Jarosite reference

### Fix 2: Scale Factor Correction When Loading
**Location:** [process_hyperion.py:1647-1696](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\process_hyperion.py#L1647-L1696)

**What it does:**
- Automatically divides by scale factor (100) when loading existing reflectance files
- Converts uint16 → float32 properly

---

## Step-by-Step Reprocessing

### Prerequisites
```bash
# Activate conda environment
conda activate hyperion_roger

# Navigate to SUREHYP directory
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
```

### Option 1: Force Complete Reprocessing (RECOMMENDED)

This will reprocess atmospheric correction with the new fixes:

```bash
# Step 1: Delete existing reflectance files to force reprocessing
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT"
del EO1H2020342013284110KF_reflectance.*

# Step 2: Also delete derived products
del EO1H2020342013284110KF_NDVI.npy
del EO1H2020342013284110KF_statistics.txt
del EO1H2020342013284110KF_valid_pixels_mask.npy
del quicklooks\EO1H2020342013284110KF_*.*

# Step 3: Go back to SUREHYP directory
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"

# Step 4: Run processing
python process_hyperion.py
```

**What will happen:**
1. Script will skip preprocessing (radiance file exists) ✅
2. **Script will RERUN atmospheric correction** with new fixes ✅
3. The enhanced `clip_reflectance_outliers` will mask bad bands
4. Reflectance will be saved with proper values
5. Post-processing will create clean spectra plots

**Expected output:**
```
STEP 2: ATMOSPHERIC CORRECTION
============================================================
Loading existing preprocessed radiance for atmospheric correction...
...
Computing radiance to reflectance conversion...

------------------------------------------------------------
APPLYING POST-CORRECTION PREPROCESSING FIXES
------------------------------------------------------------

[Fix 1/3] Clipping reflectance outliers...
    Detected unscaled reflectance (median max: 0.4523)
    ⚠️  WARNING: Found 2 bands with extreme max values (bad calibration)
    These bands will be masked (set to NaN):
      Band 33: max = 3500000.00  ← Bad band at 752nm
      Band 195: max = 1000000.00  ← Bad band at 2385nm
    Clipping 15234 pixel values to max = 1.0000
    ✅ After clipping: max = 1.0000, mean = 0.2456

[Fix 2/3] Masking water vapor absorption bands...
    Masked 15 bands in range 1350-1450 nm
    Masked 16 bands in range 1800-1950 nm
    Result: 163 good bands, 31 masked bands

[Fix 3/3] Smoothing VNIR-SWIR detector transition...
    Smoothed 2450123 valid pixels in VNIR-SWIR transition

Saving reflectance with scale factor 100...
```

### Option 2: Quick Test (Load Existing with Scale Fix)

If you want to test the scale factor fix without reprocessing:

```bash
python process_hyperion.py
```

This will:
1. Skip preprocessing (exists)
2. Skip atmospheric correction (exists)
3. Load existing reflectance **with scale factor correction**
4. Run post-processing with corrected values

**BUT**: The source data still has bad bands, so the scale fix alone won't solve the problem completely!

---

## Validation After Reprocessing

### Step 1: Check Raw Values
```bash
python check_reflectance_values.py
```

**Expected output after Fix 1:**
```
--- RAW STORED VALUES (uint16) ---
Min (>0): 1
Max (>0): 100  ← Should be ~100 (not 3.5 million!)
Mean (>0): 25.34

--- AFTER SCALE CORRECTION (÷100) ---
Min (>0): 0.000100
Max (>0): 1.000000  ← Perfect!
Mean (>0): 0.253400

--- TOP 5 BANDS WITH HIGHEST RAW VALUES ---
Band  45 ( 874.53 nm): raw=    100, scaled=1.0000  ← NIR peak, normal
Band  44 ( 864.35 nm): raw=     95, scaled=0.9500
Band  43 ( 854.18 nm): raw=     92, scaled=0.9200
Band  46 ( 884.70 nm): raw=     89, scaled=0.8900
Band  47 ( 894.88 nm): raw=     87, scaled=0.8700

--- CHECKING SPECIFIC WAVELENGTHS ---
~750nm (band 33, 752.43nm): raw=0, scaled=nan  ← Masked!
~1400nm (band 98, 1406.84nm): raw=0, scaled=nan  ← Masked!
~2000nm (band 195, 2385.40nm): raw=0, scaled=nan  ← Masked!

✅ SUCCESS: Max scaled value 1.0000 in reasonable range!
```

### Step 2: Check Spectra Plots
```bash
# Look at the new spectra plot
explorer.exe "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT\quicklooks\EO1H2020342013284110KF_spectra.png"
```

**What you should see:**
- ✅ NO spike at ~750nm (masked as NaN, will show gap or zero)
- ✅ Gap at ~1400nm (water vapor, masked)
- ✅ Gap at ~1800-1950nm (water vapor, masked)
- ✅ NO spike at ~2000nm (masked)
- ✅ Smooth spectral curve with max ~0.8 (similar to Jarosite reference)

### Step 3: Comprehensive Validation
```bash
python validate_reflectance.py
```

**Expected output:**
```
REFLECTANCE SCALE ANALYSIS
==============================================================================
Valid pixels: 2,891,141 / 2,890,441 (99.9%)

Reflectance Statistics (valid pixels only):
  Min:    0.001234
  Max:    0.987654  ← Good!
  Mean:   0.234567
  Median: 0.212345
  Std:    0.123456

DIAGNOSIS
==============================================================================
✅ PASS: Reflectance scale appears correct (0-1 range)

BAND-SPECIFIC ANALYSIS
==============================================================================
Top 5 bands with highest values:
  Band  45 ( 874.53 nm): Max = 0.99  ← NIR vegetation, normal
  Band  44 ( 864.35 nm): Max = 0.95
  Band  43 ( 854.18 nm): Max = 0.92
  Band  46 ( 884.70 nm): Max = 0.89
  Band  47 ( 894.88 nm): Max = 0.87

Near 750nm (VNIR-SWIR transition):
  Band index: 33, Wavelength: 752.43 nm
  Max value: nan  ← Properly masked!

Near 2050nm (SWIR end region):
  Band index: 195, Wavelength: 2385.40 nm
  Max value: nan  ← Properly masked!
```

---

## Troubleshooting

### Issue: "Still seeing extreme values after reprocessing"

**Check:**
1. Did you delete the old reflectance file before reprocessing?
   ```bash
   ls -l "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT\EO1H2020342013284110KF_reflectance.*"
   ```
   If files exist from before the fix, delete them!

2. Check the console output during processing - look for:
   ```
   ⚠️  WARNING: Found 2 bands with extreme max values (bad calibration)
   ```

3. If you don't see this message, the fix may not be active.

### Issue: "No data after masking bad bands"

**Symptoms:**
```
ERROR: No valid data remaining after bad band removal!
```

**Cause:** Too many bands were masked

**Solution:** Adjust the threshold in `clip_reflectance_outliers`:
```python
# Line 631: Make threshold less aggressive
bad_band_threshold_unscaled = 5.0  # Was 2.0
```

### Issue: "Spectra still look noisy"

**This is normal!** Hyperion data is inherently noisy. The key checks:
- Max values should be < 1.5 ✅
- No extreme spikes (millions) ✅
- Gaps at water vapor bands ✅
- General spectral shape should match reference materials

---

## For SAM Classification

After successful reprocessing:

### 1. Load Corrected Reflectance
```python
import numpy as np
import spectral.io.envi as envi

# Load reflectance
img = envi.open('OUT/EO1H2020342013284110KF_reflectance.hdr')
R_raw = img.load()

# Apply scale factor (automatically handled in process_hyperion.py now)
scale_factor = float(img.metadata['scale factor'][0])
R = R_raw.astype(np.float32) / scale_factor

# Load good bands mask
good_bands = np.load('OUT/EO1H2020342013284110KF_reflectance_good_bands_mask.npy')

# Filter out NaN bands (bad bands masked during clipping)
nan_bands = ~np.any(np.isnan(R), axis=(0,1))
final_good_bands = good_bands & nan_bands

print(f'Total bands: {R.shape[2]}')
print(f'Good bands (after masking): {np.sum(final_good_bands)}')

# Use only clean bands
R_clean = R[:, :, final_good_bands]
```

### 2. Prepare Endmember Library

**CRITICAL:** Your endmember library (e.g., Jarosite spectrum) must:
- Be in 0-1 range (like your reference graph)
- Have matching wavelengths to your good bands
- Exclude water vapor bands (1350-1450nm, 1800-1950nm)
- Exclude the bad bands that were masked

```python
# Get wavelengths of good bands
wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
wavelengths_clean = wavelengths[final_good_bands]

print(f'Wavelength range: {wavelengths_clean.min():.1f} - {wavelengths_clean.max():.1f} nm')
print(f'Number of good bands: {len(wavelengths_clean)}')

# Resample your endmember library to match these wavelengths
# (You may need interpolation or band matching)
```

### 3. Run SAM

```python
from spectral import spectral_angles

# Flatten spatial dimensions
pixels = R_clean.reshape(-1, R_clean.shape[2])

# For each endmember
angles = spectral_angles(pixels, endmember_spectrum_resampled)

# Reshape back
angle_map = angles.reshape(R_clean.shape[0], R_clean.shape[1])

# Classify (threshold in radians)
threshold = 0.1  # Adjust based on results
classification = angle_map < threshold
```

---

## Summary

### What to do NOW:
1. ✅ Delete old reflectance files
2. ✅ Run `python process_hyperion.py`
3. ✅ Run `python check_reflectance_values.py`
4. ✅ Run `python validate_reflectance.py`
5. ✅ Check spectra plots in `OUT/quicklooks/`
6. ✅ If all good, proceed with SAM classification

### Expected final result:
- Reflectance: 0.0 - 1.0 ✅
- Bad bands: Masked as NaN ✅
- Spectra: Clean curves like Jarosite reference ✅
- Ready for SAM ✅

---

**Good luck with your processing!**
**Test environment:** `hyperion_roger`
**Last updated:** 2026-01-13
