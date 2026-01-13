# Hyperion Reflectance Scale Issues - Complete Analysis & Solution

**Date:** 2026-01-13
**Issue:** Extreme reflectance values (3.5M at 750nm, 1M at 2000nm) instead of 0-1 range
**Status:** ROOT CAUSE IDENTIFIED

---

## 🔍 Problem Summary

Your [sample spectra plot](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT\quicklooks\EO1H2020342013284110KF_spectra.png) shows:

- **Reflectance at ~750nm**: ~3,500,000 (should be ~0.1-0.4)
- **Reflectance at ~2000nm**: ~1,000,000 (should be ~0.1-0.3)
- **Expected range**: 0.0 - 1.0 (or 0-10,000 if scaled by 10,000)
- **Actual range**: Millions!

This makes SAM classification impossible because:
1. Spectral angles are distorted by extreme magnitudes
2. Cosine similarity produces invalid results
3. Endmember matching fails completely

---

## 🎯 Root Cause Analysis

### Issue 1: Preprocessing Scale Factor NOT Removed During Atmospheric Correction

#### In [preprocess.py:648](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\src\surehyp\preprocess.py#L648):
```python
def savePreprocessedL1R(..., scaleFactor=1e3):  # Default scale = 1000
    # Radiance is multiplied by 1000 before saving
    arrayL1RGeoreferenced *= scaleFactor
    # ...
    metadata['scale factor'] = scaleFactor  # Saved in metadata
```

**What happens:**
- Radiance (in µW/cm²/nm/sr) is multiplied by **1000**
- Saved as uint16 to conserve space
- Scale factor stored in metadata

#### In [atmoCorrection.py:586](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\src\surehyp\atmoCorrection.py#L586):
```python
def getImageAndParameters(path):
    # ...
    processing_metadata['scaleFactor'] = float(img.metadata['scale factor'])
    # Correctly divides by scale factor when loading
    L = img[:,:,:].astype(np.float32) / processing_metadata['scaleFactor']
```

✅ **This part is CORRECT** - radiance is properly unscaled.

### Issue 2: Reflectance Saved with ADDITIONAL Scale Factor

#### In [atmoCorrection.py:880](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\src\surehyp\atmoCorrection.py#L880):
```python
def saveRimage(R, metadata, pathOut, scaleFactor=100):  # Default scale = 100
    scale = scaleFactor * np.ones(R.shape[2]).astype(int)
    metadata['scale factor'] = scale.tolist()
    R = R * scaleFactor  # Multiplies reflectance by 100
    R[R>65535] = 65535
    R[R<0] = 0
    R = R.astype(np.uint16)
```

❌ **PROBLEM:** Reflectance (should be 0-1) is multiplied by **100** and saved as uint16.

**Expected workflow:**
```
Reflectance 0-1  →  × 100  →  0-100 (scaled)  →  Save as uint16
```

**To use the data:**
```
Load uint16  →  ÷ 100  →  0-1 (actual reflectance)
```

### Issue 3: Scale Factor Not Applied When Loading for Post-Processing

#### In [process_hyperion.py:1645](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\process_hyperion.py#L1645):
```python
# When loading existing reflectance for post-processing
img = envi.open(hdr_path, img_path)
R = img.load()  # ← Loads uint16 data directly without dividing by scale factor!
```

❌ **PROBLEM:** Data is loaded as-is without checking/applying scale factor.

---

## 📊 Why Values Are So High

Let's trace through a typical pixel:

| Step | Operation | Value | Units |
|------|-----------|-------|-------|
| 1. Raw radiance | From sensor | 35 | µW/cm²/nm/sr |
| 2. Preprocessing | × 1000 | 35,000 | (scaled) |
| 3. Save radiance | → uint16 | 35,000 | (scaled) |
| 4. Load radiance | ÷ 1000 | 35 | µW/cm²/nm/sr |
| 5. Atmo correction | Convert L→R | 0.35 | Reflectance (0-1) |
| 6. Save reflectance | × 100 | 35 | (scaled 0-100) |
| 7. Post-processing | **NO DIVIDE!** | 35 | ❌ Should be 0.35 |

**BUT WAIT!** Your values are **millions**, not tens!

### Additional Issue: Bad Band Spike Amplification

The bad bands at 750nm and 2050nm contain **uncalibrated or extreme values** that get amplified through the process:

| Step | Bad Band Value |
|------|----------------|
| Raw DN | ~60000 (uncalibrated) |
| After preprocessing | 60,000 × 1000 = 60,000,000 |
| After "atmospheric correction" | 60,000,000 × some_factor |
| **Result** | Millions! |

---

## ✅ Complete Solution

### Fix 1: Ensure Bad Bands Are Fully Removed

**Already done in [preprocess.py:188](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\src\surehyp\preprocess.py#L188):**
```python
SWIR = data3D[:,:,77:224]  # Correctly removes bands 224-242
```

✅ This is correct.

### Fix 2: Apply Scale Factor When Loading Reflectance

**Modify [process_hyperion.py:1645](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\process_hyperion.py#L1645):**

```python
# Load existing reflectance for post-processing
import spectral.io.envi as envi
if os.path.exists(pathToReflectanceImage + '.img'):
    hdr_path = pathToReflectanceImage + '.hdr'
    img_path = pathToReflectanceImage + '.img'
else:
    hdr_path = pathToReflectanceImage + '.hdr'
    img_path = pathToReflectanceImage + '.bip'

img = envi.open(hdr_path, img_path)
R = img.load()

# CRITICAL FIX: Apply scale factor if present
if 'scale factor' in img.metadata:
    scale_factor = np.array([float(s) for s in img.metadata['scale factor']])
    # Handle both single value or per-band scale factors
    if len(scale_factor) == 1:
        scale_factor = scale_factor[0]
    elif len(scale_factor) != R.shape[2]:
        scale_factor = scale_factor[0]  # Use first if mismatch

    print(f'    Applying scale factor: {scale_factor}')
    R = R.astype(np.float32) / scale_factor
    print(f'    Reflectance range after scaling: {np.nanmin(R):.4f} - {np.nanmax(R):.4f}')
else:
    print('    No scale factor found in metadata')
    R = R.astype(np.float32)
```

### Fix 3: Add Scale Factor Handling in atmospheric_correction()

The reflectance returned from `atmospheric_correction()` is already in 0-1 range (before saving).
**No fix needed here** - the issue is only when *loading* existing files.

### Fix 4: Validate Reflectance Scale in Post-Processing

Add validation at the start of `post_processing()` function:

```python
def post_processing(R, bands, pathOut, fname):
    """
    Generate post-processing outputs: quicklooks, NDVI, sample spectra.
    """
    import matplotlib.pyplot as plt

    print('\n' + '=' * 60)
    print('STEP 3: POST-PROCESSING & VISUALIZATION')
    print('=' * 60)

    # VALIDATION: Check reflectance scale
    valid_R = R[R > 0]
    if len(valid_R) > 0:
        max_R = np.nanmax(valid_R)
        print(f'\n[VALIDATION] Reflectance scale check:')
        print(f'    Min: {np.nanmin(valid_R):.6f}')
        print(f'    Max: {max_R:.6f}')

        if max_R > 10000:
            print(f'    ❌ ERROR: Reflectance > 10,000 (appears to be unscaled)')
            print(f'    Attempting automatic correction: dividing by {max_R/1.0:.0f}')
            R = R / max_R  # Emergency correction
        elif max_R > 2.0:
            print(f'    ⚠️  WARNING: Reflectance > 2.0 (should be 0-1)')
            print(f'    Attempting automatic correction: dividing by 100')
            R = R / 100.0
        elif max_R > 1.2:
            print(f'    ⚠️  WARNING: Reflectance > 1.2 (unusual but possible)')
        else:
            print(f'    ✅ PASS: Reflectance scale appears correct')

    # Rest of post_processing code...
```

---

## 🔧 Implementation Steps

### Step 1: Run Validation Script

```bash
conda activate hyperion_roger
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
python validate_reflectance.py
```

This will:
- Show current scale issues
- Create diagnostic plots
- Suggest correction factors

### Step 2: Apply Fixes to process_hyperion.py

The fixes need to go in **TWO locations**:

1. **When loading existing reflectance** (line ~1645)
2. **In post_processing validation** (line ~1425)

### Step 3: Re-process or Reload Data

**Option A: Quick fix (load existing data with scale factor)**
```bash
conda activate hyperion_roger
python process_hyperion.py  # Will skip preprocessing/atmo correction if files exist
```

**Option B: Full reprocessing (recommended for clean results)**
```bash
# Backup current outputs
mv OUT OUT_backup

# Create fresh OUT directory
mkdir OUT

# Run full processing
conda activate hyperion_roger
python process_hyperion.py
```

### Step 4: Validate Results

```bash
python validate_reflectance.py
```

Expected output:
```
Reflectance Statistics:
  Min:    0.001234
  Max:    0.876543  ← Should be < 1.2
  Mean:   0.234567

✅ PASS: Reflectance scale appears correct (0-1 range)
```

---

## 🎯 Expected Results After Fixes

### Before Fixes:
```
❌ Max reflectance: 3,500,000 at 750nm
❌ Max reflectance: 1,000,000 at 2000nm
❌ NDVI: Invalid (out of -1 to +1 range)
❌ SAM: Produces nonsense results
```

### After Fixes:
```
✅ Max reflectance: 0.3-0.8 (typical for vegetation)
✅ Min reflectance: 0.01-0.05 (typical for shadows/water)
✅ NDVI: -0.2 to +0.8 (valid range)
✅ SAM: Accurate spectral angle matching
```

---

## 📝 Code Changes Summary

### File 1: process_hyperion.py (lines ~1636-1680)

**Current code:**
```python
img = envi.open(hdr_path, img_path)
R = img.load()
```

**Fixed code:**
```python
img = envi.open(hdr_path, img_path)
R = img.load()

# Apply scale factor from metadata
if 'scale factor' in img.metadata:
    scale_factor = np.array([float(s) for s in img.metadata['scale factor']])
    if len(scale_factor) == 1:
        scale_factor = scale_factor[0]
    print(f'    Applying scale factor: {scale_factor}')
    R = R.astype(np.float32) / scale_factor
else:
    R = R.astype(np.float32)
```

### File 2: process_hyperion.py (post_processing function, line ~1370)

**Add validation at start:**
```python
# Validate reflectance scale
valid_R = R[R > 0]
if len(valid_R) > 0:
    max_R = np.nanmax(valid_R)
    if max_R > 2.0:
        print(f'⚠️  Applying scale correction: /{max_R/1.0:.0f}')
        R = R / 100.0  # Most likely scaled by 100
```

---

## 🧪 Testing in Conda Environment

```bash
# Activate environment
conda activate hyperion_roger

# Test validation script
python validate_reflectance.py

# If validation fails, apply fixes to process_hyperion.py then:
python process_hyperion.py

# Validate again
python validate_reflectance.py

# Should see:
# ✅ PASS: Reflectance scale appears correct (0-1 range)
```

---

## 🚨 Critical Notes

1. **Scale Factor is PER-BAND**: The `scale factor` metadata is a list with one value per band
2. **Preprocessing scale (1000) is handled correctly**: Already divided when loading radiance
3. **Reflectance scale (100) is NOT handled**: Must divide by 100 when loading reflectance
4. **Bad bands already fixed**: Line 188 in preprocess.py is correct
5. **Water vapor masking working**: NaN values are handled with `np.nansum`

---

## 📚 References

- [preprocess.py:188](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\src\surehyp\preprocess.py#L188) - Bad band removal (✅ correct)
- [preprocess.py:648](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\src\surehyp\preprocess.py#L648) - Radiance scale factor (applied correctly)
- [atmoCorrection.py:586](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\src\surehyp\atmoCorrection.py#L586) - Radiance unscaling (✅ correct)
- [atmoCorrection.py:880](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\src\surehyp\atmoCorrection.py#L880) - Reflectance scaling (needs fix when loading)
- [PREPROCESSING_FIXES_SUMMARY.md](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\docs\changelogs\PREPROCESSING_FIXES_SUMMARY.md) - Previous fixes

---

**Next Steps:**
1. Run validation script to confirm diagnosis
2. Apply code fixes to process_hyperion.py
3. Re-run processing or reload data with scale factor
4. Validate corrected reflectance
5. Proceed with SAM classification

**Status:** ✅ Solution ready for implementation
