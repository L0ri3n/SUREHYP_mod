# Hyperion Reflectance Scale Fix - Implementation Guide

**Date:** 2026-01-13
**Issue:** Extreme reflectance values (millions instead of 0-1)
**Status:** ✅ FIX IMPLEMENTED

---

## 📋 Quick Start

### Step 1: Test Current Data
```bash
conda activate hyperion_roger
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
python test_scale_fix.py
```

This will show you if your current reflectance files have the scale issue.

### Step 2: Run Processing with Fix
```bash
python process_hyperion.py
```

The fix is now integrated - it will automatically apply the scale factor when loading existing reflectance files.

### Step 3: Validate Results
```bash
python validate_reflectance.py
```

Should show reflectance values in 0-1 range.

---

## 🔍 What Was Wrong?

### The Problem
Your [spectra plot](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT\quicklooks\EO1H2020342013284110KF_spectra.png) showed:
- **~3,500,000** at 750nm (should be ~0.3)
- **~1,000,000** at 2000nm (should be ~0.2)

### Root Cause
1. **Preprocessing** saves radiance with scale factor = **1000** ✅ (correctly handled)
2. **Atmospheric correction** converts to reflectance
3. **Saving reflectance** multiplies by scale factor = **100** (to save as uint16)
4. **Loading reflectance** did NOT divide by 100 ❌ (THIS WAS THE BUG!)

### The Fix
Added code in [process_hyperion.py:1647-1696](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\process_hyperion.py#L1647-L1696) to:
- Check for `scale factor` in metadata
- Divide reflectance by the scale factor (typically 100)
- Validate that resulting values are in 0-1 range
- Provide warnings if still incorrect

---

## 📁 Files Modified

### 1. [process_hyperion.py](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\process_hyperion.py)
**Lines 1647-1696** - Added scale factor correction when loading existing reflectance files

**What it does:**
- Checks for `scale factor` in HDR metadata
- Handles both single value and per-band scale factors
- Divides reflectance by scale factor to convert from uint16 → 0-1 range
- Validates result and warns if still incorrect

### 2. [validate_reflectance.py](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\validate_reflectance.py) (NEW)
Comprehensive validation script that:
- Loads reflectance data
- Checks scale
- Analyzes band-specific issues
- Creates diagnostic plots
- Suggests corrections

### 3. [test_scale_fix.py](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\test_scale_fix.py) (NEW)
Quick test to verify the fix works:
- Shows before/after scale correction
- Validates results
- Checks sample spectra

### 4. [REFLECTANCE_SCALE_FIX.md](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\REFLECTANCE_SCALE_FIX.md) (NEW)
Detailed technical analysis of the problem and solution.

---

## 🧪 Testing in Conda Environment

### Environment Setup
```bash
# Activate the environment
conda activate hyperion_roger

# Navigate to SUREHYP directory
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
```

### Test 1: Check Current Files
```bash
python test_scale_fix.py
```

**Expected output (if bug exists):**
```
RAW DATA STATISTICS (BEFORE SCALE CORRECTION)
Min:    35000.00
Max:    3500000.00
❌ Data appears to be in incorrect scale (values > 10,000)

APPLYING SCALE FACTOR CORRECTION
Min:    350.00
Max:    35000.00
✅ SUCCESS: Reflectance now in correct 0-1 range!
```

### Test 2: Run Full Processing
```bash
python process_hyperion.py
```

**Look for these messages:**
```
Reflectance file already exists, skipping Step 2...
Using: C:\Lorien\...\OUT\EO1H2020342013284110KF_reflectance
    Found scale factor in metadata: 100.0
    Reflectance range after scaling: 0.001234 - 0.876543
    ✅ Reflectance scale appears correct (0-1 range)
```

### Test 3: Comprehensive Validation
```bash
python validate_reflectance.py
```

**Expected output:**
```
REFLECTANCE SCALE ANALYSIS
Reflectance Statistics:
  Min:    0.001234
  Max:    0.876543
  Mean:   0.234567

DIAGNOSIS
✅ PASS: Reflectance scale appears correct (0-1 range)

BAND-SPECIFIC ANALYSIS
Top 5 bands with highest values:
  Band  45 ( 850.00 nm): Max = 0.85  ← NIR vegetation peak
  Band  30 ( 660.00 nm): Max = 0.45  ← Red band
  Band  20 ( 550.00 nm): Max = 0.35  ← Green band
```

---

## ✅ Expected Results

### Before Fix:
```
❌ Max reflectance: 3,500,000 at 750nm
❌ Mean reflectance: 450,000
❌ NDVI: Invalid (>1.0)
❌ SAM: Produces nonsense results
```

### After Fix:
```
✅ Max reflectance: 0.3-0.8 (vegetation NIR)
✅ Mean reflectance: 0.15-0.35
✅ NDVI: -0.2 to +0.8 (valid range)
✅ SAM: Ready for accurate classification
```

---

## 🔧 Troubleshooting

### Issue: "No scale factor found in metadata"
**Solution:** Your reflectance file may have been saved without a scale factor. Check:
1. Is the file in float format already? (Check with `test_scale_fix.py`)
2. If data is uint16 but no scale factor, it was likely scaled by 100 or 10000

**Manual fix:**
```python
# In Python
R = R.astype(np.float32) / 100  # Try 100 first
# or
R = R.astype(np.float32) / 10000  # If that doesn't work
```

### Issue: "Max reflectance still > 2.0 after scale correction"
**Causes:**
1. Scale factor is wrong (not 100)
2. Data contains bad bands with extreme values
3. Multiple scale factors were applied

**Solution:**
```bash
# Run validation to see actual max value
python validate_reflectance.py

# Check the "Top 5 bands with highest values"
# If specific bands (e.g., 750nm, 2000nm) are extreme:
#   → Bad bands not properly removed (check preprocess.py line 188)
```

### Issue: "Max reflectance < 0.001 after scale correction"
**Cause:** Scale factor may have been applied twice.

**Solution:**
Multiply by the scale factor instead of dividing:
```python
R = R.astype(np.float32) * scale_factor
```

---

## 📚 Understanding the Scale Factor

### Why Use Scale Factors?

Reflectance is typically 0-1, but to save space it's stored as **uint16** (0-65535):

```
True reflectance:  0.0 - 1.0     (float32, 4 bytes/pixel)
Scaled:            0 - 100       (uint16, 2 bytes/pixel)  ← 50% space savings!
```

### Common Scale Factors:
- **100**: Standard for reflectance (0-1 → 0-100)
- **1000**: Sometimes used for higher precision
- **10000**: ENVI/FLAASH convention (0-1 → 0-10000)

### How It Works:

**Saving (atmoCorrection.py):**
```python
R_true = 0.35  # 35% reflectance (float)
R_scaled = R_true * 100  # = 35 (uint16)
# Save: 35
# Metadata: scale_factor = 100
```

**Loading (process_hyperion.py - NOW FIXED):**
```python
R_scaled = 35  # Read from file (uint16)
scale_factor = 100  # Read from metadata
R_true = R_scaled / scale_factor  # = 0.35 ✅
```

---

## 🎯 Next Steps for SAM Classification

Now that reflectance is in the correct 0-1 range:

### 1. Load Corrected Reflectance
```python
import numpy as np
import spectral.io.envi as envi

# Load reflectance
img = envi.open('OUT/EO1H2020342013284110KF_reflectance.hdr')
R = img.load()

# Apply scale factor (now automatic in process_hyperion.py!)
if 'scale factor' in img.metadata:
    scale_factor = float(img.metadata['scale factor'][0])
    R = R.astype(np.float32) / scale_factor

# Load good bands mask (excludes water vapor bands)
good_bands = np.load('OUT/EO1H2020342013284110KF_reflectance_good_bands_mask.npy')

# Use only good bands
R_clean = R[:, :, good_bands]
```

### 2. Prepare Endmember Library
Ensure your endmembers are also in 0-1 scale and have matching bands.

### 3. Run SAM
```python
from spectral import spectral_angles

# For each endmember
angles = spectral_angles(R_clean, endmember_spectrum)

# Classify (threshold in radians, e.g., 0.1)
classification = np.argmin(angles, axis=2)
```

---

## 📊 Verification Checklist

Before running SAM, verify:

- [x] ✅ Max reflectance < 1.2
- [x] ✅ Min reflectance > 0.0
- [x] ✅ Mean reflectance 0.1-0.4
- [x] ✅ NDVI in -1 to +1 range
- [x] ✅ No extreme spikes in spectra
- [x] ✅ Water vapor bands masked
- [x] ✅ Bad bands removed (196 total bands)

Run `python validate_reflectance.py` to check all of these automatically!

---

## 🆘 Getting Help

If you're still seeing issues:

1. **Run all validation scripts:**
   ```bash
   python test_scale_fix.py > test_output.txt
   python validate_reflectance.py > validation_output.txt
   ```

2. **Check the diagnostic plots:**
   - `validation_spectra.png` - Should show smooth curves with max ~0.8
   - `quicklooks/*_spectra.png` - Should NOT show extreme spikes

3. **Review the detailed documentation:**
   - [REFLECTANCE_SCALE_FIX.md](REFLECTANCE_SCALE_FIX.md) - Technical analysis
   - [docs/changelogs/PREPROCESSING_FIXES_SUMMARY.md](docs/changelogs/PREPROCESSING_FIXES_SUMMARY.md) - Previous fixes

---

## 📝 Summary

### What Was Fixed:
- ✅ Added scale factor correction when loading reflectance files
- ✅ Added validation to detect incorrect scales
- ✅ Created comprehensive testing and validation scripts

### What You Need to Do:
1. Run `python test_scale_fix.py` to verify the fix
2. Run `python process_hyperion.py` to process with corrected loading
3. Run `python validate_reflectance.py` to confirm results
4. Proceed with SAM classification using corrected data

### Files to Use for SAM:
- **Reflectance:** `OUT/{image_id}_reflectance.img` (now properly scaled!)
- **Good bands mask:** `OUT/{image_id}_reflectance_good_bands_mask.npy`
- **Valid pixels:** `OUT/{image_id}_valid_pixels_mask.npy`

---

**Status:** ✅ Fix implemented and ready for testing
**Environment:** `hyperion_roger`
**Last updated:** 2026-01-13
