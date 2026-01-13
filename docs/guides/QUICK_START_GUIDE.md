# Quick Start Guide - Testing Hyperion Preprocessing Fixes

## What Was Fixed?

✅ **Critical Bug**: Bands 224-242 (uncalibrated SWIR) now properly removed
✅ **Water vapor masking**: Bands at ~1400nm and ~1900nm now masked
✅ **Outlier clipping**: Extreme spikes (65,000+) now clipped to normal values
✅ **VNIR-SWIR smoothing**: Detector discontinuity at ~920nm now smoothed
✅ **Wavelength field error**: Fixed KeyError by keeping wavelengths in radiance file

## Quick Test (5 minutes)

### 1. Run the processing
```bash
conda activate hyperion
cd c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP
python process_hyperion.py
```

### 2. Check the console output
Look for this section:
```
APPLYING POST-CORRECTION PREPROCESSING FIXES
------------------------------------------------------------
[Fix 1/3] Clipping reflectance outliers...
[Fix 2/3] Masking water vapor absorption bands...
[Fix 3/3] Smoothing VNIR-SWIR detector transition...
POST-CORRECTION FIXES COMPLETE!
```

### 3. Check the spectra plot
Open: `OUT/quicklooks/EO1H2020342016359110KF_spectra.png`

**Before**: Extreme spikes at 750nm and 2050nm (~65,000)
**After**: Smooth curves with reasonable reflectance (< 1.0 or < 10,000)

### 4. Verify reflectance range
```python
import numpy as np
import spectral.io.envi as envi

img = envi.open('OUT/EO1H2020342016359110KF_reflectance.hdr',
                'OUT/EO1H2020342016359110KF_reflectance.img')
R = img.load()

print(f"Min: {np.nanmin(R):.4f}")
print(f"Max: {np.nanmax(R):.4f}")
print(f"Mean: {np.nanmean(R):.4f}")
```

**Expected**:
- Max should be < 1.0 (or < 10,000 if scaled)
- No values like 65,000

## Files Modified

1. **src/surehyp/preprocess.py** (line 188)
   - Fixed SWIR band range from `77:223` to `77:224`

2. **process_hyperion.py** (lines 484-631, 773-793)
   - Added 3 new functions for post-correction fixes
   - Integrated into atmospheric_correction workflow

## Next Steps for SAM Classification

### Load clean reflectance with good bands only:
```python
# Load reflectance
img = envi.open('OUT/EO1H2020342016359110KF_reflectance.hdr',
                'OUT/EO1H2020342016359110KF_reflectance.img')
R = img.load()

# Load good bands mask
good_bands = np.load('OUT/EO1H2020342016359110KF_reflectance_good_bands_mask.npy')

# Use only good bands for SAM
R_clean = R[:, :, good_bands]
wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
wavelengths_clean = wavelengths[good_bands]

print(f"Total bands: {R.shape[2]}")
print(f"Good bands: {np.sum(good_bands)}")
print(f"Masked bands: {np.sum(~good_bands)} (water vapor)")
```

### Your SAM results should now:
- ✅ Have no extreme angle distortions
- ✅ More accurate spectral matching
- ✅ Better discrimination between materials

## Troubleshooting

**Q: KeyError: 'wavelength' during atmospheric correction?**
A: Old radiance file might have wavelengths removed. Solution:
```bash
rm OUT/EO1H2020342016359110KF_preprocessed.*
python process_hyperion.py
```
The new code keeps wavelengths in radiance files to prevent this.

**Q: Processing stops with an error?**
A: Check that scipy is installed: `conda list scipy`

**Q: Still seeing high reflectance values?**
A: If values are ~10,000, that's normal (scaled reflectance). Only worry if > 10,000.

**Q: No bands masked for water vapor?**
A: Check your wavelength range - if it doesn't include 1400nm or 1900nm regions, this is normal.

## Full Documentation

See [PREPROCESSING_FIXES_SUMMARY.md](PREPROCESSING_FIXES_SUMMARY.md) for complete details.

---

**Ready to test?** Just run: `python process_hyperion.py`
