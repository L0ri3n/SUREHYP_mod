# Summary of Changes to Fix SAM Issues
## Branch: snap-wavelength-compatibility

---

## Overview

This document summarizes the changes made to address preprocessing issues that could cause Spectral Angle Mapper (SAM) to fail. The changes include:

1. ✅ Scale factor warning messages
2. ✅ Mean reflectance spectrum validation output
3. ✅ Helper functions for SAM data loading

---

## Changes Made

### 1. Enhanced `saveRimage()` Function
**File**: `src/surehyp/atmoCorrection.py` (lines 880-916)

**What changed**:
Added informative warning messages that print when saving reflectance data:

```python
print('\n' + '='*70)
print('IMPORTANT: Reflectance Scale Factor Information')
print('='*70)
print('Reflectance data is being saved with scale factor = {}'.format(scaleFactor))
print('Data range BEFORE scaling: {:.4f} to {:.4f}'.format(np.min(R[R>0]), np.max(R)))
# ... applies scaling ...
print('Data range AFTER scaling: {:.1f} to {:.1f}'.format(np.min(R[R>0]), np.max(R)))
print('NOTE: For Spectral Angle Mapper (SAM), you MUST divide by {} to get 0-1 range:'.format(scaleFactor))
print('      cube = cube.astype(np.float32) / {}'.format(scaleFactor))
print('      endmembers = endmembers / {}'.format(scaleFactor))
```

**Why this helps**:
- Clearly shows the user that data is being scaled
- Displays the value ranges before and after scaling
- Provides explicit instructions for SAM usage
- Prevents confusion about why SAM results are incorrect

---

### 2. Added Mean Reflectance Spectrum Plotting
**File**: `process_hyperion.py` (lines 885-975)

**What changed**:
Added new function `plot_mean_reflectance_spectrum()` that:
- Computes mean reflectance across all valid pixels for each band
- Creates a 2-panel plot showing:
  - **Panel 1**: Mean spectrum with ±1 std deviation envelope
  - **Panel 2**: Coefficient of variation (spectral variability)
- Highlights key spectral regions (red edge, NIR, water absorption)
- Validates spectral shape against expected vegetation signatures
- Prints diagnostic information about reflectance values

**Example output**:
```
    Spectral shape validation:
    - Mean reflectance range: 0.0234 to 0.5821
    - Expected for vegetation: ~0.03-0.05 (blue), ~0.4-0.6 (NIR)
    - Blue (~450nm): 0.0312
    - Red (~670nm): 0.0456
    - NIR (~850nm): 0.4823
    ✓ Vegetation red-edge signature detected (NIR > Red)
```

**Why this helps**:
- **Validates spectral shape**: Ensures the spectrum looks like actual reflectance
- **Detects scale errors**: If mean values are 10-100x too high, indicates scale factor issue
- **Checks preprocessing quality**: Unusual spectral shapes indicate preprocessing problems
- **Confirms SAM readiness**: Proper spectral shape means data is ready for SAM

**Output file**: `quicklooks/<image_name>_mean_spectrum.png`

---

### 3. Updated Post-Processing Function
**File**: `process_hyperion.py` (lines 978-1073)

**What changed**:
- Updated from 5 to 6 steps (added mean spectrum plotting as step 3)
- Changed all `print(f'...')` to `print('...'.format())` for Python 3.5 compatibility
- Changed title from "POST-PROCESSING & VISUALIZATION" to "POST-PROCESSING & VALIDATION"

**Why this helps**:
- Integrates validation into the standard workflow
- Ensures validation always runs when post-processing is enabled
- Compatible with older Python versions (3.5+)

---

### 4. Created SAM Data Helper Module
**File**: `sam_data_helper.py` (NEW FILE)

**What this provides**:
A complete helper module with functions for SAM users:

#### Function: `load_reflectance_for_sam(hdr_path)`
- Automatically loads reflectance data
- Detects and applies scale factor
- Validates data range
- Returns cube in correct 0-1 range for SAM

**Usage example**:
```python
from sam_data_helper import load_reflectance_for_sam

# Load and automatically scale data
cube, wavelengths, metadata = load_reflectance_for_sam('reflectance.hdr')

# Data is now in 0-1 range, ready for SAM!
```

#### Function: `validate_reflectance_scale(data, name)`
- Checks if data is in 0-1 range
- Warns if data appears scaled
- Prevents SAM errors before they happen

#### Function: `test_sam_computation(cube, endmember, pixel_coords)`
- Tests SAM on a single pixel
- Validates that cosine values are ≤ 1.0
- Detects scale mismatches between cube and endmembers
- Shows step-by-step computation for debugging

**Example output**:
```
Test SAM computation at pixel (100, 100):
  Dot product: 0.123456
  Norm (pixel): 0.654321
  Norm (endmember): 0.789012
  Cosine: 0.953210
  Angle: 17.23 degrees
  OK: Valid SAM angle
```

#### Function: `compare_band_alignment(cube_wavelengths, endmember_wavelengths)`
- Checks if image and endmember wavelengths match
- Detects band count mismatches
- Warns if wavelength differences exceed tolerance

---

## How to Use the Updated Workflow

### Step 1: Run Preprocessing (as before)
```bash
conda activate hyperion
python process_hyperion.py
```

**New output you'll see**:
- During atmospheric correction, when saving reflectance:
  ```
  ======================================================================
  IMPORTANT: Reflectance Scale Factor Information
  ======================================================================
  Reflectance data is being saved with scale factor = 100
  Data range BEFORE scaling: 0.0123 to 0.8456
  Data range AFTER scaling: 1.2 to 84.6

  NOTE: For Spectral Angle Mapper (SAM), you MUST divide by 100 to get 0-1 range:
        cube = cube.astype(np.float32) / 100
        endmembers = endmembers / 100
  ======================================================================
  ```

- During post-processing:
  ```
  [3/6] Plotting MEAN reflectance spectrum (validation)...
    Mean reflectance spectrum saved to: quicklooks/..._mean_spectrum.png

    Spectral shape validation:
    - Mean reflectance range: 0.0234 to 0.5821
    - Expected for vegetation: ~0.03-0.05 (blue), ~0.4-0.6 (NIR)
    - Blue (~450nm): 0.0312
    - Red (~670nm): 0.0456
    - NIR (~850nm): 0.4823
    ✓ Vegetation red-edge signature detected (NIR > Red)
  ```

### Step 2: Validate Output
Check the mean spectrum plot in the quicklooks folder:
```
OUT/quicklooks/<image_name>_mean_spectrum.png
```

**What to look for**:
- ✅ Mean reflectance in range 0.02-0.8 (vegetation typical)
- ✅ Red edge: rise from ~670nm (red) to ~850nm (NIR)
- ✅ NIR plateau: relatively flat 750-1300nm
- ✅ Water absorption dips at ~1400nm and ~1900nm
- ❌ If values > 1.0: Scale factor not applied correctly
- ❌ If spectrum is flat/noisy: Preprocessing may have failed

### Step 3: Use Helper Functions in Your SAM Script

**Option A: Use the helper module (RECOMMENDED)**
```python
import numpy as np
from sam_data_helper import load_reflectance_for_sam, test_sam_computation

# Load reflectance (automatically scaled to 0-1)
cube, wavelengths, metadata = load_reflectance_for_sam('path/to/reflectance.hdr')

# Load your endmembers
# ... your endmember loading code ...

# Make sure endmembers are also scaled if needed
# If endmembers come from the same preprocessed image, apply same scaling:
endmembers = endmembers / 100.0  # Only if endmembers are also scaled!

# Test on one pixel before running full SAM
test_sam_computation(cube, endmembers['Vegetation'], pixel_coords=(100, 100))

# If test passes, run full SAM
# ... your SAM code ...
```

**Option B: Manual scaling**
```python
import spectral.io.envi as envi
import numpy as np

# Load data
img = envi.open('reflectance.hdr')
cube = img.load()

# Check for scale factor and apply
if 'scale factor' in img.metadata:
    scale_factor = float(img.metadata['scale factor'][0])
    print('Applying scale factor: {}'.format(scale_factor))
    cube = cube.astype(np.float32) / scale_factor

# Quick validation
print('Cube range: {:.4f} to {:.4f}'.format(np.min(cube[cube>0]), np.max(cube)))
# Expected: 0.02 to 0.8 (approximately)

# Apply same scaling to endmembers
endmembers = endmembers / scale_factor

# Now run SAM
```

---

## Validation Checklist

Before running SAM, verify:

- [ ] Mean spectrum plot looks like expected reflectance (0-1 range)
- [ ] Red edge visible in mean spectrum (~670nm to ~850nm)
- [ ] Scale factor warning message appeared during preprocessing
- [ ] Loaded cube has values in 0-1 range (use helper functions)
- [ ] Endmembers have same scale as image (both 0-1)
- [ ] Band count matches between image and endmembers
- [ ] Test SAM computation gives valid cosine (≤ 1.0)

---

## Files Modified

1. **src/surehyp/atmoCorrection.py**
   - Modified `saveRimage()` function (lines 880-916)
   - Added scale factor warning messages

2. **process_hyperion.py**
   - Added `plot_mean_reflectance_spectrum()` function (lines 885-975)
   - Updated `post_processing()` function (lines 978-1073)
   - Changed to Python 3.5+ compatible formatting

3. **sam_data_helper.py** (NEW)
   - Complete helper module for SAM users
   - Automatic scale factor handling
   - Validation functions
   - Test functions for debugging

4. **PREPROCESSING_ISSUES_FOUND.md** (NEW)
   - Detailed analysis of issues
   - Documentation of findings
   - Testing guidelines

---

## Testing the Changes

### Test 1: Run preprocessing with existing data
```bash
conda activate hyperion
cd "C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/SUREHYP"
python process_hyperion.py
```

**Look for**:
- Scale factor warning messages during save
- Mean spectrum plot in quicklooks folder
- Validation output in console

### Test 2: Test helper functions
```bash
conda activate hyperion
python sam_data_helper.py "C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/OUT/EO1H2020342016359110KF_reflectance.hdr"
```

**Expected output**:
```
Loading reflectance data for SAM...
  Data shape: (rows, cols, bands)
  Data type (raw): uint16
  Value range (raw): 1 to 85
  Scale factor found: 100
  Applying scale factor...
  Value range (scaled): 0.0100 to 0.8500
  Validating Loaded cube: range 0.0100 to 0.8500
  OK: Loaded cube in valid range for SAM

======================================================================
Data successfully loaded and validated for SAM!
======================================================================
```

### Test 3: Validate mean spectrum
Open the generated plot:
```
OUT/quicklooks/EO1H2020342016359110KF_mean_spectrum.png
```

Check:
- Top panel: Smooth curve from ~0.02 to ~0.6
- Vegetation signature: Low in blue/red, high in NIR
- Bottom panel: CV shows variability across scene

---

## Conda Environment Note

**IMPORTANT**: Use the `hyperion` conda environment for all operations:

```bash
# The preprocessing script requires Python 3.6+ for f-strings
conda activate hyperion
python process_hyperion.py

# The helper script works with Python 3.5+
conda activate hyperion
python sam_data_helper.py <file.hdr>
```

**Do NOT use the base environment** (Python 3.5.4) as it doesn't support f-strings.

---

## Summary

### What was the problem?
- Reflectance data saved with scale factor = 100 (values 0-100 instead of 0-1)
- No warning to SAM users about the scale factor
- No easy way to validate if data is in correct range

### What did we fix?
1. ✅ Added clear warning messages when saving reflectance
2. ✅ Added mean spectrum validation plot
3. ✅ Created helper module for automatic scale factor handling
4. ✅ Added validation and test functions

### What should you do now?
1. Run preprocessing (will show new warnings and validation)
2. Check the mean spectrum plot
3. Use the helper module in your SAM script OR manually apply scale factor
4. Validate before running SAM

### Result
SAM will now work correctly because:
- You'll know data is scaled
- You have tools to automatically handle scaling
- You can validate data before running SAM
- Test functions catch scale errors early
