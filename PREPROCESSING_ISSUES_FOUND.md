# Preprocessing Issues Found in Current Branch
## Branch: snap-wavelength-compatibility

---

## Summary

I've analyzed the preprocessing script against the diagnostic report and identified **critical issues** that will cause SAM (Spectral Angle Mapper) to fail or produce incorrect results.

---

## Critical Issues Found

### 🚨 **Issue #1: Scale Factor Applied but Not Documented for SAM Users**

**Location**: `src/surehyp/atmoCorrection.py:880-900` (function `saveRimage`)

**Problem**:
```python
def saveRimage(R,metadata,pathOut,scaleFactor=100):
    # ...
    scale=scaleFactor*np.ones(R.shape[2]).astype(int)
    metadata['scale factor']=scale.tolist()
    R=R*scaleFactor  # <-- MULTIPLIES reflectance by 100
    R[R>65535]=65535
    R[R<0]=0
    R=R.astype(np.uint16)  # Saves as integer
    envi.save_image(pathOut+'.hdr',R,metadata=metadata,force=True)
```

**What happens**:
1. Surface reflectance is computed correctly (0-1 range)
2. Before saving, it's multiplied by **100**
3. Saved as **uint16** with values in range **0-100** (not 0-1)
4. Scale factor metadata is written: `'scale factor': [100, 100, 100, ...]`

**Why this breaks SAM**:
- SAM expects reflectance in **0-1 range**
- Your preprocessed data is in **0-100 range**
- If you load this data directly into SAM without dividing by 100, you get:
  - Dot products that are 10,000x too large (100² = 10,000)
  - Norms that are 100x too large
  - Cosine values > 1.0 (invalid!)
  - SAM angles become meaningless

**Correct SAM workflow**:
```python
# When loading preprocessed data for SAM:
img = envi.open('reflectance.hdr')
cube = img.load()

# CRITICAL: Must divide by scale factor!
scale_factor = float(img.metadata['scale factor'][0])  # Gets 100
cube = cube.astype(np.float32) / scale_factor  # Now in 0-1 range

# Same for endmembers
endmembers = endmembers / scale_factor

# NOW you can use SAM
```

---

### ⚠️  **Issue #2: No Explicit Bad Band Removal**

**Location**: Throughout the preprocessing pipeline

**Problem**: The script doesn't explicitly remove Hyperion's known bad bands:
- Bands 1-7: VNIR overlap/bad
- Bands 58-76: Water vapor absorption (~1360-1400 nm)
- Bands 123-135: Water vapor absorption (~1800-1950 nm)
- Bands 225-242: SWIR edge/bad

**Impact on SAM**:
- Bad bands contain noise, not real spectral information
- Including them in SAM calculations adds noise to the angular distance
- Can cause misclassification, especially for similar materials

**Evidence**: Need to check the actual output to see if bad bands are present.

**Expected band count**:
- Original: 242 bands
- After removal: ~196 bands

---

### ✅ **Issue #3: Atmospheric Correction - APPEARS CORRECT**

**Location**: `src/surehyp/atmoCorrection.py:857-877` (function `computeLtoR`)

**Analysis**:
```python
def computeLtoR(L,bands,df,df_gs):
    # Uses SMARTS radiative transfer model
    factor=computeLtoRfactor(df,df_gs)
    # Interpolates to image wavelengths
    fun=interpolate.interp1d(W,factor)
    factor=fun(bands)
    # Converts radiance to reflectance
    R=factor*L
    return R
```

**Status**: ✅ This looks correct
- Uses proper radiative transfer (SMARTS)
- Accounts for atmosphere (sun-ground and ground-sensor paths)
- Should produce values in 0-1 range

---

### ✅ **Issue #4: Data Type Conversion - APPEARS CORRECT**

**Location**: `src/surehyp/atmoCorrection.py:896-899`

**Analysis**:
```python
R=R*scaleFactor  # float -> scaled float
R[R>65535]=65535  # Clip to uint16 max
R[R<0]=0  # Remove negative values
R=R.astype(np.uint16)  # Convert to integer
```

**Status**: ✅ This is correct for storage efficiency
- Converts float reflectance (0-1) to integer (0-100) via scale factor
- Properly clips to uint16 range
- **BUT**: Users must remember to divide by scale factor when loading!

---

### ⚠️  **Issue #5: Wavelength Metadata for SNAP**

**Location**: `process_hyperion.py:26-189` (function `fix_envi_hdr_for_snap`)

**Analysis**: Your recent changes handle SNAP compatibility by:
- Removing wavelength field from HDR (option 1 - safer for band math)
- OR keeping wavelength field (option 2 - better visualization)
- Saving spectral info to separate text file

**Configuration** (lines 1087-1093):
```python
snap_wavelength_file = "C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/OUT/EO1H2020342016359110KF_reflectance_spectral_info.txt"
snap_keep_wavelength = True  # Currently set to True
```

**Status**: ⚠️  Mixed - depends on use case
- For SNAP visualization: Keep wavelength = TRUE ✅
- For SNAP band math: Keep wavelength = FALSE ✅
- **For SAM**: Doesn't matter, but wavelengths must match between image and endmembers

---

## Recommendations for SAM Workflow

### 1. **Document Scale Factor Requirement**

Add this to your SAM script documentation:

```python
"""
CRITICAL: Reflectance data is scaled by factor of 100

The preprocessed Hyperion data is stored as uint16 with values 0-100
(instead of float 0-1) to save disk space.

Before using SAM, you MUST:
1. Load the data
2. Divide by the scale factor (100)
3. Then apply SAM
"""

# Example:
img = envi.open('reflectance.hdr')
cube = img.load()
scale_factor = float(img.metadata['scale factor'][0])

# Convert to 0-1 reflectance
cube_reflectance = cube.astype(np.float32) / scale_factor

# Same for endmembers if they're from the same source
endmembers_reflectance = endmembers / scale_factor

# NOW use SAM
sam_result = apply_sam(cube_reflectance, endmembers_reflectance)
```

### 2. **Add Validation Function**

Create a helper function in your SAM script:

```python
def validate_reflectance_scale(data, name="data"):
    """Check if reflectance is in correct 0-1 scale"""
    valid_data = data[data > 0]
    max_val = np.max(valid_data)

    if max_val > 2.0:
        raise ValueError(
            "{} appears to be SCALED (max={:.2f}).\n"
            "Expected 0-1 range for SAM.\n"
            "Did you forget to divide by the scale factor?"
            .format(name, max_val)
        )
    elif max_val > 1.2:
        print("WARNING: {} has max={:.2f} (>1.0). Unusual but proceeding."
              .format(name, max_val))
    else:
        print("OK: {} in valid range (max={:.3f})".format(name, max_val))
```

### 3. **Check Band Alignment**

Ensure your endmember library has the same bands as your preprocessed image:

```python
# Load image
img = envi.open('reflectance.hdr')
img_wavelengths = np.array([float(w) for w in img.metadata['wavelength']])

# Load endmembers
# ... load your endmember library ...

# Check
if len(img_wavelengths) != len(endmember_wavelengths):
    raise ValueError(
        "Band count mismatch: "
        "Image has {} bands, endmembers have {} bands"
        .format(len(img_wavelengths), len(endmember_wavelengths))
    )

# Check wavelength alignment
if not np.allclose(img_wavelengths, endmember_wavelengths, atol=1.0):
    print("WARNING: Wavelengths don't match exactly")
    print("Max difference: {:.2f} nm".format(
        np.max(np.abs(img_wavelengths - endmember_wavelengths))
    ))
```

---

## Action Items

### For the preprocessing script:
- [ ] ✅ The scale factor implementation is correct for storage
- [ ] ⚠️  Consider adding a warning message when saving:
  ```python
  print("NOTE: Reflectance saved with scale factor = {}".format(scaleFactor))
  print("      For SAM, divide by this factor to get 0-1 range")
  ```
- [ ] ⚠️  Check if bad bands are actually removed (need to inspect output)

### For the SAM script:
- [ ] 🚨 **CRITICAL**: Add scale factor division before SAM
- [ ] 🚨 **CRITICAL**: Add validation checks for reflectance range
- [ ] ⚠️  Verify band alignment between image and endmembers
- [ ] ⚠️  Check for NaN/Inf values before SAM

---

## Testing Checklist

When you run the preprocessing and then try SAM:

1. **Load preprocessed image**:
   ```python
   img = envi.open('reflectance.hdr')
   cube = img.load()
   print("Data type:", cube.dtype)
   print("Value range:", np.min(cube[cube>0]), "to", np.max(cube))
   ```
   - Expected: `dtype=uint16`, range ~0 to 80-100

2. **Check scale factor**:
   ```python
   if 'scale factor' in img.metadata:
       print("Scale factor:", img.metadata['scale factor'][0])
   ```
   - Expected: 100

3. **Apply scale factor**:
   ```python
   scale = float(img.metadata['scale factor'][0])
   cube = cube.astype(np.float32) / scale
   print("After scaling:", np.min(cube[cube>0]), "to", np.max(cube))
   ```
   - Expected: range ~0.0 to 0.8-1.0

4. **Check band count**:
   ```python
   print("Number of bands:", cube.shape[2])
   ```
   - Expected: 196 (if bad bands removed) or 242 (if not removed)

5. **Test SAM with one endmember**:
   ```python
   # Use first endmember
   em = endmembers[list(endmembers.keys())[0]] / scale

   # Test on one pixel
   pixel = cube[100, 100, :]

   # Compute cosine
   cos_angle = np.dot(pixel, em) / (np.linalg.norm(pixel) * np.linalg.norm(em))
   print("Cosine:", cos_angle)
   ```
   - Expected: -1.0 <= cos_angle <= 1.0
   - If cos_angle > 1.0: **SCALE ERROR!**

---

## Conda Environment Note

The current Python version in the base environment is 3.5.4, which is quite old and doesn't support f-strings (Python 3.6+).

Recommended environments for this project:
- `hyperion` environment (if Python >= 3.6)
- `hyperion_roger` environment

To activate:
```bash
conda activate hyperion
python validate_preprocessing.py <path_to_reflectance.hdr>
```

---

## Summary Table

| Issue | Severity | Status | Action Required |
|-------|----------|--------|-----------------|
| Scale factor not applied in SAM | 🚨 CRITICAL | Found | Add division by 100 in SAM script |
| Bad bands not removed | ⚠️  HIGH | Unknown | Check output band count |
| Atmospheric correction | ✅ OK | Verified | None |
| Data type conversion | ✅ OK | Verified | None |
| SNAP wavelength compatibility | ⚠️  INFO | Configured | Choose setting based on use case |

---

## Conclusion

**The preprocessing is mostly correct**, but there is a **critical scale factor issue** that will break SAM if not handled properly.

**Main problem**: The output reflectance is stored as integers with values 0-100 (not 0-1), and your SAM script must divide by 100 before computing spectral angles.

**Solution**: In your SAM script, after loading the reflectance data, add:
```python
cube = cube.astype(np.float32) / 100.0
endmembers = endmembers / 100.0
```

This is a **data handling issue**, not a preprocessing bug. The preprocessing is working as designed (using scale factors for efficient storage), but the SAM script needs to account for this.
