# Hyperion Preprocessing Diagnostic Report
## Identifying Reflectance Scale Issues

---

## Overview

This report provides a comprehensive checklist to diagnose potential mistakes in your Hyperion preprocessing script (SUREHYP or custom script) that could be causing reflectance scale problems.

**Goal:** Identify if preprocessing is corrupting the 0-1 reflectance scale expected by SAM.

---

## Common Preprocessing Mistakes

### 1. **Incorrect Radiance-to-Reflectance Conversion**

**What should happen:**
```
Reflectance = (Radiance × π × d²) / (ESUN × cos(θ))
```
Where:
- Radiance: At-sensor radiance (W/m²/sr/μm)
- d: Earth-Sun distance (AU)
- ESUN: Solar irradiance at top of atmosphere
- θ: Solar zenith angle

**Result should be:** 0-1 (or 0-100 if as percentage)

**Common mistakes:**
- ❌ Forgot to divide by solar irradiance → values too high
- ❌ Forgot to multiply by π → values too low
- ❌ Used wrong ESUN values (wrong units or wavelengths)
- ❌ Used wrong solar zenith angle
- ❌ Used reflectance scale factor incorrectly

**How to check:**
```python
# After preprocessing, check your reflectance values
refl_data = envi.open('your_reflectance_file.hdr')
refl_cube = refl_data.load()

# Check band in visible range (e.g., band 20 ≈ 550nm green)
band_20 = refl_cube[:, :, 19]  # 0-indexed

print("Band 20 statistics:")
print(f"  Min: {np.min(band_20[band_20 > 0])}")
print(f"  Max: {np.max(band_20)}")
print(f"  Mean: {np.mean(band_20[band_20 > 0])}")
print(f"  Median: {np.median(band_20[band_20 > 0])}")

# Expected values for vegetation/soil:
# - Min: 0.02-0.05
# - Max: 0.3-0.8 (vegetation in NIR can go higher)
# - Mean: 0.1-0.4

# Red flags:
# - Max > 1.5: Likely still in radiance or wrong conversion
# - Max > 10: Definitely scaled wrong (probably 0-10000 scale)
# - Max > 100: Still in DN values or radiance
# - Mean < 0.001: Conversion might have divided twice
```

### 2. **Scale Factor Not Applied or Applied Incorrectly**

**What should happen:**
Many sensors store reflectance as integers to save space:
- Reflectance = DN_value × scale_factor
- Common scale factors: 0.0001 (for 0-10000 range) or 1/65535 (for 16-bit)

**Common mistakes:**
- ❌ Scale factor in metadata but not applied
- ❌ Scale factor applied twice (divided by 10000 in preprocessing AND in SAM script)
- ❌ Wrong scale factor used

**How to check:**
```python
# Check metadata for scale factor
metadata = envi.open('your_file.hdr').metadata

# Look for these keys:
if 'reflectance scale factor' in metadata:
    print(f"Reflectance scale factor: {metadata['reflectance scale factor']}")

if 'data ignore value' in metadata:
    print(f"No-data value: {metadata['data ignore value']}")

# Check actual data type
print(f"Data type: {metadata.get('data type', 'unknown')}")
# Data type codes:
# 1 = 8-bit unsigned integer (0-255)
# 2 = 16-bit signed integer (-32768 to 32767)
# 3 = 32-bit signed integer
# 4 = 32-bit float
# 5 = 64-bit float (double)
# 12 = 16-bit unsigned integer (0-65535)

# If data type is 1, 2, 3, or 12 → integer data → needs scale factor
# If data type is 4 or 5 → float data → may already be scaled

# Check if data looks like it needs scaling:
refl_cube = refl_data.load()
if metadata.get('data type') in ['1', '2', '3', '12']:
    print("WARNING: Integer data type detected!")
    print("Check if scale factor has been applied")
    
    if np.max(refl_cube) > 2.0:
        print("ERROR: Data appears unscaled (max > 2.0)")
        if np.max(refl_cube) > 10000:
            print("  → Likely needs /65535 scaling")
        else:
            print("  → Likely needs /10000 scaling")
```

### 3. **Atmospheric Correction Not Applied or Failed**

**What should happen:**
Raw Hyperion data (Level 1R) is at-sensor radiance, affected by atmosphere.
Must apply atmospheric correction to get surface reflectance:
- FLAASH (commercial)
- ATCOR (commercial)
- Py6S (free Python)
- QUAC (quick atmospheric correction)

**Common mistakes:**
- ❌ Using at-sensor radiance thinking it's reflectance
- ❌ Atmospheric correction failed silently
- ❌ Wrong atmospheric parameters (water vapor, aerosols, etc.)
- ❌ Negative reflectance values after correction (algorithm failure)

**How to check:**
```python
# Atmospheric correction creates distinct absorption features
# Check for water vapor absorption around 1400nm and 1900nm

# Load metadata and find bands near these wavelengths
wavelengths = np.array(metadata['wavelength'], dtype=float)

# Find bands near water absorption (should be removed in preprocessing)
water_band_1 = np.argmin(np.abs(wavelengths - 1400))  # ~1400nm
water_band_2 = np.argmin(np.abs(wavelengths - 1900))  # ~1900nm

# Sample a vegetation pixel
veg_pixel = refl_cube[500, 500, :]  # Adjust indices to vegetation area

# Plot spectrum
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(wavelengths, veg_pixel)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Sample Spectrum - Check for Atmospheric Features')
plt.axvline(1400, color='red', linestyle='--', label='Water vapor 1')
plt.axvline(1900, color='red', linestyle='--', label='Water vapor 2')
plt.legend()
plt.grid(True)
plt.savefig('spectrum_check.png', dpi=150)

# Red flags:
# - Deep absorption at 1400nm and 1900nm → atmosphere not corrected
# - Negative reflectance values → correction failed
# - Reflectance > 1.0 in visible bands → not properly corrected
# - Spiky/noisy spectrum → bad correction or still in radiance
```

### 4. **Bad Band Masking Issues**

**What should happen:**
Hyperion has known bad bands that should be masked:
- Bands 1-7: VNIR overlap/bad
- Bands 58-76: Water vapor ~1360-1400 nm
- Bands 123-135: Water vapor ~1800-1950 nm
- Bands 225-242: SWIR edge/bad

**Common mistakes:**
- ❌ Bad bands not removed → noise affects SAM
- ❌ Bad bands set to zero but still included in analysis
- ❌ Bad bands removed but endmember library still has all 242 bands
- ❌ Wrong band indices (0-indexed vs 1-indexed confusion)

**How to check:**
```python
# Check if bad bands are present
print(f"Total bands in cube: {refl_cube.shape[2]}")
print(f"Expected after bad band removal: ~196")

if refl_cube.shape[2] == 242:
    print("WARNING: All 242 bands present - bad bands not removed!")
elif refl_cube.shape[2] == 196:
    print("Good: Bad bands appear to be removed")
else:
    print(f"Unexpected band count: {refl_cube.shape[2]}")

# Check specific bad band regions
# If bands 58-76 are supposed to be removed, check around band 60
if refl_cube.shape[2] == 242:
    bad_band = refl_cube[:, :, 60]
    print(f"Band 60 (should be bad):")
    print(f"  All zeros: {np.all(bad_band == 0)}")
    print(f"  Max value: {np.max(np.abs(bad_band))}")
    
    if np.max(np.abs(bad_band)) > 0:
        print("ERROR: Bad bands not properly masked!")
```

### 5. **Data Type Conversion Errors**

**What should happen:**
Integer → Float conversion with proper scaling

**Common mistakes:**
- ❌ Integer overflow (16-bit data converted to 8-bit)
- ❌ Integer division instead of float division
- ❌ Truncation during conversion
- ❌ Sign errors (unsigned vs signed integers)

**How to check:**
```python
# Check data type consistency
print(f"Cube dtype: {refl_cube.dtype}")
print(f"Expected: float32 or float64")

if refl_cube.dtype in [np.uint8, np.uint16, np.int16, np.int32]:
    print("ERROR: Data still in integer format!")
    print("Reflectance should be float type")

# Check for truncation
unique_values = np.unique(refl_cube[refl_cube > 0])
print(f"Number of unique values: {len(unique_values)}")

if len(unique_values) < 100:
    print("WARNING: Very few unique values - possible truncation")
```

### 6. **Metadata Corruption**

**What should happen:**
ENVI header file contains accurate wavelength information

**Common mistakes:**
- ❌ Wavelengths in metadata don't match actual data
- ❌ Wavelength units wrong (nanometers vs micrometers)
- ❌ Band order changed but metadata not updated
- ❌ Scale factor in metadata inconsistent with data

**How to check:**
```python
# Verify wavelengths are reasonable
wavelengths = np.array(metadata['wavelength'], dtype=float)

print(f"Wavelength range: {wavelengths.min()} - {wavelengths.max()}")
print(f"First 10 wavelengths: {wavelengths[:10]}")
print(f"Expected: 400-2500 nm (or 0.4-2.5 µm)")

# Red flags:
if wavelengths.max() < 10:
    print("ERROR: Wavelengths in micrometers, should be nanometers!")
    print("Need to multiply by 1000")
    
if wavelengths.min() < 100 or wavelengths.max() > 3000:
    print("ERROR: Wavelength range unrealistic")

# Check wavelength spacing
spacing = np.diff(wavelengths)
print(f"Wavelength spacing: {np.mean(spacing):.2f} ± {np.std(spacing):.2f} nm")
print(f"Expected: ~10 nm")

if np.mean(spacing) > 20 or np.mean(spacing) < 5:
    print("WARNING: Unusual wavelength spacing")
```

---

## Comprehensive Preprocessing Validation Script

**Run this after preprocessing to validate your data:**

```python
import numpy as np
from spectral import envi
import matplotlib.pyplot as plt

def validate_preprocessing(file_path):
    """
    Comprehensive validation of Hyperion preprocessing
    
    Parameters:
    -----------
    file_path : str
        Path to preprocessed .hdr file
    """
    
    print("="*70)
    print("HYPERION PREPROCESSING VALIDATION")
    print("="*70)
    
    # Load data
    img = envi.open(file_path)
    cube = img.load()
    metadata = img.metadata
    
    # Section 1: Basic Info
    print("\n[1] BASIC INFORMATION")
    print("-"*70)
    print(f"Dimensions: {cube.shape}")
    print(f"Data type: {cube.dtype}")
    print(f"Metadata data type: {metadata.get('data type', 'not specified')}")
    
    # Section 2: Reflectance Scale
    print("\n[2] REFLECTANCE SCALE CHECK")
    print("-"*70)
    
    valid_data = cube[cube > 0]
    if len(valid_data) > 0:
        print(f"Min value: {np.min(valid_data):.6f}")
        print(f"Max value: {np.max(valid_data):.6f}")
        print(f"Mean value: {np.mean(valid_data):.6f}")
        print(f"Std dev: {np.std(valid_data):.6f}")
        
        # Diagnosis
        max_val = np.max(valid_data)
        if max_val > 10000:
            print("❌ FAIL: Values > 10000 (likely 16-bit DN)")
            print("   → Need to divide by 65535")
        elif max_val > 2.0:
            print("❌ FAIL: Values > 2.0 (likely integer reflectance)")
            print("   → Need to divide by 10000")
        elif max_val > 1.2:
            print("⚠️  WARNING: Values > 1.2 (unusual but possible)")
        else:
            print("✓ PASS: Reflectance scale looks correct (0-1)")
    else:
        print("❌ FAIL: No valid data found (all zeros)")
    
    # Section 3: Band Count
    print("\n[3] BAND COUNT CHECK")
    print("-"*70)
    print(f"Number of bands: {cube.shape[2]}")
    
    if cube.shape[2] == 242:
        print("⚠️  WARNING: All 242 bands present")
        print("   → Bad bands should be removed (~196 usable bands)")
    elif 190 <= cube.shape[2] <= 200:
        print("✓ PASS: Band count consistent with bad band removal")
    else:
        print(f"⚠️  WARNING: Unexpected band count ({cube.shape[2]})")
    
    # Section 4: Wavelength Check
    print("\n[4] WAVELENGTH CHECK")
    print("-"*70)
    
    if 'wavelength' in metadata:
        wavelengths = np.array(metadata['wavelength'], dtype=float)
        print(f"Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f}")
        print(f"First wavelength: {wavelengths[0]:.2f}")
        print(f"Last wavelength: {wavelengths[-1]:.2f}")
        
        if wavelengths.max() < 10:
            print("❌ FAIL: Wavelengths in micrometers (should be nm)")
            print("   → Multiply by 1000")
        elif 400 <= wavelengths.min() <= 450 and 2350 <= wavelengths.max() <= 2500:
            print("✓ PASS: Wavelengths in correct range (400-2500 nm)")
        else:
            print("⚠️  WARNING: Unusual wavelength range")
        
        # Check spacing
        spacing = np.diff(wavelengths)
        mean_spacing = np.mean(spacing)
        print(f"Mean wavelength spacing: {mean_spacing:.2f} nm")
        
        if 8 <= mean_spacing <= 12:
            print("✓ PASS: Wavelength spacing correct (~10 nm)")
        else:
            print("⚠️  WARNING: Unusual wavelength spacing")
    else:
        print("❌ FAIL: No wavelength information in metadata")
    
    # Section 5: Valid Pixels
    print("\n[5] VALID PIXEL CHECK")
    print("-"*70)
    
    total_pixels = cube.shape[0] * cube.shape[1]
    valid_pixels = np.sum(np.any(cube > 0, axis=2))
    valid_pct = (valid_pixels / total_pixels) * 100
    
    print(f"Total pixels: {total_pixels:,}")
    print(f"Valid pixels: {valid_pixels:,} ({valid_pct:.1f}%)")
    
    if valid_pct < 1:
        print("❌ FAIL: < 1% valid pixels")
    elif valid_pct < 25:
        print("⚠️  WARNING: < 25% valid pixels (check masking)")
    else:
        print("✓ PASS: Reasonable number of valid pixels")
    
    # Section 6: Negative Values Check
    print("\n[6] NEGATIVE VALUES CHECK")
    print("-"*70)
    
    negative_count = np.sum(cube < 0)
    if negative_count > 0:
        print(f"❌ WARNING: {negative_count:,} negative values found")
        print(f"   Min value: {np.min(cube):.6f}")
        print("   → Atmospheric correction may have failed")
        print("   → Or improper masking of no-data values")
    else:
        print("✓ PASS: No negative values")
    
    # Section 7: Spectral Profile Check
    print("\n[7] SPECTRAL PROFILE CHECK")
    print("-"*70)
    
    # Sample a few pixels and check if spectra look reasonable
    mid_row = cube.shape[0] // 2
    mid_col = cube.shape[1] // 2
    
    sample_pixel = cube[mid_row, mid_col, :]
    
    if np.any(sample_pixel > 0):
        print("Sample pixel spectrum statistics:")
        print(f"  Min: {np.min(sample_pixel[sample_pixel > 0]):.6f}")
        print(f"  Max: {np.max(sample_pixel):.6f}")
        print(f"  Mean: {np.mean(sample_pixel[sample_pixel > 0]):.6f}")
        
        # Check for typical vegetation red edge
        if 'wavelength' in metadata:
            wavelengths = np.array(metadata['wavelength'], dtype=float)
            
            # Find red (~680nm) and NIR (~850nm) bands
            red_idx = np.argmin(np.abs(wavelengths - 680))
            nir_idx = np.argmin(np.abs(wavelengths - 850))
            
            red_val = sample_pixel[red_idx]
            nir_val = sample_pixel[nir_idx]
            
            if red_val > 0 and nir_val > 0:
                ndvi = (nir_val - red_val) / (nir_val + red_val)
                print(f"  Sample NDVI: {ndvi:.3f}")
                
                if -1 <= ndvi <= 1:
                    print("  ✓ NDVI in valid range")
                else:
                    print("  ❌ NDVI out of range (scale error!)")
        
        # Plot sample spectrum
        plt.figure(figsize=(12, 6))
        if 'wavelength' in metadata:
            plt.plot(wavelengths, sample_pixel)
            plt.xlabel('Wavelength (nm)')
        else:
            plt.plot(sample_pixel)
            plt.xlabel('Band number')
        plt.ylabel('Reflectance')
        plt.title(f'Sample Pixel Spectrum (row={mid_row}, col={mid_col})')
        plt.grid(True, alpha=0.3)
        plt.savefig('preprocessing_validation_spectrum.png', dpi=150, bbox_inches='tight')
        print("\n  Saved spectrum plot: preprocessing_validation_spectrum.png")
    else:
        print("❌ Sample pixel is all zeros (invalid)")
    
    # Section 8: Scale Factor Check
    print("\n[8] SCALE FACTOR CHECK")
    print("-"*70)
    
    if 'reflectance scale factor' in metadata:
        scale_factor = float(metadata['reflectance scale factor'])
        print(f"Scale factor in metadata: {scale_factor}")
        
        # Check if it's been applied
        if scale_factor != 1.0:
            max_expected = 1.0 / scale_factor
            actual_max = np.max(cube)
            
            if actual_max > max_expected * 0.5:
                print(f"⚠️  WARNING: Scale factor may not be applied")
                print(f"   Expected max after scaling: ~1.0")
                print(f"   Actual max: {actual_max:.2f}")
            else:
                print("✓ Scale factor appears to be applied")
    else:
        print("No scale factor in metadata")
    
    # Final Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print("\nChecklist:")
    
    # Compile results
    checks = []
    
    # Scale check
    max_val = np.max(valid_data) if len(valid_data) > 0 else 0
    if max_val <= 1.2:
        checks.append("✓ Reflectance scale correct")
    else:
        checks.append("❌ Reflectance scale INCORRECT")
    
    # Band count
    if 190 <= cube.shape[2] <= 200:
        checks.append("✓ Band count correct")
    else:
        checks.append("⚠️  Band count unusual")
    
    # Valid pixels
    if valid_pct > 25:
        checks.append("✓ Sufficient valid pixels")
    else:
        checks.append("❌ Too few valid pixels")
    
    # Negative values
    if negative_count == 0:
        checks.append("✓ No negative values")
    else:
        checks.append("⚠️  Negative values present")
    
    for check in checks:
        print(f"  {check}")
    
    print("\n" + "="*70)
    
    return cube, metadata


# Usage:
# cube, metadata = validate_preprocessing('your_preprocessed_file.hdr')
```

---

## Quick Diagnostic Commands

**Run these immediately to check your data:**

```python
# Quick check #1: Is it reflectance?
max_val = np.max(cube[cube > 0])
print(f"Max reflectance: {max_val:.4f}")
if max_val > 2.0:
    print("ERROR: Not reflectance scale!")

# Quick check #2: Are there valid pixels?
valid_pct = np.sum(np.any(cube > 0, axis=2)) / (cube.shape[0] * cube.shape[1]) * 100
print(f"Valid pixels: {valid_pct:.1f}%")

# Quick check #3: Do spectral angles make sense?
test_pixel = cube[100, 100, :]
test_endmember = endmembers[list(endmembers.keys())[0]]

if np.any(test_pixel > 0):
    cos_angle = np.dot(test_pixel, test_endmember) / \
                (np.linalg.norm(test_pixel) * np.linalg.norm(test_endmember))
    
    print(f"Test cosine: {cos_angle:.4f}")
    
    if abs(cos_angle) > 1.0:
        print("ERROR: Invalid cosine - SCALE MISMATCH!")
    else:
        angle_deg = np.degrees(np.arccos(cos_angle))
        print(f"Test angle: {angle_deg:.2f} degrees")
```

---

## Most Common Preprocessing Mistakes (Summary)

1. **Scale not applied**: Data in 0-10000 range, not divided by 10000
2. **Wrong scale factor**: Used /1000 instead of /10000, or vice versa
3. **Scale applied twice**: Divided by 10000 in preprocessing AND in SAM script
4. **Still in radiance**: Forgot atmospheric correction entirely
5. **Bad atmospheric correction**: Negative values or values > 1.5
6. **Integer data type**: Never converted from uint16 to float32
7. **Band mismatch**: Bad bands removed from cube but not from endmembers
8. **Wavelength units wrong**: Micrometers instead of nanometers

---

## Recommended Action Plan

1. **Run validation script** on your preprocessed data
2. **Check all 8 sections** in the validation output
3. **Identify which checks fail**
4. **Fix preprocessing script** or apply corrections in SAM script
5. **Re-validate** after fixes
6. **Then run SAM** with corrected data

The validation script will create a detailed report showing exactly what's wrong.

---

## Expected Output (Good Preprocessing)

```
======================================================================
HYPERION PREPROCESSING VALIDATION
======================================================================

[1] BASIC INFORMATION
----------------------------------------------------------------------
Dimensions: (1000, 800, 196)
Data type: float32

[2] REFLECTANCE SCALE CHECK
----------------------------------------------------------------------
Min value: 0.001234
Max value: 0.876543
Mean value: 0.234567
✓ PASS: Reflectance scale looks correct (0-1)

[3] BAND COUNT CHECK
----------------------------------------------------------------------
Number of bands: 196
✓ PASS: Band count consistent with bad band removal

[4] WAVELENGTH CHECK
----------------------------------------------------------------------
Wavelength range: 426.82 - 2395.50
✓ PASS: Wavelengths in correct range (400-2500 nm)
Mean wavelength spacing: 10.08 nm
✓ PASS: Wavelength spacing correct (~10 nm)

[5] VALID PIXEL CHECK
----------------------------------------------------------------------
Total pixels: 800,000
Valid pixels: 654,321 (81.8%)
✓ PASS: Reasonable number of valid pixels

[6] NEGATIVE VALUES CHECK
----------------------------------------------------------------------
✓ PASS: No negative values

[7] SPECTRAL PROFILE CHECK
----------------------------------------------------------------------
Sample NDVI: 0.687
✓ NDVI in valid range

[8] SCALE FACTOR CHECK
----------------------------------------------------------------------
✓ Scale factor appears to be applied

======================================================================
VALIDATION SUMMARY
======================================================================

Checklist:
  ✓ Reflectance scale correct
  ✓ Band count correct
  ✓ Sufficient valid pixels
  ✓ No negative values
```

If you see this output, your preprocessing is correct!
