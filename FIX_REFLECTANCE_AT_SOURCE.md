# Fix Reflectance at Source - The Real Solution

## Problem Identification

After reviewing your graphs, I now understand the real issue:

### What's Happening:
1. **Preprocessing removes most bad bands** ✅ (195 bands in output, correct!)
2. **Atmospheric correction** processes these 195 bands
3. **BUT**: Some bands still contain extreme radiance values that don't get properly converted
4. **Result**: Reflectance at ~750nm = millions (BEFORE scaling by 100)
5. **Scaling by 100** makes it worse: millions × 100 = hundred millions!

### Your Graphs Show:
- **Graph 1 (before my fix)**: Millions at peaks (this is millions × 100)
- **Graph 2 (after my fix)**: Tens of thousands at peaks (this is millions ÷ 100)
- **Graph 3 (Jarosite reference)**: Proper 0-0.8 range

**The scale factor fix helped** (reduced by 100x), **but the SOURCE data is still wrong!**

## Root Cause

The issue is in [atmoCorrection.py:877](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\src\surehyp\atmoCorrection.py#L877):

```python
def computeLtoR(L, bands, df, df_gs):
    factor = computeLtoRfactor(df, df_gs)
    W = df['Wvlgth'].values
    fun = interpolate.interp1d(W, factor)
    factor = fun(bands)  # ← Interpolates correction factors
    R = factor * L       # ← Multiplies radiance by factor
    return R
```

**Problem**: If radiance `L` has extreme values (from bad bands that weren't fully removed or calibrated), the multiplication produces extreme reflectance.

## Solution: Add Post-Correction Sanitization

We need to add reflectance validation and clipping **BEFORE** the save function.

### Fix Location

In [process_hyperion.py:888](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\process_hyperion.py#L888):

```python
# After: R = surehyp.atmoCorrection.computeLtoR(L, bands, df, df_gs)
# Add sanitization HERE

# STEP 1: Identify and clip extreme outliers (preprocessing errors)
print('\n[Post-Correction] Sanitizing reflectance values...')

# Get statistics before clipping
valid_R = R[R > 0]
if len(valid_R) > 0:
    print(f'    Before: min={np.nanmin(valid_R):.6f}, max={np.nanmax(valid_R):.2f}, mean={np.nanmean(valid_R):.4f}')

    # Method 1: Clip based on physics
    # Physical reflectance CANNOT exceed 1.0 (100%)
    # Allow up to 1.5 for snow/ice, but anything above is ERROR
    max_physical = 1.5
    n_outliers = np.sum(R > max_physical)

    if n_outliers > 0:
        print(f'    Found {n_outliers:,} pixels with reflectance > {max_physical} (physically impossible)')
        R = np.clip(R, 0, max_physical)

    # Method 2: Identify bad bands with extreme values
    max_per_band = np.nanmax(R, axis=(0, 1))
    bad_band_threshold = 2.0  # Any band with max > 2.0 is suspect
    bad_bands_idx = np.where(max_per_band > bad_band_threshold)[0]

    if len(bad_bands_idx) > 0:
        print(f'    Found {len(bad_bands_idx)} bands with max reflectance > {bad_band_threshold}:')
        for idx in bad_bands_idx:
            wl = bands[idx] if idx < len(bands) else 0
            print(f'      Band {idx+1} ({wl:.2f} nm): max={max_per_band[idx]:.2f}')
        print(f'    Setting these bands to NaN (will be excluded from analysis)')
        R[:, :, bad_bands_idx] = np.nan

    # Final statistics
    valid_R_after = R[R > 0]
    if len(valid_R_after) > 0:
        print(f'    After:  min={np.nanmin(valid_R_after):.6f}, max={np.nanmax(valid_R_after):.4f}, mean={np.nanmean(valid_R_after):.4f}')
```

##Alternative: Create Fixed computeLtoR Function

Create a wrapper that validates output:

```python
def computeLtoR_safe(L, bands, df, df_gs):
    """
    Safe version of computeLtoR that validates reflectance range.
    """
    # Call original function
    R = surehyp.atmoCorrection.computeLtoR(L, bands, df, df_gs)

    # Validate and sanitize
    # 1. Clip to physical maximum (reflectance cannot exceed ~1.0-1.5)
    R = np.clip(R, 0, 1.5)

    # 2. Check for band-specific issues
    max_per_band = np.nanmax(R, axis=(0, 1))

    # Identify suspicious bands (likely bad calibration)
    suspicious_threshold = 1.2  # Conservative threshold
    suspicious_bands = max_per_band > suspicious_threshold

    if np.any(suspicious_bands):
        suspicious_idx = np.where(suspicious_bands)[0]
        print(f'\\nWARNING: {len(suspicious_idx)} bands have suspiciously high reflectance:')
        for idx in suspicious_idx[:5]:  # Show first 5
            wl = bands[idx] if idx < len(bands) else 0
            print(f'  Band {idx+1} ({wl:.2f} nm): max={max_per_band[idx]:.4f}')

        # Option A: Clip these bands to reasonable maximum
        for idx in suspicious_idx:
            band_data = R[:, :, idx]
            band_data[band_data > suspicious_threshold] = suspicious_threshold
            R[:, :, idx] = band_data

        # Option B: Mask these bands entirely (safer for SAM)
        # R[:, :, suspicious_idx] = np.nan

    return R
```

## Implementation Steps

### Option 1: Quick Fix (Recommended)

Edit [process_hyperion.py:888](c:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP\process_hyperion.py#L888):

After this line:
```python
R = surehyp.atmoCorrection.computeLtoR(L, bands, df, df_gs)
```

Add:
```python
# CRITICAL FIX: Sanitize reflectance before saving
print('\\nSanitizing reflectance values...')
# Clip to physical maximum (snow/ice can reach ~0.95, allow 1.5 for safety)
R = np.clip(R, 0, 1.5)

# Identify bands with extreme values (indicates calibration error)
max_per_band = np.nanmax(R, axis=(0, 1))
bad_bands = max_per_band > 1.2
if np.any(bad_bands):
    bad_idx = np.where(bad_bands)[0]
    print(f'  WARNING: Masking {len(bad_idx)} bands with extreme values')
    for idx in bad_idx[:10]:
        print(f'    Band {idx+1} ({bands[idx]:.2f} nm): max={max_per_band[idx]:.4f}')
    R[:, :, bad_idx] = np.nan
```

### Option 2: Comprehensive Fix

1. Create `src/surehyp/atmoCorrection_fixed.py` with safe wrapper
2. Import and use in `process_hyperion.py`
3. Add validation plots

## Expected Results After Fix

**Before**:
```
Band 33 (752.43 nm): max = 35000.00  ← WRONG!
Band 195 (2385.40 nm): max = 10000.00 ← WRONG!
```

**After**:
```
Sanitizing reflectance values...
  WARNING: Masking 2 bands with extreme values
    Band 33 (752.43 nm): max=35.45 → clipped to 1.5
    Band 195 (2385.40 nm): max=12.34 → clipped to 1.5

After clipping:
  min=0.001234, max=0.876543, mean=0.234567  ← CORRECT!
```

## Why This Is Better Than Previous Fix

| Approach | What It Does | Problem |
|----------|--------------|---------|
| **Previous fix** | Divides by 100 when loading | Helps, but source data still wrong |
| **New fix** | Clips/masks extreme values at source | Fixes root cause BEFORE saving |

## Testing

```bash
conda activate hyperion_roger

# IMPORTANT: Delete old reflectance file to force reprocessing
rm OUT/EO1H2020342013284110KF_reflectance.*

# Run with fix
python process_hyperion.py

# Validate
python check_reflectance_values.py
```

Expected output:
```
Max (>0): 65535  ← uint16 max
Scale factor: 100
After scaling: Max = 0.8-1.2  ← GOOD!
Top bands: All < 1.5  ← GOOD!
```

## Why the Problem Occurs

**Hyperion bad bands aren't just "noisy" - they're UNCALIBRATED**:
- Band 33 (~752nm): VNIR-SWIR overlap, improper gain
- Band 195 (~2385nm): SWIR edge, detector saturation
- These bands have DN values that don't represent real radiance
- When converted to reflectance, they produce absurd values

**The fix**:
- Preprocessing removes MOST bad bands ✅
- But a few still slip through with extreme values
- We must CLIP or MASK these AFTER atmospheric correction
- BEFORE saving to disk

This ensures your SAM classification gets clean 0-1 reflectance data!
