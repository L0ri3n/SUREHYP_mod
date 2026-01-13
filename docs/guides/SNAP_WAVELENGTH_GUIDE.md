# SNAP Wavelength Compatibility Guide

This guide explains how to configure spectral wavelength information in ENVI HDR files for optimal SNAP compatibility.

## Overview

When processing Hyperion imagery, spectral wavelength information can be included in the ENVI header (`.hdr`) file. However, SNAP software has specific requirements and behaviors regarding wavelength metadata that can affect both visualization and band math operations.

## The Problem

SNAP automatically appends wavelength values to band names when the `wavelength` field is present in the HDR file. For example, a band named `band_1` with wavelength `426.82` becomes `band_1 (426.82)` in SNAP. This causes issues:

- **Band math expressions**: The parentheses in band names like `band_1 (426.82)` cause "Undefined function" errors
- **Visualization**: Without wavelength metadata, spectral plots don't show wavelength labels

## The Solution

The updated `fix_envi_hdr_for_snap()` function provides flexible control over wavelength metadata:

### Configuration Options

In `process_hyperion.py`, you'll find these configuration options (around line 1066):

```python
# ============================================================
# SNAP WAVELENGTH COMPATIBILITY OPTIONS
# ============================================================

# Option 1: Load wavelengths from external file
snap_wavelength_file = None
# Example: snap_wavelength_file = basePath + 'OUT/EO1H2020342016359110KF_reflectance_spectral_info.txt'

# Option 2: Keep wavelength field in HDR file
# - True: Better for SNAP visualization (wavelength labels in plots)
# - False: Safer for SNAP band math (avoids expression errors)
snap_keep_wavelength = True
```

### Usage Scenarios

#### Scenario 1: Better Visualization (Recommended for viewing)

Set `snap_keep_wavelength = True` to keep wavelength information in the HDR file:

```python
snap_keep_wavelength = True
```

**Benefits:**
- Spectral plots show wavelength labels
- Better for visual analysis and exploration
- Wavelength information immediately available in SNAP

**Limitations:**
- Band math expressions may fail if they reference band names
- Workaround: Use band indices (e.g., `$1`, `$2`) instead of names in expressions

#### Scenario 2: Safe for Band Math (Recommended for analysis)

Set `snap_keep_wavelength = False` to remove wavelength from HDR:

```python
snap_keep_wavelength = False
```

**Benefits:**
- Band math expressions work reliably
- No "Undefined function" errors
- Wavelength info saved separately in `*_spectral_info.txt`

**Limitations:**
- Spectral plots won't show wavelength labels by default
- Need to manually reference the `*_spectral_info.txt` file

#### Scenario 3: Custom Wavelength Calibration

Use an external wavelength file for custom spectral calibration:

```python
snap_wavelength_file = 'C:/path/to/custom_wavelengths.txt'
snap_keep_wavelength = True
```

**Wavelength File Format:**

The file can use either of these formats:

**Format 1: CSV with FWHM**
```
# Comment lines start with #
band_1, 426.82, 11.3871
band_2, 436.99, 11.3871
band_3, 447.17, 11.3871
```

**Format 2: Plain wavelengths**
```
# One wavelength per line
426.82
436.99
447.17
```

## How It Works

1. **During processing**, the script computes wavelengths from sensor calibration
2. **After saving**, the `fix_envi_hdr_for_snap()` function processes the HDR file:
   - If `wavelength_file` is provided, loads wavelengths from that file
   - If `keep_wavelength = True`, adds/updates wavelength field in HDR
   - If `keep_wavelength = False`, removes wavelength field from HDR
   - Always saves wavelength info to `*_spectral_info.txt` for reference

3. **The result**: An HDR file optimized for your SNAP workflow

## Example: Using Your Existing Spectral Info File

You already have a spectral info file generated from previous processing:

```
C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT\EO1H2020342016359110KF_reflectance_spectral_info.txt
```

To use it in your next processing run:

```python
# In process_hyperion.py configuration section:
snap_wavelength_file = basePath + 'OUT/EO1H2020342016359110KF_reflectance_spectral_info.txt'
snap_keep_wavelength = True  # Include wavelengths in HDR
```

This will use the wavelengths from your existing file instead of recomputing them.

## Recommendations

**For general use:**
- Start with `snap_keep_wavelength = True` for better visualization
- If you encounter band math errors, switch to `snap_keep_wavelength = False`

**For automated workflows:**
- Use `snap_keep_wavelength = False` for reliability
- Reference the `*_spectral_info.txt` file when wavelength data is needed

**For custom calibrations:**
- Create a wavelength file with your calibrated values
- Set `snap_wavelength_file` to point to your file
- Set `snap_keep_wavelength = True` to include them in the HDR

## Files Generated

After processing, you'll find:

1. `*_reflectance.hdr` - ENVI header file (with or without wavelength field)
2. `*_reflectance_spectral_info.txt` - Wavelength reference (always created)
3. `*_reflectance.hdr.backup` - Original HDR backup (first time only)

## Testing Your Configuration

After changing the configuration:

1. Run the processing script
2. Open the output in SNAP
3. Test visualization: Check if spectral plots show wavelengths
4. Test band math: Try an expression like `($2 - $1) / ($2 + $1)`

## Branch Information

This feature was developed in the `snap-wavelength-compatibility` branch.

To switch back to main: `git checkout main`
To use this branch: `git checkout snap-wavelength-compatibility`

## Questions?

This modification maintains backward compatibility. If you don't set any options, the script behaves as before (removes wavelengths for safety).
