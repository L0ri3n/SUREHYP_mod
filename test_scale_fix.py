"""
Quick test script to validate the scale factor fix
Run in conda environment: hyperion_roger

This will test loading a reflectance file and checking if the scale is correct.
"""

import numpy as np
import spectral.io.envi as envi
import os

# Configuration
base_path = r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT"
fname = 'EO1H2020342013284110KF'  # Current image being processed
pathToReflectanceImage = os.path.join(base_path, fname + '_reflectance')

print("="*80)
print("TESTING SCALE FACTOR FIX")
print("="*80)
print(f"\nTarget file: {pathToReflectanceImage}")

# Check if file exists
hdr_path = pathToReflectanceImage + '.hdr'
img_path = pathToReflectanceImage + '.img' if os.path.exists(pathToReflectanceImage + '.img') else pathToReflectanceImage + '.bip'

if not os.path.exists(hdr_path):
    print(f"\n❌ ERROR: File not found: {hdr_path}")
    exit(1)

print(f"HDR path: {hdr_path}")
print(f"IMG path: {img_path}")

# Load image
print("\n" + "-"*80)
print("LOADING IMAGE")
print("-"*80)

img = envi.open(hdr_path, img_path)
R_raw = img.load()

print(f"Image shape: {R_raw.shape}")
print(f"Data type: {R_raw.dtype}")

# Check for scale factor
print("\n" + "-"*80)
print("SCALE FACTOR CHECK")
print("-"*80)

if 'scale factor' in img.metadata:
    scale_factor_list = img.metadata['scale factor']
    print(f"Scale factor found in metadata:")
    print(f"  Type: {type(scale_factor_list)}")
    print(f"  First value: {scale_factor_list[0] if isinstance(scale_factor_list, list) else scale_factor_list}")
    print(f"  Length: {len(scale_factor_list) if isinstance(scale_factor_list, list) else 1}")

    # Parse scale factor
    if isinstance(scale_factor_list, list):
        scale_factor = np.array([float(s) for s in scale_factor_list])
        if len(np.unique(scale_factor)) == 1:
            scale_factor = scale_factor[0]
            print(f"  All values identical: {scale_factor}")
        else:
            print(f"  Per-band scale factors: min={scale_factor.min()}, max={scale_factor.max()}")
    else:
        scale_factor = float(scale_factor_list)
        print(f"  Single value: {scale_factor}")
else:
    print("❌ No scale factor found in metadata!")
    scale_factor = None

# Analyze raw data
print("\n" + "-"*80)
print("RAW DATA STATISTICS (BEFORE SCALE CORRECTION)")
print("-"*80)

valid_raw = R_raw[R_raw > 0]
if len(valid_raw) > 0:
    print(f"Valid pixels: {len(valid_raw):,}")
    print(f"Min:    {np.min(valid_raw):.2f}")
    print(f"Max:    {np.max(valid_raw):.2f}")
    print(f"Mean:   {np.mean(valid_raw):.2f}")
    print(f"Median: {np.median(valid_raw):.2f}")

    # Diagnosis
    max_raw = np.max(valid_raw)
    if max_raw > 10000:
        print("\n❌ Data appears to be in incorrect scale (values > 10,000)")
    elif max_raw > 2.0:
        print("\n⚠️  Data may need scale correction (values > 2.0)")
    else:
        print("\n✅ Data already appears to be in 0-1 scale")

# Apply scale factor if present
if scale_factor is not None:
    print("\n" + "-"*80)
    print("APPLYING SCALE FACTOR CORRECTION")
    print("-"*80)

    R_corrected = R_raw.astype(np.float32) / scale_factor

    valid_corrected = R_corrected[R_corrected > 0]
    if len(valid_corrected) > 0:
        print(f"\nCorrected data statistics:")
        print(f"Min:    {np.min(valid_corrected):.6f}")
        print(f"Max:    {np.max(valid_corrected):.6f}")
        print(f"Mean:   {np.mean(valid_corrected):.6f}")
        print(f"Median: {np.median(valid_corrected):.6f}")

        # Validate
        max_corrected = np.max(valid_corrected)
        if max_corrected > 2.0:
            print(f"\n❌ STILL INCORRECT: Max value {max_corrected:.2f} > 2.0")
            print(f"   Suggested additional correction: ÷ {max_corrected/1.0:.0f}")
        elif max_corrected < 0.001:
            print(f"\n⚠️  WARNING: Max value {max_corrected:.6f} < 0.001")
            print("   Scale factor may have been applied twice")
        else:
            print(f"\n✅ SUCCESS: Reflectance now in correct 0-1 range!")

        # Sample spectrum check
        print("\n" + "-"*80)
        print("SAMPLE SPECTRUM CHECK")
        print("-"*80)

        # Find a random valid pixel
        valid_pixels = np.argwhere(np.nansum(R_corrected, axis=2) > 0)
        if len(valid_pixels) > 0:
            sample_idx = valid_pixels[len(valid_pixels)//2]  # Middle pixel
            sample_spectrum = R_corrected[sample_idx[0], sample_idx[1], :]

            print(f"\nSample pixel at ({sample_idx[0]}, {sample_idx[1]}):")
            print(f"  Min reflectance: {np.nanmin(sample_spectrum):.6f}")
            print(f"  Max reflectance: {np.nanmax(sample_spectrum):.6f}")
            print(f"  Mean reflectance: {np.nanmean(sample_spectrum[sample_spectrum > 0]):.6f}")

            # Check for extreme bands
            max_band_idx = np.nanargmax(sample_spectrum)
            print(f"\n  Highest reflectance at band {max_band_idx+1}: {sample_spectrum[max_band_idx]:.6f}")

            if 'wavelength' in img.metadata:
                wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
                print(f"  Wavelength: {wavelengths[max_band_idx]:.2f} nm")

else:
    print("\n⚠️  Cannot apply scale correction - no scale factor found")

# Final summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if scale_factor is not None:
    if max_corrected <= 1.2:
        print("\n✅ FIX SUCCESSFUL!")
        print("   Reflectance data is now in the correct 0-1 range.")
        print("   The data is ready for SAM classification.")
    else:
        print("\n❌ FIX INCOMPLETE")
        print(f"   Max reflectance is still {max_corrected:.2f} (should be < 1.2)")
        print("   Additional corrections may be needed.")
else:
    print("\n⚠️  NO SCALE FACTOR TO CORRECT")
    print("   Check if the data was saved with scale factor.")

print("\n" + "="*80)
