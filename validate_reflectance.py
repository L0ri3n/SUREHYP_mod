"""
Validation script for Hyperion reflectance data
Run this in conda environment: hyperion_roger

Usage:
    conda activate hyperion_roger
    python validate_reflectance.py
"""

import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import os
from pathlib import Path

def validate_hyperion_reflectance(file_path):
    """
    Comprehensive validation of Hyperion reflectance data.

    Parameters:
    -----------
    file_path : str
        Path to reflectance .hdr file (without extension)
    """

    print("="*80)
    print("HYPERION REFLECTANCE VALIDATION")
    print("="*80)
    print(f"\nAnalyzing: {file_path}")

    # Load data
    hdr_path = file_path + '.hdr'
    img_path = file_path + '.img' if os.path.exists(file_path + '.img') else file_path + '.bip'

    if not os.path.exists(hdr_path):
        print(f"ERROR: HDR file not found: {hdr_path}")
        return None

    try:
        img = envi.open(hdr_path, img_path)
        R = img.load()
        metadata = img.metadata
    except Exception as e:
        print(f"ERROR loading file: {e}")
        return None

    # Get wavelengths
    if 'wavelength' in metadata:
        bands = np.array([float(w) for w in metadata['wavelength']])
    else:
        # Try spectral_info.txt
        spectral_info_path = file_path + '_spectral_info.txt'
        if os.path.exists(spectral_info_path):
            bands = []
            with open(spectral_info_path, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        bands.append(float(parts[1].strip()))
            bands = np.array(bands)
        else:
            print("WARNING: No wavelength information found")
            bands = np.arange(R.shape[2])

    # Check scale factor in metadata
    scale_factor = 1.0
    if 'reflectance scale factor' in metadata:
        scale_factor = float(metadata['reflectance scale factor'])
    elif 'scale factor' in metadata:
        scale_factor = float(metadata['scale factor'])

    print(f"\n{'='*80}")
    print("METADATA INFORMATION")
    print(f"{'='*80}")
    print(f"Image dimensions: {R.shape[0]} x {R.shape[1]} x {R.shape[2]}")
    print(f"Data type: {R.dtype}")
    print(f"Scale factor in metadata: {scale_factor}")
    print(f"Wavelength range: {bands.min():.1f} - {bands.max():.1f} nm")
    print(f"Number of bands: {len(bands)}")

    # Section 1: Reflectance Scale Check
    print(f"\n{'='*80}")
    print("REFLECTANCE SCALE ANALYSIS")
    print(f"{'='*80}")

    # Use nansum to handle masked water vapor bands
    valid_pixels = np.nansum(R, axis=2) > 0
    n_valid = np.sum(valid_pixels)
    n_total = R.shape[0] * R.shape[1]

    print(f"\nValid pixels: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")

    if n_valid > 0:
        valid_R = R[valid_pixels]
        valid_R_nonzero = valid_R[valid_R > 0]

        if len(valid_R_nonzero) > 0:
            print(f"\nReflectance Statistics (valid pixels only):")
            print(f"  Min:    {np.min(valid_R_nonzero):.2f}")
            print(f"  Max:    {np.max(valid_R_nonzero):.2f}")
            print(f"  Mean:   {np.mean(valid_R_nonzero):.2f}")
            print(f"  Median: {np.median(valid_R_nonzero):.2f}")
            print(f"  Std:    {np.std(valid_R_nonzero):.2f}")

            # Percentiles
            p01 = np.percentile(valid_R_nonzero, 1)
            p99 = np.percentile(valid_R_nonzero, 99)
            print(f"\nPercentiles:")
            print(f"  1st:  {p01:.2f}")
            print(f"  99th: {p99:.2f}")

            # Diagnosis
            max_val = np.max(valid_R_nonzero)
            print(f"\n{'='*80}")
            print("DIAGNOSIS")
            print(f"{'='*80}")

            if max_val > 100000:
                print("❌ CRITICAL ERROR: Values > 100,000")
                print("   → Data appears to be scaled radiance, not reflectance")
                print(f"   → Suggested correction: Divide by {max_val/1.0:.0f}")
            elif max_val > 10000:
                print("❌ ERROR: Values > 10,000")
                print("   → Likely 16-bit scaled radiance")
                print("   → Suggested correction: Divide by 65535 or check scale factor")
            elif max_val > 2.0:
                print("❌ ERROR: Values > 2.0")
                print("   → Data is scaled by 10,000 or 1,000")
                print(f"   → Suggested correction: Divide by {scale_factor if scale_factor > 1 else 10000}")
            elif max_val > 1.2:
                print("⚠️  WARNING: Values > 1.2")
                print("   → Unusual but may be valid for very bright surfaces")
            else:
                print("✅ PASS: Reflectance scale appears correct (0-1 range)")

            # Check for specific problematic bands
            print(f"\n{'='*80}")
            print("BAND-SPECIFIC ANALYSIS")
            print(f"{'='*80}")

            # Find max value per band
            max_per_band = np.nanmax(R, axis=(0, 1))

            # Find top 5 problematic bands
            top_5_idx = np.argsort(max_per_band)[-5:][::-1]

            print("\nTop 5 bands with highest values:")
            for idx in top_5_idx:
                print(f"  Band {idx+1:3d} ({bands[idx]:7.2f} nm): Max = {max_per_band[idx]:.2f}")

            # Check specific regions
            # Around 750nm (should be clean VNIR-SWIR transition)
            idx_750 = np.argmin(np.abs(bands - 750))
            print(f"\nNear 750nm (VNIR-SWIR transition):")
            print(f"  Band index: {idx_750+1}, Wavelength: {bands[idx_750]:.2f} nm")
            print(f"  Max value: {max_per_band[idx_750]:.2f}")
            if max_per_band[idx_750] > 10:
                print("  ⚠️  WARNING: Abnormally high - check bad band removal")

            # Around 2050nm (end of SWIR)
            if bands.max() >= 2000:
                idx_2050 = np.argmin(np.abs(bands - 2050))
                print(f"\nNear 2050nm (SWIR end region):")
                print(f"  Band index: {idx_2050+1}, Wavelength: {bands[idx_2050]:.2f} nm")
                print(f"  Max value: {max_per_band[idx_2050]:.2f}")
                if max_per_band[idx_2050] > 10:
                    print("  ⚠️  WARNING: Abnormally high - uncalibrated SWIR bands may still be present")

    # Section 2: Spectral Profile Analysis
    print(f"\n{'='*80}")
    print("SPECTRAL PROFILE ANALYSIS")
    print(f"{'='*80}")

    # Sample 5 random valid pixels
    if n_valid >= 5:
        np.random.seed(42)
        valid_coords = np.argwhere(valid_pixels)
        sample_indices = np.random.choice(len(valid_coords), min(5, len(valid_coords)), replace=False)

        plt.figure(figsize=(14, 8))

        for idx in sample_indices:
            row, col = valid_coords[idx]
            spectrum = R[row, col, :]
            plt.plot(bands, spectrum, label=f'Pixel ({row}, {col})', alpha=0.7, linewidth=1.5)

        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Reflectance', fontsize=12)
        plt.title('Sample Spectra from Reflectance Image', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([400, min(2500, bands.max())])

        # Add reference lines
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Expected max (1.0)')
        plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        # Highlight problem regions
        plt.axvline(x=750, color='orange', linestyle=':', alpha=0.5, label='~750nm check')
        if bands.max() >= 2000:
            plt.axvline(x=2050, color='orange', linestyle=':', alpha=0.5, label='~2050nm check')

        output_dir = os.path.dirname(file_path)
        if not output_dir:
            output_dir = '.'
        plot_path = os.path.join(output_dir, 'validation_spectra.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✅ Saved spectral plot: {plot_path}")

    # Section 3: NDVI Check (should be -1 to +1)
    print(f"\n{'='*80}")
    print("NDVI VALIDATION (Scale Independence Check)")
    print(f"{'='*80}")

    # Find NIR and Red bands
    nir_idx = np.argmin(np.abs(bands - 850))
    red_idx = np.argmin(np.abs(bands - 660))

    nir = R[:, :, nir_idx].astype(float)
    red = R[:, :, red_idx].astype(float)

    with np.errstate(invalid='ignore', divide='ignore'):
        ndvi = (nir - red) / (nir + red)

    ndvi = np.where(np.isfinite(ndvi), ndvi, 0)
    ndvi_valid = ndvi[valid_pixels]

    if len(ndvi_valid) > 0:
        print(f"\nNDVI Statistics:")
        print(f"  Min:  {np.min(ndvi_valid):.3f}")
        print(f"  Max:  {np.max(ndvi_valid):.3f}")
        print(f"  Mean: {np.mean(ndvi_valid):.3f}")

        if np.min(ndvi_valid) < -1.5 or np.max(ndvi_valid) > 1.5:
            print("  ❌ ERROR: NDVI out of valid range (-1 to +1)")
            print("  → This indicates SEVERE scale issues!")
        elif np.abs(np.max(ndvi_valid)) < 0.01:
            print("  ⚠️  WARNING: NDVI near zero - possible scale collapse")
        else:
            print("  ✅ PASS: NDVI in valid range")

    # Final Summary
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")

    if n_valid > 0 and len(valid_R_nonzero) > 0:
        max_val = np.max(valid_R_nonzero)

        if max_val > 2.0:
            print("\n❌ REFLECTANCE SCALE IS INCORRECT")
            print("\nRECOMMENDED FIXES:")
            print("1. Check atmospheric correction output in atmoCorrection.py")
            print("2. Verify scale factor is applied correctly during radiance-to-reflectance")
            print(f"3. Divide current data by: {max_val:.0f} (or check metadata scale factor: {scale_factor})")
            print("\nLIKELY ROOT CAUSES:")
            if max_val > 100000:
                print("  - Data is still in scaled radiance units")
                print("  - Atmospheric correction may have failed")
            elif scale_factor > 1:
                print(f"  - Scale factor ({scale_factor}) applied during preprocessing but not removed")
                print("  - atmoCorrection.py may not account for preprocessed radiance scaling")
            else:
                print("  - Radiance-to-reflectance conversion has wrong scale factor")
        else:
            print("\n✅ REFLECTANCE SCALE APPEARS CORRECT")
            print("\nData is ready for SAM classification")

    print(f"\n{'='*80}")

    return R, bands, metadata


if __name__ == '__main__':
    import sys

    # Default file paths
    base_path = r"C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT"

    # Check for command-line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use the currently configured image from process_hyperion.py
        fname = 'EO1H2020342013284110KF'
        file_path = os.path.join(base_path, fname + '_reflectance')

    print(f"Validating: {file_path}")

    if not os.path.exists(file_path + '.hdr'):
        print(f"\nERROR: File not found: {file_path}.hdr")
        print(f"\nUsage: python validate_reflectance.py [path_to_reflectance_file]")
        print(f"Example: python validate_reflectance.py {base_path}/EO1H2020342013284110KF_reflectance")
        sys.exit(1)

    R, bands, metadata = validate_hyperion_reflectance(file_path)
