"""
Comprehensive Validation Script for Hyperion Preprocessing
Checks for common preprocessing mistakes that cause SAM issues

Based on: Preprocessing_Diagnostic_Report.md
"""

import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import os
import sys


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

    # Check if file exists
    if not os.path.exists(file_path):
        # Try with different extensions
        for ext in ['.hdr', '.img', '.bip']:
            test_path = file_path.replace('.hdr', ext).replace('.img', ext).replace('.bip', ext)
            if os.path.exists(test_path):
                if ext == '.hdr':
                    file_path = test_path
                else:
                    file_path = test_path.replace(ext, '.hdr')
                break
        else:
            print(f"❌ ERROR: File not found: {file_path}")
            return None, None

    # Load data
    try:
        img = envi.open(file_path)
        cube = img.load()
        metadata = img.metadata
    except Exception as e:
        print(f"❌ ERROR: Could not load file: {e}")
        return None, None

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
        elif max_val > 100:
            print("❌ FAIL: Values > 100 (likely integer reflectance with scale factor)")
            print(f"   → Current max: {max_val:.2f}")
            print("   → Check if scale factor needs to be applied")
            print("   → If scale factor = 100, reflectance should be in range 0-100")
            print("   → For SAM, you need 0-1 scale (divide by 100)")
        elif max_val > 2.0:
            print("❌ FAIL: Values > 2.0 (unusual scale)")
            print(f"   → Current max: {max_val:.2f}")
            print("   → Should be 0-1 for reflectance")
        elif max_val > 1.2:
            print("⚠️  WARNING: Values > 1.2 (unusual but possible)")
            print(f"   → Current max: {max_val:.2f}")
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
        wavelengths = np.array([float(w) for w in metadata['wavelength']])
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
        wavelengths = None

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
        if wavelengths is not None:
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
        try:
            plt.figure(figsize=(12, 6))
            if wavelengths is not None:
                plt.plot(wavelengths, sample_pixel)
                plt.xlabel('Wavelength (nm)')
            else:
                plt.plot(sample_pixel)
                plt.xlabel('Band number')
            plt.ylabel('Reflectance')
            plt.title(f'Sample Pixel Spectrum (row={mid_row}, col={mid_col})')
            plt.grid(True, alpha=0.3)

            output_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else '.'
            output_file = os.path.join(output_dir, 'preprocessing_validation_spectrum.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\n  Saved spectrum plot: {output_file}")
            plt.close()
        except Exception as e:
            print(f"  Warning: Could not save spectrum plot: {e}")
    else:
        print("❌ Sample pixel is all zeros (invalid)")

    # Section 8: Scale Factor Check
    print("\n[8] SCALE FACTOR CHECK")
    print("-"*70)

    if 'scale factor' in metadata:
        scale_factor_str = metadata['scale factor']
        if isinstance(scale_factor_str, list):
            scale_factor = float(scale_factor_str[0])
        else:
            scale_factor = float(scale_factor_str)

        print(f"Scale factor in metadata: {scale_factor}")

        # Check if it's been applied
        if scale_factor != 1.0:
            max_expected_scaled = scale_factor * 1.0  # Expected max if reflectance is 0-1
            actual_max = np.max(cube) if len(cube[cube > 0]) > 0 else 0

            print(f"   Expected max (if 0-1 reflectance scaled): ~{max_expected_scaled:.1f}")
            print(f"   Actual max in data: {actual_max:.2f}")

            # Check if data is scaled
            if actual_max > max_expected_scaled * 0.5:
                print(f"⚠️  WARNING: Data appears to be SCALED by {scale_factor}")
                print(f"   → For SAM, you must DIVIDE by {scale_factor} to get 0-1 reflectance")
                print(f"   → Current range: 0-{actual_max:.1f}")
                print(f"   → After dividing by {scale_factor}: 0-{actual_max/scale_factor:.3f}")
            else:
                print("✓ Data appears to already be in 0-1 scale")
    else:
        print("No scale factor in metadata")
        print("⚠️  Data may already be in 0-1 scale, or scale not documented")

    # Final Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print("\nChecklist:")

    # Compile results
    checks = []

    # Scale check
    if len(valid_data) > 0:
        max_val = np.max(valid_data)
        if max_val <= 1.2:
            checks.append("✓ Reflectance scale correct (0-1)")
        elif max_val <= 100 and 'scale factor' in metadata:
            checks.append("⚠️  Reflectance SCALED - divide by scale factor for SAM")
        else:
            checks.append("❌ Reflectance scale INCORRECT")
    else:
        checks.append("❌ No valid data")

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
    print("\n🔍 KEY FINDING FOR SAM:")
    print("="*70)

    if 'scale factor' in metadata and len(valid_data) > 0:
        scale_factor_str = metadata['scale factor']
        if isinstance(scale_factor_str, list):
            scale_factor = float(scale_factor_str[0])
        else:
            scale_factor = float(scale_factor_str)

        actual_max = np.max(valid_data)

        if scale_factor == 100 and actual_max > 50:
            print("❗ IMPORTANT: Your reflectance data is scaled by 100")
            print(f"   Current range: 0-{actual_max:.1f}")
            print(f"   SAM expects: 0-1")
            print(f"   SOLUTION: Divide your data by {scale_factor} before using SAM")
            print("\n   In your SAM script, add:")
            print("   reflectance_cube = reflectance_cube / 100.0")
            print("   endmembers = endmembers / 100.0")

    print("="*70)

    return cube, metadata


def quick_diagnostic(file_path):
    """
    Quick 3-step diagnostic for immediate feedback
    """
    print("="*70)
    print("QUICK DIAGNOSTIC")
    print("="*70)

    try:
        img = envi.open(file_path)
        cube = img.load()
        metadata = img.metadata
    except Exception as e:
        print(f"❌ ERROR: Could not load file: {e}")
        return

    # Quick check #1: Is it reflectance?
    max_val = np.max(cube[cube > 0]) if len(cube[cube > 0]) > 0 else 0
    print(f"\n[1] Max reflectance: {max_val:.4f}")
    if max_val > 100:
        print("    ❌ ERROR: Not reflectance scale! Values > 100")
    elif max_val > 2.0:
        print(f"    ❌ ERROR: Values too high (max={max_val:.2f})")
        if 'scale factor' in metadata:
            print(f"    → Scale factor found: {metadata['scale factor']}")
    else:
        print("    ✓ Scale looks reasonable")

    # Quick check #2: Are there valid pixels?
    valid_pct = np.sum(np.any(cube > 0, axis=2)) / (cube.shape[0] * cube.shape[1]) * 100
    print(f"\n[2] Valid pixels: {valid_pct:.1f}%")
    if valid_pct < 10:
        print("    ❌ ERROR: Very few valid pixels")
    else:
        print("    ✓ Sufficient valid pixels")

    # Quick check #3: Scale factor
    if 'scale factor' in metadata:
        print(f"\n[3] Scale factor: {metadata['scale factor']}")
        print("    ⚠️  Data is SCALED - must divide by scale factor for SAM")
    else:
        print("\n[3] No scale factor in metadata")
        print("    Data may already be in 0-1 scale")

    print("\n" + "="*70)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validate_preprocessing.py <path_to_hdr_file>")
        print("\nExample:")
        print("  python validate_preprocessing.py C:/path/to/reflectance.hdr")
        print("\nOr for quick check:")
        print("  python validate_preprocessing.py <path_to_hdr_file> --quick")
        sys.exit(1)

    file_path = sys.argv[1]

    if len(sys.argv) > 2 and sys.argv[2] == '--quick':
        quick_diagnostic(file_path)
    else:
        validate_preprocessing(file_path)
