"""
Check actual reflectance values in the file
Run: conda activate hyperion_roger && python check_reflectance_values.py
"""

import numpy as np
import spectral.io.envi as envi
import os

# Load the current reflectance file
base_path = r'C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT'
fname = 'EO1H2020342013284110KF'
path = os.path.join(base_path, fname + '_reflectance')

print("="*80)
print("CHECKING REFLECTANCE FILE VALUES")
print("="*80)
print(f"\nFile: {path}")

img = envi.open(path + '.hdr', path + '.img')
R = img.load()

print(f"\nRaw data type: {R.dtype}")
print(f"Raw data shape: {R.shape}")

# Check actual stored values (before any conversion)
print(f"\n--- RAW STORED VALUES (uint16) ---")
print(f"Min (all): {np.min(R)}")
print(f"Max (all): {np.max(R)}")
print(f"Min (>0): {np.min(R[R > 0])}")
print(f"Max (>0): {np.max(R[R > 0])}")
print(f"Mean (>0): {np.mean(R[R > 0]):.2f}")

# Check scale factor
print(f"\n--- SCALE FACTOR FROM METADATA ---")
if 'scale factor' in img.metadata:
    sf = img.metadata['scale factor']
    sf_val = sf[0] if isinstance(sf, list) else sf
    print(f"Scale factor: {sf_val}")

    # Apply scale factor
    R_scaled = R.astype(np.float32) / float(sf_val)
    print(f"\n--- AFTER SCALE CORRECTION (÷{sf_val}) ---")
    print(f"Min (>0): {np.min(R_scaled[R_scaled > 0]):.6f}")
    print(f"Max (>0): {np.max(R_scaled[R_scaled > 0]):.6f}")
    print(f"Mean (>0): {np.mean(R_scaled[R_scaled > 0]):.6f}")
else:
    print("No scale factor found!")
    R_scaled = R.astype(np.float32)

# Get band with max value
max_per_band = np.max(R, axis=(0,1))
top_5_idx = np.argsort(max_per_band)[-5:][::-1]

print(f"\n--- TOP 5 BANDS WITH HIGHEST RAW VALUES ---")
if 'wavelength' in img.metadata:
    wl = [float(w) for w in img.metadata['wavelength']]
    for idx in top_5_idx:
        print(f"Band {idx+1:3d} ({wl[idx]:7.2f} nm): raw={max_per_band[idx]:8.0f}, scaled={max_per_band[idx]/float(sf_val):.4f}")
else:
    for idx in top_5_idx:
        print(f"Band {idx+1:3d}: raw={max_per_band[idx]:8.0f}, scaled={max_per_band[idx]/float(sf_val):.4f}")

# Check specific problem bands
print(f"\n--- CHECKING SPECIFIC WAVELENGTHS ---")
if 'wavelength' in img.metadata:
    wl_array = np.array(wl)

    # Check 750nm (VNIR-SWIR transition)
    idx_750 = np.argmin(np.abs(wl_array - 750))
    print(f"~750nm (band {idx_750+1}, {wl[idx_750]:.2f}nm): raw={max_per_band[idx_750]:.0f}, scaled={max_per_band[idx_750]/float(sf_val):.4f}")

    # Check 1400nm (water vapor)
    idx_1400 = np.argmin(np.abs(wl_array - 1400))
    print(f"~1400nm (band {idx_1400+1}, {wl[idx_1400]:.2f}nm): raw={max_per_band[idx_1400]:.0f}, scaled={max_per_band[idx_1400]/float(sf_val):.4f}")

    # Check 2000nm (SWIR edge)
    if wl_array.max() >= 2000:
        idx_2000 = np.argmin(np.abs(wl_array - 2000))
        print(f"~2000nm (band {idx_2000+1}, {wl[idx_2000]:.2f}nm): raw={max_per_band[idx_2000]:.0f}, scaled={max_per_band[idx_2000]/float(sf_val):.4f}")

# Diagnosis
print(f"\n{'='*80}")
print("DIAGNOSIS")
print(f"{'='*80}")

max_scaled = np.max(R_scaled[R_scaled > 0])

if max_scaled > 1000:
    print(f"❌ CRITICAL: Max scaled value {max_scaled:.0f} >> 1.0")
    print(f"   The data appears to STILL have extreme values BEFORE scaling.")
    print(f"   This means the atmospheric correction produced incorrect reflectance.")
    print(f"\n   ROOT CAUSE: Bad bands not removed BEFORE atmospheric correction!")
    print(f"   SOLUTION: Reprocess from radiance, removing bad bands first.")
elif max_scaled > 2.0:
    print(f"❌ ERROR: Max scaled value {max_scaled:.2f} > 2.0")
    print(f"   Scale factor correction helped but values still too high.")
elif max_scaled < 0.001:
    print(f"❌ ERROR: Max scaled value {max_scaled:.6f} < 0.001")
    print(f"   Scale factor may have been applied twice.")
else:
    print(f"✅ SUCCESS: Max scaled value {max_scaled:.4f} in reasonable range!")
    print(f"   Reflectance data appears correct for SAM classification.")

print(f"\n{'='*80}")
