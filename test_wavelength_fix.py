"""
Test script to add wavelengths to existing HDR file
"""
import sys
sys.path.insert(0, 'c:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/SUREHYP')

from process_hyperion import fix_envi_hdr_for_snap

# Paths
hdr_file = "C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/OUT/EO1H2020342016359110KF_reflectance.hdr"
wavelength_file = "C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/OUT/EO1H2020342016359110KF_reflectance_spectral_info.txt"

print("Testing wavelength addition to HDR file...")
print("HDR file: {}".format(hdr_file))
print("Wavelength file: {}".format(wavelength_file))
print("-" * 60)

# Apply the fix with wavelength file and keep_wavelength=True
fix_envi_hdr_for_snap(
    hdr_path=hdr_file,
    wavelength_file=wavelength_file,
    keep_wavelength=True
)

print("\nDone! Check the HDR file to see if wavelengths were added.")
print("The original file is backed up as .hdr.backup")
