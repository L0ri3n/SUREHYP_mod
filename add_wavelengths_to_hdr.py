"""
Simple script to add wavelengths to an existing HDR file for SNAP compatibility
Works with Python 3.5+
"""
import os
import re

def add_wavelengths_to_hdr(hdr_path, wavelength_file):
    """Add wavelengths from external file to HDR"""

    if not os.path.exists(hdr_path):
        print("Error: HDR file not found: " + hdr_path)
        return

    if not os.path.exists(wavelength_file):
        print("Error: Wavelength file not found: " + wavelength_file)
        return

    # Read wavelength file
    print("Loading wavelengths from: " + wavelength_file)
    wavelengths = []
    fwhm_values = []

    with open(wavelength_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    wavelengths.append(parts[1].strip())
                    if len(parts) >= 3:
                        fwhm_values.append(parts[2].strip())

    print("Loaded {} wavelengths".format(len(wavelengths)))

    # Read HDR file
    with open(hdr_path, 'r') as f:
        content = f.read()

    # Create backup
    backup_path = hdr_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)
        print("Created backup: " + backup_path)

    # Remove existing wavelength and fwhm fields if present
    content = re.sub(r'\nwavelength\s*=\s*\{[^}]*\}', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\nfwhm\s*=\s*\{[^}]*\}', '', content, flags=re.IGNORECASE)

    # Add wavelength field
    wavelength_str = ' , '.join(wavelengths)

    # Find where to insert (after band names)
    if re.search(r'band names\s*=', content, re.IGNORECASE):
        content = re.sub(
            r'(band names\s*=\s*\{[^}]*\})',
            r'\1\nwavelength = { ' + wavelength_str + ' }',
            content,
            flags=re.IGNORECASE
        )
    else:
        # Insert after bands line
        content = re.sub(
            r'(bands\s*=\s*\d+)',
            r'\1\nwavelength = { ' + wavelength_str + ' }',
            content
        )

    # Add FWHM if available
    if fwhm_values:
        fwhm_str = ' , '.join(fwhm_values)
        content = re.sub(
            r'(wavelength\s*=\s*\{[^}]*\})',
            r'\1\nfwhm = { ' + fwhm_str + ' }',
            content,
            flags=re.IGNORECASE
        )

    # Write modified HDR
    with open(hdr_path, 'w') as f:
        f.write(content)

    print("Successfully added wavelengths to HDR file!")
    print("Wavelengths: {}, FWHM values: {}".format(len(wavelengths), len(fwhm_values)))

# Main execution
if __name__ == '__main__':
    hdr_file = "C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/OUT/EO1H2020342016359110KF_reflectance.hdr"
    wavelength_file = "C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/OUT/EO1H2020342016359110KF_reflectance_spectral_info.txt"

    print("=" * 60)
    print("Adding Wavelengths to ENVI HDR for SNAP")
    print("=" * 60)
    print("")

    add_wavelengths_to_hdr(hdr_file, wavelength_file)

    print("")
    print("=" * 60)
    print("Done! You can now open the file in SNAP.")
    print("The wavelength metadata should now appear in the properties.")
    print("=" * 60)
