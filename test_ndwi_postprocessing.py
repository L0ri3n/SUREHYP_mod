"""
Test script to generate NDWI for the second image using the fixed plot_sample_spectra function
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from process_hyperion import post_processing
import spectral.io.envi as envi

# Configuration
base_path = 'C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/'
output_path = base_path + 'OUT/'
fname = 'EO1H2020342016359110KF'
reflectance_path = output_path + fname + '_reflectance'

print(f'Testing post-processing for: {fname}')
print(f'Loading reflectance from: {reflectance_path}')

# Load reflectance
if os.path.exists(reflectance_path + '.img'):
    img = envi.open(reflectance_path + '.hdr', reflectance_path + '.img')
else:
    img = envi.open(reflectance_path + '.hdr', reflectance_path + '.bip')

R = img.load()

# Get wavelengths
if 'wavelength' in img.metadata:
    bands = np.array([float(w) for w in img.metadata['wavelength']])
else:
    spectral_info_path = reflectance_path + '_spectral_info.txt'
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

print(f'Reflectance shape: {R.shape}')
print(f'Number of bands: {len(bands)}')

# Run post-processing
post_processing(R, bands, output_path, fname)

print('\nPost-processing completed successfully!')
