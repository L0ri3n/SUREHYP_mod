"""
Helper functions for loading Hyperion data for Spectral Angle Mapper (SAM)

CRITICAL: The preprocessed reflectance data is scaled by 100 and stored as uint16.
This script provides helper functions to properly load and validate data for SAM.
"""

import numpy as np
import spectral.io.envi as envi


def load_reflectance_for_sam(hdr_path):
    """
    Load reflectance data and automatically apply scale factor for SAM.

    Parameters:
    -----------
    hdr_path : str
        Path to the .hdr file

    Returns:
    --------
    cube : numpy.ndarray
        Reflectance cube in 0-1 range (float32)
    wavelengths : numpy.ndarray
        Wavelengths in nm
    metadata : dict
        ENVI metadata
    """
    print('Loading reflectance data for SAM...')

    # Load image
    img = envi.open(hdr_path)
    cube = img.load()
    metadata = img.metadata

    print('  Data shape: {}'.format(cube.shape))
    print('  Data type (raw): {}'.format(cube.dtype))
    print('  Value range (raw): {} to {}'.format(np.min(cube[cube>0]), np.max(cube)))

    # Check for scale factor
    if 'scale factor' in metadata:
        scale_factor_list = metadata['scale factor']
        if isinstance(scale_factor_list, list):
            scale_factor = float(scale_factor_list[0])
        else:
            scale_factor = float(scale_factor_list)

        print('  Scale factor found: {}'.format(scale_factor))
        print('  Applying scale factor...')

        # Apply scale factor
        cube = cube.astype(np.float32) / scale_factor

        print('  Value range (scaled): {:.4f} to {:.4f}'.format(
            np.min(cube[cube>0]), np.max(cube)))
    else:
        print('  WARNING: No scale factor in metadata')
        print('  Assuming data is already in 0-1 range')
        cube = cube.astype(np.float32)

    # Get wavelengths
    if 'wavelength' in metadata:
        wavelengths = np.array([float(w) for w in metadata['wavelength']])
    else:
        print('  WARNING: No wavelength information in metadata')
        wavelengths = None

    # Validate
    validate_reflectance_scale(cube, "Loaded cube")

    return cube, wavelengths, metadata


def validate_reflectance_scale(data, name="data"):
    """
    Check if reflectance is in correct 0-1 scale for SAM.

    Parameters:
    -----------
    data : numpy.ndarray
        Reflectance data to check
    name : str
        Name of the data (for error messages)
    """
    valid_data = data[data > 0]

    if len(valid_data) == 0:
        print('  ERROR: No valid data (all zeros)')
        return False

    max_val = np.max(valid_data)
    min_val = np.min(valid_data)

    print('  Validating {}: range {:.4f} to {:.4f}'.format(name, min_val, max_val))

    if max_val > 2.0:
        print('  ERROR: {} appears to be SCALED (max={:.2f})'.format(name, max_val))
        print('         Expected 0-1 range for SAM')
        print('         Did you forget to divide by the scale factor?')
        return False
    elif max_val > 1.2:
        print('  WARNING: {} has max={:.2f} (>1.0). Unusual but proceeding.'.format(
            name, max_val))
        return True
    else:
        print('  OK: {} in valid range for SAM'.format(name))
        return True


def test_sam_computation(cube, endmember, pixel_coords=(100, 100)):
    """
    Test SAM computation on a single pixel to verify everything is correct.

    Parameters:
    -----------
    cube : numpy.ndarray
        Reflectance cube (rows, cols, bands)
    endmember : numpy.ndarray
        Endmember spectrum (bands,)
    pixel_coords : tuple
        (row, col) coordinates to test

    Returns:
    --------
    angle : float
        SAM angle in degrees
    """
    row, col = pixel_coords

    # Get pixel spectrum
    pixel = cube[row, col, :]

    # Check if pixel is valid
    if not np.any(pixel > 0):
        print('  ERROR: Test pixel is all zeros')
        return None

    # Compute cosine similarity
    dot_product = np.dot(pixel, endmember)
    norm_pixel = np.linalg.norm(pixel)
    norm_endmember = np.linalg.norm(endmember)

    cos_angle = dot_product / (norm_pixel * norm_endmember)

    print('  Test SAM computation at pixel ({}, {}):'.format(row, col))
    print('    Dot product: {:.6f}'.format(dot_product))
    print('    Norm (pixel): {:.6f}'.format(norm_pixel))
    print('    Norm (endmember): {:.6f}'.format(norm_endmember))
    print('    Cosine: {:.6f}'.format(cos_angle))

    # Check if cosine is valid
    if abs(cos_angle) > 1.0:
        print('    ERROR: Invalid cosine (|cos| > 1.0)!')
        print('           This indicates a SCALE MISMATCH between cube and endmember')
        return None

    # Compute angle
    angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
    angle_deg = np.degrees(angle_rad)

    print('    Angle: {:.2f} degrees'.format(angle_deg))

    if angle_deg < 90:
        print('    OK: Valid SAM angle')
    else:
        print('    WARNING: Large angle (>90 deg) - pixel may not match this endmember')

    return angle_deg


def compare_band_alignment(cube_wavelengths, endmember_wavelengths, tolerance=1.0):
    """
    Check if image and endmember wavelengths are aligned.

    Parameters:
    -----------
    cube_wavelengths : numpy.ndarray
        Image wavelengths in nm
    endmember_wavelengths : numpy.ndarray
        Endmember wavelengths in nm
    tolerance : float
        Maximum allowed difference in nm (default 1.0 nm)

    Returns:
    --------
    aligned : bool
        True if wavelengths are aligned
    """
    print('  Checking band alignment...')

    if len(cube_wavelengths) != len(endmember_wavelengths):
        print('    ERROR: Band count mismatch!')
        print('           Image: {} bands'.format(len(cube_wavelengths)))
        print('           Endmembers: {} bands'.format(len(endmember_wavelengths)))
        return False

    # Check wavelength differences
    diff = np.abs(cube_wavelengths - endmember_wavelengths)
    max_diff = np.max(diff)

    print('    Band count: {} (matching)'.format(len(cube_wavelengths)))
    print('    Max wavelength difference: {:.2f} nm'.format(max_diff))

    if max_diff > tolerance:
        print('    WARNING: Wavelengths differ by more than {} nm'.format(tolerance))
        print('             This may cause inaccurate SAM results')
        return False
    else:
        print('    OK: Wavelengths aligned within {} nm'.format(tolerance))
        return True


# Example usage
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sam_data_helper.py <path_to_reflectance.hdr>")
        print("\nThis script will:")
        print("  1. Load the reflectance data")
        print("  2. Apply the scale factor automatically")
        print("  3. Validate that the data is in the correct 0-1 range for SAM")
        sys.exit(1)

    hdr_path = sys.argv[1]

    # Load data
    cube, wavelengths, metadata = load_reflectance_for_sam(hdr_path)

    print("\n" + "="*70)
    print("Data successfully loaded and validated for SAM!")
    print("="*70)
    print("\nTo use in your SAM script:")
    print("  from sam_data_helper import load_reflectance_for_sam")
    print("  cube, wavelengths, metadata = load_reflectance_for_sam('path/to/file.hdr')")
    print("\nThe cube will be automatically scaled to 0-1 range.")
    print("="*70)
