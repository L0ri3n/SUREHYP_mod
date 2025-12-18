"""
Multi-temporal Inundation Mapping from Hyperion NDWI
======================================================

This script processes two Hyperion images from different time periods,
computes NDWI for each, performs spatial alignment, and generates an
inundation map showing areas with increased water presence.

Author: Generated with Claude Code
Date: 2025-12-18
"""

import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import spectral.io.envi as envi
from scipy import ndimage
from osgeo import gdal, osr
import json

# Import the process_hyperion module functions
import sys
sys.path.insert(0, os.path.dirname(__file__))
from process_hyperion import compute_ndwi


def load_reflectance_and_bands(reflectance_path):
    """
    Load reflectance image and wavelength bands from ENVI format.

    Parameters:
    -----------
    reflectance_path : str
        Path to reflectance file (without extension)

    Returns:
    --------
    R : numpy.ndarray
        Reflectance array (rows, cols, bands)
    bands : numpy.ndarray
        Wavelengths in nm
    geotransform : tuple
        GDAL geotransform (for georeferencing)
    projection : str
        WKT projection string
    """
    print(f'\n  Loading: {os.path.basename(reflectance_path)}')

    # Determine file format
    if os.path.exists(reflectance_path + '.img'):
        hdr_path = reflectance_path + '.hdr'
        img_path = reflectance_path + '.img'
    elif os.path.exists(reflectance_path + '.bip'):
        hdr_path = reflectance_path + '.hdr'
        img_path = reflectance_path + '.bip'
    else:
        raise FileNotFoundError(f"No .img or .bip file found at {reflectance_path}")

    # Load with spectral library
    img = envi.open(hdr_path, img_path)
    R = img.load()

    # Get wavelengths
    if 'wavelength' in img.metadata:
        bands = np.array([float(w) for w in img.metadata['wavelength']])
    else:
        # Load from spectral info file
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
            print(f'    Loaded wavelengths from: {spectral_info_path}')
        else:
            raise ValueError(f"No wavelength data found in HDR or spectral info file")

    # Get geospatial information using GDAL
    ds = gdal.Open(img_path)
    if ds is not None:
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()
        ds = None
    else:
        print('    Warning: Could not load geospatial metadata, using default')
        geotransform = None
        projection = None

    print(f'    Shape: {R.shape}, Bands: {len(bands)}')
    print(f'    Wavelength range: {bands.min():.1f} - {bands.max():.1f} nm')

    return R, bands, geotransform, projection


def compute_ndwi_from_reflectance(R, bands, image_name):
    """
    Compute NDWI from reflectance data.

    Parameters:
    -----------
    R : numpy.ndarray
        Reflectance array (rows, cols, bands)
    bands : numpy.ndarray
        Wavelengths in nm
    image_name : str
        Name for logging

    Returns:
    --------
    ndwi : numpy.ndarray
        NDWI image (2D array)
    """
    print(f'\n  Computing NDWI for {image_name}...')
    ndwi = compute_ndwi(R, bands)
    # Ensure 2D output
    if ndwi.ndim > 2:
        ndwi = np.squeeze(ndwi)
    print(f'    NDWI range: {ndwi.min():.3f} to {ndwi.max():.3f}')
    print(f'    NDWI mean: {ndwi.mean():.3f}')
    print(f'    NDWI shape: {ndwi.shape}')
    return ndwi


def align_images_spatially(img1, gt1, img2, gt2):
    """
    Align two images spatially based on their geotransforms.
    Extracts the overlapping region from both images.

    Parameters:
    -----------
    img1 : numpy.ndarray
        First image array
    gt1 : tuple
        Geotransform for first image (x_origin, pixel_width, 0, y_origin, 0, pixel_height)
    img2 : numpy.ndarray
        Second image array
    gt2 : tuple
        Geotransform for second image

    Returns:
    --------
    img1_aligned : numpy.ndarray
        Subset of img1 in overlapping region
    img2_aligned : numpy.ndarray
        Subset of img2 in overlapping region
    overlap_geotransform : tuple
        Geotransform for the overlapping region
    """
    print('\n  Performing spatial alignment...')

    if gt1 is None or gt2 is None:
        print('    Warning: No geotransform available, assuming images are already aligned')
        # Simply crop to minimum dimensions
        min_rows = min(img1.shape[0], img2.shape[0])
        min_cols = min(img1.shape[1], img2.shape[1])
        return img1[:min_rows, :min_cols], img2[:min_rows, :min_cols], gt1

    # Extract geotransform parameters
    x1_origin, pixel_width1, _, y1_origin, _, pixel_height1 = gt1
    x2_origin, pixel_width2, _, y2_origin, _, pixel_height2 = gt2

    # Check if pixel sizes match
    if not (np.isclose(pixel_width1, pixel_width2, rtol=1e-6) and
            np.isclose(pixel_height1, pixel_height2, rtol=1e-6)):
        print(f'    Warning: Different pixel sizes detected!')
        print(f'      Image 1: {pixel_width1:.6f} x {pixel_height1:.6f}')
        print(f'      Image 2: {pixel_width2:.6f} x {pixel_height2:.6f}')
        print(f'    Proceeding with image 1 pixel size as reference')

    # Calculate bounds for each image
    x1_max = x1_origin + pixel_width1 * img1.shape[1]
    y1_min = y1_origin + pixel_height1 * img1.shape[0]  # pixel_height is negative

    x2_max = x2_origin + pixel_width2 * img2.shape[1]
    y2_min = y2_origin + pixel_height2 * img2.shape[0]

    # Calculate overlapping bounds
    overlap_x_min = max(x1_origin, x2_origin)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_max = min(y1_origin, y2_origin)
    overlap_y_min = max(y1_min, y2_min)

    print(f'    Image 1 bounds: X[{x1_origin:.2f}, {x1_max:.2f}], Y[{y1_min:.2f}, {y1_origin:.2f}]')
    print(f'    Image 2 bounds: X[{x2_origin:.2f}, {x2_max:.2f}], Y[{y2_min:.2f}, {y2_origin:.2f}]')
    print(f'    Overlap bounds: X[{overlap_x_min:.2f}, {overlap_x_max:.2f}], Y[{overlap_y_min:.2f}, {overlap_y_max:.2f}]')

    # Check if there's valid overlap
    if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
        raise ValueError('No spatial overlap between images!')

    # Calculate pixel indices for overlapping region in each image
    col1_start = int((overlap_x_min - x1_origin) / pixel_width1)
    col1_end = int((overlap_x_max - x1_origin) / pixel_width1)
    row1_start = int((overlap_y_max - y1_origin) / pixel_height1)
    row1_end = int((overlap_y_min - y1_origin) / pixel_height1)

    col2_start = int((overlap_x_min - x2_origin) / pixel_width2)
    col2_end = int((overlap_x_max - x2_origin) / pixel_width2)
    row2_start = int((overlap_y_max - y2_origin) / pixel_height2)
    row2_end = int((overlap_y_min - y2_origin) / pixel_height2)

    # Ensure indices are within bounds
    col1_start = max(0, min(col1_start, img1.shape[1]))
    col1_end = max(0, min(col1_end, img1.shape[1]))
    row1_start = max(0, min(row1_start, img1.shape[0]))
    row1_end = max(0, min(row1_end, img1.shape[0]))

    col2_start = max(0, min(col2_start, img2.shape[1]))
    col2_end = max(0, min(col2_end, img2.shape[1]))
    row2_start = max(0, min(row2_start, img2.shape[0]))
    row2_end = max(0, min(row2_end, img2.shape[0]))

    # Extract overlapping regions
    img1_aligned = img1[row1_start:row1_end, col1_start:col1_end]
    img2_aligned = img2[row2_start:row2_end, col2_start:col2_end]

    print(f'    Image 1 subset: rows[{row1_start}:{row1_end}], cols[{col1_start}:{col1_end}]')
    print(f'    Image 2 subset: rows[{row2_start}:{row2_end}], cols[{col2_start}:{col2_end}]')
    print(f'    Image 1 aligned shape: {img1_aligned.shape}')
    print(f'    Image 2 aligned shape: {img2_aligned.shape}')

    # Ensure exact same dimensions (crop to minimum due to rounding)
    min_rows = min(img1_aligned.shape[0], img2_aligned.shape[0])
    min_cols = min(img1_aligned.shape[1], img2_aligned.shape[1])

    if img1_aligned.shape[0] != min_rows or img1_aligned.shape[1] != min_cols:
        img1_aligned = img1_aligned[:min_rows, :min_cols]
    if img2_aligned.shape[0] != min_rows or img2_aligned.shape[1] != min_cols:
        img2_aligned = img2_aligned[:min_rows, :min_cols]

    print(f'    Final aligned shape: {img1_aligned.shape}')

    # Create geotransform for overlapping region
    overlap_geotransform = (overlap_x_min, pixel_width1, 0, overlap_y_max, 0, pixel_height1)

    return img1_aligned, img2_aligned, overlap_geotransform


def compute_inundation_map(ndwi_early, ndwi_late, valid_mask_early=None, valid_mask_late=None, threshold=0.1):
    """
    Compute inundation map from two NDWI images.

    Inundation is detected where NDWI increases significantly between
    the early and late dates, indicating increased water presence.

    Parameters:
    -----------
    ndwi_early : numpy.ndarray
        NDWI from earlier date
    ndwi_late : numpy.ndarray
        NDWI from later date
    valid_mask_early : numpy.ndarray, optional
        Valid pixel mask for early image
    valid_mask_late : numpy.ndarray, optional
        Valid pixel mask for late image
    threshold : float
        Minimum NDWI increase to classify as inundation (default: 0.1)

    Returns:
    --------
    inundation_map : numpy.ndarray
        Binary map where 1 = inundated, 0 = not inundated
    ndwi_change : numpy.ndarray
        Continuous NDWI change (late - early)
    combined_valid_mask : numpy.ndarray
        Combined valid pixel mask
    """
    print(f'\n  Computing inundation map...')
    print(f'    Threshold: NDWI increase > {threshold}')

    # Ensure same shape
    if ndwi_early.shape != ndwi_late.shape:
        raise ValueError(f'NDWI images have different shapes: {ndwi_early.shape} vs {ndwi_late.shape}')

    # Compute NDWI change
    ndwi_change = ndwi_late - ndwi_early

    # Create combined valid mask
    if valid_mask_early is not None and valid_mask_late is not None:
        combined_valid_mask = valid_mask_early & valid_mask_late
    elif valid_mask_early is not None:
        combined_valid_mask = valid_mask_early
    elif valid_mask_late is not None:
        combined_valid_mask = valid_mask_late
    else:
        # Create mask based on valid data
        combined_valid_mask = np.isfinite(ndwi_early) & np.isfinite(ndwi_late)

    # Compute inundation: positive NDWI change above threshold
    inundation_map = np.zeros_like(ndwi_change, dtype=np.uint8)
    inundation_map[(ndwi_change > threshold) & combined_valid_mask] = 1

    # Statistics
    n_total = combined_valid_mask.sum()
    n_inundated = inundation_map.sum()
    pct_inundated = 100 * n_inundated / n_total if n_total > 0 else 0

    valid_change = ndwi_change[combined_valid_mask]

    print(f'    NDWI change statistics (valid pixels):')
    print(f'      Min: {valid_change.min():.3f}')
    print(f'      Max: {valid_change.max():.3f}')
    print(f'      Mean: {valid_change.mean():.3f}')
    print(f'      Std: {valid_change.std():.3f}')
    print(f'    Inundated pixels: {n_inundated} / {n_total} ({pct_inundated:.2f}%)')

    return inundation_map, ndwi_change, combined_valid_mask


def save_inundation_results(inundation_map, ndwi_change, ndwi_early, ndwi_late,
                            output_dir, filename_base, geotransform=None, projection=None):
    """
    Save inundation map and related products as GeoTIFF and numpy arrays.

    Parameters:
    -----------
    inundation_map : numpy.ndarray
        Binary inundation map
    ndwi_change : numpy.ndarray
        NDWI change array
    ndwi_early : numpy.ndarray
        Early NDWI
    ndwi_late : numpy.ndarray
        Late NDWI
    output_dir : str
        Output directory
    filename_base : str
        Base filename for outputs
    geotransform : tuple, optional
        GDAL geotransform
    projection : str, optional
        WKT projection string
    """
    print(f'\n  Saving inundation results to: {output_dir}')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save as numpy arrays
    np.save(output_dir + filename_base + '_inundation_map.npy', inundation_map)
    np.save(output_dir + filename_base + '_ndwi_change.npy', ndwi_change)
    np.save(output_dir + filename_base + '_ndwi_early.npy', ndwi_early)
    np.save(output_dir + filename_base + '_ndwi_late.npy', ndwi_late)

    print(f'    Saved numpy arrays')

    # Save as GeoTIFF if geospatial info available
    if geotransform is not None and projection is not None:
        def save_geotiff(array, filepath, dtype=gdal.GDT_Float32):
            driver = gdal.GetDriverByName('GTiff')
            rows, cols = array.shape
            ds = driver.Create(filepath, cols, rows, 1, dtype,
                             options=['COMPRESS=LZW', 'TILED=YES'])
            ds.SetGeoTransform(geotransform)
            ds.SetProjection(projection)
            band = ds.GetRasterBand(1)
            band.WriteArray(array)
            band.SetNoDataValue(-9999)
            band.FlushCache()
            ds = None

        save_geotiff(inundation_map, output_dir + filename_base + '_inundation_map.tif', gdal.GDT_Byte)
        save_geotiff(ndwi_change, output_dir + filename_base + '_ndwi_change.tif', gdal.GDT_Float32)
        save_geotiff(ndwi_early, output_dir + filename_base + '_ndwi_early.tif', gdal.GDT_Float32)
        save_geotiff(ndwi_late, output_dir + filename_base + '_ndwi_late.tif', gdal.GDT_Float32)

        print(f'    Saved GeoTIFF files')


def visualize_inundation(inundation_map, ndwi_change, ndwi_early, ndwi_late,
                        output_dir, filename_base, date_early, date_late):
    """
    Create comprehensive visualizations of inundation analysis.

    Parameters:
    -----------
    inundation_map : numpy.ndarray
        Binary inundation map
    ndwi_change : numpy.ndarray
        NDWI change array
    ndwi_early : numpy.ndarray
        Early NDWI
    ndwi_late : numpy.ndarray
        Late NDWI
    output_dir : str
        Output directory for plots
    filename_base : str
        Base filename for plots
    date_early : str
        Date label for early image
    date_late : str
        Date label for late image
    """
    print(f'\n  Creating visualizations...')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Four-panel comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Early NDWI
    im1 = axes[0, 0].imshow(ndwi_early, cmap='Blues', vmin=-0.5, vmax=0.5)
    axes[0, 0].set_title(f'NDWI - {date_early}', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], label='NDWI', shrink=0.8)

    # Late NDWI
    im2 = axes[0, 1].imshow(ndwi_late, cmap='Blues', vmin=-0.5, vmax=0.5)
    axes[0, 1].set_title(f'NDWI - {date_late}', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], label='NDWI', shrink=0.8)

    # NDWI Change
    im3 = axes[1, 0].imshow(ndwi_change, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    axes[1, 0].set_title(f'NDWI Change ({date_late} - {date_early})', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[1, 0], label='NDWI Change', shrink=0.8)

    # Inundation Map
    cmap_inundation = LinearSegmentedColormap.from_list('inundation',
                                                         [(0.9, 0.9, 0.9), (0, 0.4, 0.8)])
    im4 = axes[1, 1].imshow(inundation_map, cmap=cmap_inundation, vmin=0, vmax=1)
    axes[1, 1].set_title('Inundation Map', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    cbar4 = plt.colorbar(im4, ax=axes[1, 1], label='Inundated', shrink=0.8, ticks=[0, 1])
    cbar4.set_ticklabels(['No', 'Yes'])

    plt.tight_layout()
    plt.savefig(output_dir + filename_base + '_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {filename_base}_comparison.png')

    # 2. Inundation map overlay
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Show late NDWI as background
    ax.imshow(ndwi_late, cmap='gray', vmin=-0.5, vmax=0.5, alpha=0.6)

    # Overlay inundation in red
    inundation_overlay = np.ma.masked_where(inundation_map == 0, inundation_map)
    ax.imshow(inundation_overlay, cmap='Reds', vmin=0, vmax=1, alpha=0.7)

    ax.set_title(f'Inundation Detection\n{date_early} to {date_late}',
                fontsize=16, fontweight='bold')
    ax.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Inundated areas'),
        Patch(facecolor='gray', alpha=0.6, label='Background (NDWI ' + date_late + ')')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir + filename_base + '_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {filename_base}_overlay.png')

    # 3. Histogram of NDWI changes
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    valid_change = ndwi_change[np.isfinite(ndwi_change)]
    ax.hist(valid_change, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=2, label='No change')
    ax.axvline(0.1, color='red', linestyle='--', linewidth=2, label='Inundation threshold')

    ax.set_xlabel('NDWI Change', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of NDWI Changes\n{date_early} to {date_late}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir + filename_base + '_histogram.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {filename_base}_histogram.png')


def process_image_pair_for_inundation(image1_id, image2_id, base_path, output_path,
                                      run_preprocessing=False, inundation_threshold=0.1):
    """
    Complete workflow for inundation mapping from two Hyperion images.

    Parameters:
    -----------
    image1_id : str
        ID of the first (earlier) image
    image2_id : str
        ID of the second (later) image
    base_path : str
        Base path to data directories
    output_path : str
        Path for output products
    run_preprocessing : bool
        Whether to run preprocessing (if False, loads existing reflectance)
    inundation_threshold : float
        NDWI increase threshold for inundation detection

    Returns:
    --------
    results : dict
        Dictionary containing inundation map and statistics
    """
    print('\n' + '=' * 70)
    print('MULTI-TEMPORAL INUNDATION MAPPING')
    print('=' * 70)
    print(f'Early image: {image1_id}')
    print(f'Late image:  {image2_id}')
    print('=' * 70)

    # Extract dates from image IDs (format: EO1H2020342013284110KF -> 2013284)
    date1_str = image1_id[9:16]  # year + day of year
    date2_str = image2_id[9:16]

    year1, doy1 = date1_str[:4], date1_str[4:]
    year2, doy2 = date2_str[:4], date2_str[4:]

    # Paths
    reflectance1_path = output_path + image1_id + '_reflectance'
    reflectance2_path = output_path + image2_id + '_reflectance'

    # Step 1: Load or process images
    print('\n[STEP 1/5] Loading reflectance images...')

    if not (os.path.exists(reflectance1_path + '.img') or os.path.exists(reflectance1_path + '.bip')):
        if run_preprocessing:
            raise NotImplementedError('Preprocessing not yet implemented in this function. '
                                     'Please run process_hyperion.py first for both images.')
        else:
            raise FileNotFoundError(f'Reflectance file not found: {reflectance1_path}')

    if not (os.path.exists(reflectance2_path + '.img') or os.path.exists(reflectance2_path + '.bip')):
        if run_preprocessing:
            raise NotImplementedError('Preprocessing not yet implemented in this function. '
                                     'Please run process_hyperion.py first for both images.')
        else:
            raise FileNotFoundError(f'Reflectance file not found: {reflectance2_path}')

    # Load reflectance data
    R1, bands1, gt1, proj1 = load_reflectance_and_bands(reflectance1_path)
    R2, bands2, gt2, proj2 = load_reflectance_and_bands(reflectance2_path)

    # Step 2: Compute NDWI for both images
    print('\n[STEP 2/5] Computing NDWI...')
    ndwi1 = compute_ndwi_from_reflectance(R1, bands1, image1_id)
    ndwi2 = compute_ndwi_from_reflectance(R2, bands2, image2_id)

    # Load valid masks if available
    mask1_path = output_path + image1_id + '_reflectance_clearview_mask.npy'
    mask2_path = output_path + image2_id + '_reflectance_clearview_mask.npy'

    valid_mask1 = np.load(mask1_path) if os.path.exists(mask1_path) else None
    valid_mask2 = np.load(mask2_path) if os.path.exists(mask2_path) else None

    # Step 3: Spatial alignment
    print('\n[STEP 3/5] Aligning images spatially...')
    ndwi1_aligned, ndwi2_aligned, overlap_gt = align_images_spatially(
        ndwi1, gt1, ndwi2, gt2
    )

    # Also align masks
    if valid_mask1 is not None:
        valid_mask1_aligned, _, _ = align_images_spatially(valid_mask1, gt1, valid_mask2, gt2)
    else:
        valid_mask1_aligned = None

    if valid_mask2 is not None:
        _, valid_mask2_aligned, _ = align_images_spatially(valid_mask1, gt1, valid_mask2, gt2)
    else:
        valid_mask2_aligned = None

    # Step 4: Compute inundation map
    print('\n[STEP 4/5] Computing inundation map...')
    inundation_map, ndwi_change, combined_mask = compute_inundation_map(
        ndwi1_aligned, ndwi2_aligned,
        valid_mask1_aligned, valid_mask2_aligned,
        threshold=inundation_threshold
    )

    # Step 5: Save and visualize results
    print('\n[STEP 5/5] Saving results and creating visualizations...')

    output_name = f'{image1_id}_to_{image2_id}_inundation'
    inundation_dir = output_path + 'inundation/'

    save_inundation_results(
        inundation_map, ndwi_change, ndwi1_aligned, ndwi2_aligned,
        inundation_dir, output_name, overlap_gt, proj1
    )

    date_label1 = f'{year1} (DOY {doy1})'
    date_label2 = f'{year2} (DOY {doy2})'

    visualize_inundation(
        inundation_map, ndwi_change, ndwi1_aligned, ndwi2_aligned,
        inundation_dir, output_name, date_label1, date_label2
    )

    # Create summary statistics
    stats = {
        'image1_id': image1_id,
        'image2_id': image2_id,
        'date1': date_label1,
        'date2': date_label2,
        'aligned_shape': ndwi1_aligned.shape,
        'n_valid_pixels': int(combined_mask.sum()),
        'n_inundated_pixels': int(inundation_map.sum()),
        'percent_inundated': float(100 * inundation_map.sum() / combined_mask.sum()),
        'ndwi_change_mean': float(ndwi_change[combined_mask].mean()),
        'ndwi_change_std': float(ndwi_change[combined_mask].std()),
        'ndwi_change_min': float(ndwi_change[combined_mask].min()),
        'ndwi_change_max': float(ndwi_change[combined_mask].max()),
        'threshold': inundation_threshold
    }

    # Save statistics as JSON
    with open(inundation_dir + output_name + '_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'\n    Statistics saved to: {output_name}_statistics.json')

    print('\n' + '=' * 70)
    print('INUNDATION MAPPING COMPLETE!')
    print('=' * 70)
    print(f'\nResults saved to: {inundation_dir}')
    print(f'  - Inundation map: {output_name}_inundation_map.tif')
    print(f'  - NDWI change:    {output_name}_ndwi_change.tif')
    print(f'  - Visualizations: {output_name}_comparison.png')
    print(f'  - Statistics:     {output_name}_statistics.json')

    return {
        'inundation_map': inundation_map,
        'ndwi_change': ndwi_change,
        'ndwi_early': ndwi1_aligned,
        'ndwi_late': ndwi2_aligned,
        'statistics': stats
    }


if __name__ == '__main__':

    # ============================================================
    # CONFIGURATION
    # ============================================================

    base_path = 'C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/'
    output_path = base_path + 'OUT/'

    # Image IDs (chronological order: earlier first, later second)
    image1_id = 'EO1H2020342013284110KF'  # 2013, day 284 (earlier)
    image2_id = 'EO1H2020342016359110KF'  # 2016, day 359 (later)

    # Processing options
    run_preprocessing = False  # Set to True to run full preprocessing (not yet implemented)
    inundation_threshold = 0.1  # NDWI increase threshold for inundation detection

    # ============================================================
    # RUN INUNDATION MAPPING
    # ============================================================

    results = process_image_pair_for_inundation(
        image1_id=image1_id,
        image2_id=image2_id,
        base_path=base_path,
        output_path=output_path,
        run_preprocessing=run_preprocessing,
        inundation_threshold=inundation_threshold
    )

    print('\nInundation mapping completed successfully!')
    print(f'Inundated area: {results["statistics"]["percent_inundated"]:.2f}% of valid pixels')
