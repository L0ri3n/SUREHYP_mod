"""
Configuration file for Hyperion processing and inundation mapping.

Edit this file to configure:
1. Which images to process
2. Processing parameters
3. Inundation analysis settings

All scripts will read from this central configuration.
"""

# ============================================================
# PATHS CONFIGURATION
# ============================================================

basePath = 'C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/'

# Path to USGS Hyperion metadata (optional, can be empty file)
pathToL1Rmetadata = basePath + 'METADATA/METADATA.csv'

# Path to L1R images (uncompressed folders)
pathToL1Rimages = basePath + 'L1R/'

# Path to L1T images (TIF files)
pathToL1Timages = basePath + 'L1T/'

# Path for filtered/processed L1T images
pathToL1TimagesFiltered = basePath + 'L1T/filteredImages/'

# Output folder
pathOut = basePath + 'OUT/'


# ============================================================
# GOOGLE EARTH ENGINE CONFIGURATION
# ============================================================

# Your GEE Project ID
# Get it from: https://code.earthengine.google.com/
GEE_PROJECT_ID = 'remote-sensing-478802'


# ============================================================
# SMARTS CONFIGURATION
# ============================================================

smartsPath = 'C:/Program Files/SMARTS_295_PC/'


# ============================================================
# IMAGE IDs FOR PROCESSING
# ============================================================

# Single image processing (process_hyperion.py)
# Change this to process different images one at a time
CURRENT_IMAGE = 'EO1H1990312002226110PZ'


# Multi-temporal inundation mapping (inundation_mapping.py)
# List all image pairs you want to analyze
# Format: (earlier_image_id, later_image_id)

INUNDATION_IMAGE_PAIRS = [
    # Example: Compare different sets of images
    ('EO1H1990312002306110PZ', 'EO1H1990312002226110PZ'),

    # Add more pairs as needed:
    # ('EO1H2020342014150110KF', 'EO1H2020342017150110KF'),
    # ('EO1H0370412009263110KF', 'EO1H0370412009266110PF'),
]

# For convenience: access the first pair directly
IMAGE_PAIR_1_EARLY = INUNDATION_IMAGE_PAIRS[0][0]
IMAGE_PAIR_1_LATE = INUNDATION_IMAGE_PAIRS[0][1]


# ============================================================
# PROCESSING OPTIONS
# ============================================================

# Destriping method: 'Pal' (recommended) or 'Datt'
destripingMethod = 'Pal'

# Local destriping refinement (slower but better results)
localDestriping = False

# Topographic correction
# Set to True to enable (requires DEM download from GEE)
use_topo = True

# DEM source from Google Earth Engine
# Options: 'NASA/NASADEM_HGT/001' (improved SRTM), 'USGS/SRTMGL1_003' (global), etc.
demID = 'NASA/NASADEM_HGT/001'
elevationName = 'elevation'

# Fallback options if DEM download fails
use_flat_dem = False  # Set to True to use flat terrain assumption (no DEM download)
dem_fallback_elevation_m = 0  # Default elevation in meters if using flat DEM

# SNAP wavelength compatibility
snap_wavelength_file = None  # Path to external wavelength file, or None
snap_keep_wavelength = True  # Keep wavelength field in HDR for SNAP visualization


# ============================================================
# POST-PROCESSING OPTIONS
# ============================================================

# Enable/disable post-processing (quicklooks, NDVI, NDWI, etc.)
run_postprocessing = True


# ============================================================
# INUNDATION MAPPING OPTIONS
# ============================================================

# NDWI increase threshold for inundation detection
# Higher values = more conservative (only large changes)
# Lower values = more sensitive (detects smaller changes)
# Recommended range: 0.05 - 0.20
inundation_threshold = 0.1

# Multiple thresholds for comparison (optional)
# The script will generate outputs for each threshold
inundation_thresholds = [0.05, 0.10, 0.15, 0.20]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_current_image_config():
    """Get configuration for single image processing."""
    return {
        'fname': CURRENT_IMAGE,
        'nameOut_radiance': CURRENT_IMAGE + '_preprocessed',
        'nameOut_reflectance': CURRENT_IMAGE + '_reflectance',
    }

def get_inundation_config(pair_index=0):
    """Get configuration for inundation mapping.

    Parameters:
    -----------
    pair_index : int
        Index of the image pair in INUNDATION_IMAGE_PAIRS (default: 0)

    Returns:
    --------
    dict with image1_id, image2_id, threshold
    """
    if pair_index >= len(INUNDATION_IMAGE_PAIRS):
        raise IndexError(f"Pair index {pair_index} out of range. Only {len(INUNDATION_IMAGE_PAIRS)} pairs configured.")

    image1, image2 = INUNDATION_IMAGE_PAIRS[pair_index]
    return {
        'image1_id': image1,
        'image2_id': image2,
        'threshold': inundation_threshold,
        'base_path': basePath,
        'output_path': pathOut,
    }


# ============================================================
# BATCH PROCESSING CONFIGURATION
# ============================================================

# List of all images to process (for batch processing)
ALL_IMAGES = [
    'EO1H2020342013284110KF',
    'EO1H2020342016359110KF',
    # Add more image IDs here as needed
]


# ============================================================
# DEBUGGING AND LOGGING
# ============================================================

# Enable verbose output
verbose = True

# Save intermediate results
save_intermediate = True
