# SUREHYP - Project Overview

**SUREHYP** (Surface Reflectance from Hyperion) converts raw EO-1 Hyperion hyperspectral satellite imagery from radiance to surface reflectance, suitable for spectral analysis and material classification (e.g., SAM).

**Original author:** Thomas Miraglio (v1.0.1.2)
**Modified by:** Lorien Crespo (Nov 2025 - Jan 2026)

---

## Processing Pipeline

```
INPUT: L1R & L1T Hyperion data (from USGS Earth Explorer)
  |
  v
STEP 1 - PREPROCESSING (preprocess.py)
  - Separate VNIR (bands 8-56) and SWIR (bands 78-224)
  - Remove bad/uncalibrated bands (1-7, 57-77, 225-242)
  - Desmile using cross-track illumination correction
  - Destrip using local spatial statistics
  - Align VNIR and SWIR detectors
  - Georeference using L1T as reference (homography)
  OUTPUT: *_preprocessed.img/.hdr (georeferenced radiance, 196 bands)
  |
  v
STEP 2 - ATMOSPHERIC CORRECTION (atmoCorrection.py)
  - Convert radiance to reflectance using SMARTS radiative transfer model
  - Extract water vapor from image absorption bands
  - Query Google Earth Engine for elevation, ozone, and water vapor
  - Optionally apply topographic correction using DEM
  - Post-correction: mask bad bands, mask water vapor bands
    (1350-1450 nm, 1800-1950 nm), smooth VNIR-SWIR transition
  OUTPUT: *_reflectance.img/.hdr (surface reflectance, 0-1 range)
  |
  v
STEP 3 - POST-PROCESSING (process_hyperion.py)
  - Create quicklook images (RGB, false color)
  - Calculate NDVI
  - Generate sample spectra plots
  - Create valid pixel masks
  OUTPUT: quicklooks/, masks, statistics
  |
  v
READY FOR: SAM classification or other spectral analysis
```

---

## Key Files

| File | Description |
|------|-------------|
| `process_hyperion.py` | Main processing script - orchestrates the full pipeline |
| `src/surehyp/preprocess.py` | Preprocessing functions (desmile, destrip, georeference) |
| `src/surehyp/atmoCorrection.py` | Atmospheric correction functions (SMARTS, reflectance conversion) |
| `src/surehyp/various.py` | Utility functions used by other modules |
| `example.py` | Example script showing the original processing chain |

---

## Requirements

- **Python** 3.7.5 - 3.9.9
- **Key packages:** numpy, scipy, pandas, matplotlib, spectral, rasterio, gdal, richdem, earthengine-api
- **External software:**
  - [SMARTS v2.9.5+](https://www.nrel.gov/grid/solar-resource/smarts.html) - radiative transfer model
  - Google Earth Engine account with project ID

## Setup

1. Install dependencies:
   ```
   conda install pyhdf rasterio richdem gdal
   pip install surehyp
   ```

2. Authenticate GEE:
   ```
   earthengine authenticate
   ```

3. Edit `process_hyperion.py` and configure:
   - `GEE_PROJECT_ID` - your GEE project ID
   - `smartsPath` - path to your SMARTS installation
   - `basePath` - path to your input data
   - `pathOut` - output directory
   - `snap_keep_wavelength` - True/False for SNAP compatibility

4. Run:
   ```
   python process_hyperion.py
   ```

---

## Output Files

| File | Description |
|------|-------------|
| `*_preprocessed.img/.hdr` | Preprocessed radiance image |
| `*_reflectance.img/.hdr` | Surface reflectance image |
| `*_spectral_info.txt` | Wavelength and FWHM metadata |
| `*_NDVI.npy` | NDVI array |
| `*_statistics.txt` | Image statistics |
| `*_valid_pixels_mask.npy` | Valid pixel mask |
| `*_reflectance_good_bands_mask.npy` | Good bands mask (water vapor excluded) |
| `quicklooks/*_RGB.png` | True color composite |
| `quicklooks/*_FalseColor.png` | NIR-R-G false color |
| `quicklooks/*_NDVI.png` | NDVI visualization |
| `quicklooks/*_spectra.png` | Sample pixel spectra |

---

## Known Limitations

- Topographic correction is currently disabled due to NaN handling in albedo file generation; processing continues with flat terrain assumption
- SRTM coverage is limited to 60N - 56S latitude
- GEE queries may fail under quota limits (fallback defaults are used)

---

## Further Documentation

- [CHANGELOG_SUMMARY.md](../CHANGELOG_SUMMARY.md) - All modifications made (Nov 2025 - Jan 2026)
- [SNAP_WAVELENGTH_GUIDE.md](guides/SNAP_WAVELENGTH_GUIDE.md) - SNAP wavelength configuration
- [Preprocessing_Diagnostic_Report.md](guides/Preprocessing_Diagnostic_Report.md) - Troubleshooting checklist
