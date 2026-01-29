# SUREHYP Documentation

This folder contains all documentation for modifications and fixes made to the SUREHYP (Surface Reflectance from Hyperion) package.

## Folder Structure

```
docs/
├── README.md (this file)
├── PROJECT_OVERVIEW.md      # What the code does and how to run it
├── guides/                  # Usage guides and diagnostic tools
└── changelogs/              # Change history and bug fix reports
```

---

## Project Overview

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Brief summary of the processing pipeline, key files, and usage instructions

## Guides

**Location:** [guides/](guides/)

- **[SNAP_WAVELENGTH_GUIDE.md](guides/SNAP_WAVELENGTH_GUIDE.md)** - Configure wavelength metadata for SNAP compatibility
- **[Preprocessing_Diagnostic_Report.md](guides/Preprocessing_Diagnostic_Report.md)** - Comprehensive diagnostic checklist with validation scripts to identify preprocessing issues

---

## Changelogs

**Location:** [changelogs/](changelogs/)

### Master Changelog

- **[CHANGELOG_SUMMARY.md](../CHANGELOG_SUMMARY.md)** - **Comprehensive summary of ALL changes** (Nov 2025 - Jan 2026)

### Detailed Reports

- **[MODIFICATIONS_REPORT.md](changelogs/MODIFICATIONS_REPORT.md)** - Complete overview of compatibility fixes (Nov 21-22, 2025): SNAP, GEE API, rasterio, atmospheric parameters, topographic correction, SMARTS path
- **[PREPROCESSING_FIXES_SUMMARY.md](changelogs/PREPROCESSING_FIXES_SUMMARY.md)** - Preprocessing quality fixes (Dec 18, 2025): bad band removal, water vapor masking, outlier clipping, VNIR-SWIR smoothing
- **[CHANGELOG.md](changelogs/CHANGELOG.md)** - Latest bug fixes (Jan 13, 2026): zero valid pixels error, NaN-aware detection, wavelength loading fallbacks

---

## Finding What You Need

| I want to... | Go to |
|---|---|
| Understand what the code does | [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) |
| See all changes at a glance | [CHANGELOG_SUMMARY.md](../CHANGELOG_SUMMARY.md) |
| Configure SNAP compatibility | [SNAP_WAVELENGTH_GUIDE.md](guides/SNAP_WAVELENGTH_GUIDE.md) |
| Diagnose preprocessing issues | [Preprocessing_Diagnostic_Report.md](guides/Preprocessing_Diagnostic_Report.md) |

---

## All Issues Fixed

| Issue | Status | Document |
|-------|--------|----------|
| SNAP band name conflicts | Fixed | [MODIFICATIONS_REPORT](changelogs/MODIFICATIONS_REPORT.md) |
| GEE API changes (SRTM) | Fixed | [MODIFICATIONS_REPORT](changelogs/MODIFICATIONS_REPORT.md) |
| Rasterio tiling errors | Fixed | [MODIFICATIONS_REPORT](changelogs/MODIFICATIONS_REPORT.md) |
| Extreme reflectance spikes | Fixed | [PREPROCESSING_FIXES_SUMMARY](changelogs/PREPROCESSING_FIXES_SUMMARY.md) |
| Water vapor bands | Fixed | [PREPROCESSING_FIXES_SUMMARY](changelogs/PREPROCESSING_FIXES_SUMMARY.md) |
| VNIR-SWIR discontinuities | Fixed | [PREPROCESSING_FIXES_SUMMARY](changelogs/PREPROCESSING_FIXES_SUMMARY.md) |
| Wavelength KeyError | Fixed | [CHANGELOG_SUMMARY](../CHANGELOG_SUMMARY.md) |
| DEM elevation retrieval | Fixed | [CHANGELOG_SUMMARY](../CHANGELOG_SUMMARY.md) |
| Wavelength parser bug | Fixed | [CHANGELOG_SUMMARY](../CHANGELOG_SUMMARY.md) |
| Zero valid pixels error | Fixed | [CHANGELOG](changelogs/CHANGELOG.md) |

---

**Last Updated:** 2026-01-29
