# SUREHYP Documentation

This folder contains all documentation for modifications and fixes made to the SUREHYP (Surface Reflectance from Hyperion) package.

## 📁 Folder Structure

```
docs/
├── README.md (this file)
├── guides/           # Usage guides and diagnostic tools
└── changelogs/       # Change history and bug fix reports
```

---

## 📚 Guides

**Location:** [guides/](guides/)

These documents provide instructions, configuration options, and diagnostic tools for using the modified SUREHYP package.

### Quick Start
- **[QUICK_START_GUIDE.md](guides/QUICK_START_GUIDE.md)** - 5-minute testing guide for preprocessing fixes

### Configuration & Usage
- **[SNAP_WAVELENGTH_GUIDE.md](guides/SNAP_WAVELENGTH_GUIDE.md)** - Configure wavelength metadata for SNAP compatibility

### Troubleshooting
- **[Preprocessing_Diagnostic_Report.md](guides/Preprocessing_Diagnostic_Report.md)** - Comprehensive diagnostic checklist with validation scripts to identify preprocessing issues

---

## 📝 Changelogs

**Location:** [changelogs/](changelogs/)

These documents detail all modifications, bug fixes, and improvements made to the package.

### Master Changelog
- **[CHANGELOG_SUMMARY.md](../CHANGELOG_SUMMARY.md)** - **Comprehensive summary of ALL changes** (Nov 2025 - Jan 2026)
  - Complete version history with dates
  - All 15 issues fixed with detailed descriptions
  - Migration guide and configuration options
  - New functions and files modified
  - Testing and validation instructions

### Comprehensive Reports
- **[MODIFICATIONS_REPORT.md](changelogs/MODIFICATIONS_REPORT.md)** - Complete overview of all modifications (Nov 21-22, 2025)
  - SNAP compatibility fix
  - Google Earth Engine API updates
  - Rasterio tiling issues
  - Atmospheric parameter fallbacks
  - And more...

- **[CHANGELOG.md](changelogs/CHANGELOG.md)** - Latest bug fixes (Jan 13, 2026)
  - Fixed "zero valid pixels" error
  - NaN-aware valid pixel detection
  - Improved wavelength loading with fallbacks

### Specific Bug Fixes
- **[BUGFIX_SUMMARY.md](changelogs/BUGFIX_SUMMARY.md)** - Wavelength metadata parser bug fix
- **[PREPROCESSING_FIXES_SUMMARY.md](changelogs/PREPROCESSING_FIXES_SUMMARY.md)** - Critical preprocessing fixes (bad bands, water vapor masking, outlier clipping)
- **[WAVELENGTH_FIX_README.md](changelogs/WAVELENGTH_FIX_README.md)** - KeyError: 'wavelength' fix during atmospheric correction
- **[DEM_FIX_README.md](changelogs/DEM_FIX_README.md)** - DEM elevation retrieval failure fix

---

## 🗓️ Timeline of Changes

| Date | Document | Summary |
|------|----------|---------|
| **Nov 21-22, 2025** | [MODIFICATIONS_REPORT.md](changelogs/MODIFICATIONS_REPORT.md) | Major compatibility fixes for SNAP, GEE, and rasterio |
| **Dec 18, 2025** | [PREPROCESSING_FIXES_SUMMARY.md](changelogs/PREPROCESSING_FIXES_SUMMARY.md) | Extreme spike fixes, water vapor masking |
| **Dec 18, 2025** | [WAVELENGTH_FIX_README.md](changelogs/WAVELENGTH_FIX_README.md) | Wavelength KeyError fix |
| **Dec 18, 2025** | [DEM_FIX_README.md](changelogs/DEM_FIX_README.md) | DEM elevation retrieval improvements |
| **Jan 13, 2026** | [CHANGELOG.md](changelogs/CHANGELOG.md) | Valid pixels detection bug fix |

---

## 🔍 Finding What You Need

### I want to...

**Get started quickly**
→ [QUICK_START_GUIDE.md](guides/QUICK_START_GUIDE.md)

**See all changes at a glance**
→ [CHANGELOG_SUMMARY.md](../CHANGELOG_SUMMARY.md) ⭐ **Start here!**

**Understand all changes made**
→ [MODIFICATIONS_REPORT.md](changelogs/MODIFICATIONS_REPORT.md)

**Configure SNAP compatibility**
→ [SNAP_WAVELENGTH_GUIDE.md](guides/SNAP_WAVELENGTH_GUIDE.md)

**Diagnose preprocessing issues**
→ [Preprocessing_Diagnostic_Report.md](guides/Preprocessing_Diagnostic_Report.md)

**See recent bug fixes**
→ [CHANGELOG.md](changelogs/CHANGELOG.md)

**Troubleshoot specific errors**
- Wavelength errors → [WAVELENGTH_FIX_README.md](changelogs/WAVELENGTH_FIX_README.md)
- DEM errors → [DEM_FIX_README.md](changelogs/DEM_FIX_README.md)
- Extreme spikes → [PREPROCESSING_FIXES_SUMMARY.md](changelogs/PREPROCESSING_FIXES_SUMMARY.md)
- Valid pixels error → [CHANGELOG.md](changelogs/CHANGELOG.md)

---

## 🐛 All Issues Fixed

| Issue | Status | Document |
|-------|--------|----------|
| SNAP band name conflicts | ✅ Fixed | [MODIFICATIONS_REPORT](changelogs/MODIFICATIONS_REPORT.md) |
| GEE API changes (SRTM) | ✅ Fixed | [MODIFICATIONS_REPORT](changelogs/MODIFICATIONS_REPORT.md) |
| Rasterio tiling errors | ✅ Fixed | [MODIFICATIONS_REPORT](changelogs/MODIFICATIONS_REPORT.md) |
| Extreme reflectance spikes | ✅ Fixed | [PREPROCESSING_FIXES_SUMMARY](changelogs/PREPROCESSING_FIXES_SUMMARY.md) |
| Water vapor bands | ✅ Fixed | [PREPROCESSING_FIXES_SUMMARY](changelogs/PREPROCESSING_FIXES_SUMMARY.md) |
| VNIR-SWIR discontinuities | ✅ Fixed | [PREPROCESSING_FIXES_SUMMARY](changelogs/PREPROCESSING_FIXES_SUMMARY.md) |
| Wavelength KeyError | ✅ Fixed | [WAVELENGTH_FIX_README](changelogs/WAVELENGTH_FIX_README.md) |
| DEM elevation retrieval | ✅ Fixed | [DEM_FIX_README](changelogs/DEM_FIX_README.md) |
| Zero valid pixels error | ✅ Fixed | [CHANGELOG](changelogs/CHANGELOG.md) |
| Wavelength parser bug | ✅ Fixed | [BUGFIX_SUMMARY](changelogs/BUGFIX_SUMMARY.md) |

---

## 📞 Support

For issues or questions:
- Review the [Preprocessing Diagnostic Report](guides/Preprocessing_Diagnostic_Report.md) for troubleshooting steps
- Check the [QUICK_START_GUIDE](guides/QUICK_START_GUIDE.md) for common issues
- Refer to original SUREHYP documentation for general usage

---

**Last Updated:** 2026-01-13
