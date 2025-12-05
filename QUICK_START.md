# DEM Fallback System - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Activate Environment
```bash
conda activate hyperion_roger
```

### Step 2: Configure (Optional)
Open `process_hyperion.py` and edit lines **1694-1716**:

```python
# Most common configurations:

# 🌟 RECOMMENDED (Default): Maximum robustness
local_dem_backup = None
use_flat_terrain_fallback = True

# 🗂️ WITH LOCAL BACKUP: For repeated processing of same region
local_dem_backup = basePath + 'DEM/my_dem.tif'

# 🔬 RESEARCH MODE: Require real DEM data
use_flat_terrain_fallback = False
```

### Step 3: Run Processing
```bash
cd C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/SUREHYP/
python process_hyperion.py
```

---

## 🔍 What to Expect

### ✅ Success (Primary DEM works):
```
[2/12] Download DEM images from GEE (with fallback support)
    [1/4] Primary DEM: USGS/SRTMGL1_003 (primary)
    ✓ SUCCESS: Obtained DEM from USGS/SRTMGL1_003
[3/12] Reproject DEM images...
```

### 🔄 Fallback (Primary fails, alternatives work):
```
[2/12] Download DEM images from GEE (with fallback support)
    [1/4] Primary DEM: USGS/SRTMGL1_003 (primary)
    ✗ FAILED: No data available
    [2/4] Fallback DEM: NASADEM
    ✓ SUCCESS: Obtained DEM from NASADEM
[3/12] Reproject DEM images...
```

### 🏔️ Flat Terrain (All sources failed):
```
    Applying flat terrain assumption as final fallback...

    Creating synthetic flat DEM:
      - Elevation: 0 m (sea level)
      - Slope: 0° (flat terrain)
```

---

## 📊 Quick Test

Validate the system is working:

```bash
python test_dem_fallback.py
```

Expected output: All checks marked with ✓

---

## ⚡ Common Scenarios

### Scenario 1: Ocean/Coastal Area
**Problem:** Primary DEM has no data over water
**Solution:** ✅ Automatic fallback to flat terrain (sea level)

### Scenario 2: GEE Server Issues
**Problem:** Network timeout or GEE maintenance
**Solution:** ✅ Tries alternative DEM sources automatically

### Scenario 3: Repeated Processing
**Problem:** Downloading same DEM repeatedly is slow
**Solution:** Save local DEM, set `local_dem_backup` path

---

## 🛠️ Configuration Presets

Copy these into `process_hyperion.py` (lines 1701, 1716):

### Preset 1: Maximum Reliability (Default)
```python
local_dem_backup = None
use_flat_terrain_fallback = True
```
**Use when:** General processing, diverse datasets

### Preset 2: Local DEM Priority
```python
local_dem_backup = basePath + 'DEM/my_dem.tif'
use_flat_terrain_fallback = True
```
**Use when:** Processing same region repeatedly

### Preset 3: Strict Mode
```python
local_dem_backup = None
use_flat_terrain_fallback = False
```
**Use when:** Research requiring real topography only

---

## 📖 Full Documentation

- **Comprehensive Guide:** DEM_FALLBACK_README.md
- **Implementation Details:** IMPLEMENTATION_SUMMARY.md
- **Test Suite:** test_dem_fallback.py

---

## ❓ Troubleshooting One-Liners

| Issue | Solution |
|-------|----------|
| "Module dem_fallback not found" | Check file is in same directory as process_hyperion.py |
| "GEE initialization failed" | Run: earthengine authenticate |
| "All DEM sources failed" | Enable flat terrain: use_flat_terrain_fallback = True |
| Processing very slow | Use local DEM for your region |
| Need real topography | Disable flat terrain: use_flat_terrain_fallback = False |

---

## ✨ Key Benefits

- ✅ No processing failures from missing DEM data
- ✅ Automatic fallback to 4 alternative sources
- ✅ Works with ocean/water areas
- ✅ Detailed logging shows what's happening
- ✅ Zero configuration needed (smart defaults)

---

**That's it!** The system works automatically. Just run your processing as usual. 🎉

For detailed documentation, see DEM_FALLBACK_README.md
