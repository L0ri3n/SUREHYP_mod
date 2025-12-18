# Unified Configuration Guide

## 🎯 Quick Start: Single Configuration File

**All configuration is now in ONE file: `config.py`**

You no longer need to edit multiple scripts! Just edit `config.py` and run the scripts.

---

## 📝 How to Configure Everything

### **Edit `config.py` - That's It!**

Open `config.py` and find these sections:

### 1️⃣ **Processing Single Images**

```python
# ============================================================
# IMAGE IDs FOR PROCESSING
# ============================================================

# Single image processing (process_hyperion.py)
# Change this to process different images one at a time
CURRENT_IMAGE = 'EO1H2020342013284110KF'  # ← EDIT THIS LINE
```

**To process a different image:**
- Change `CURRENT_IMAGE` to your image ID
- Run: `python process_hyperion.py`

**Example:**
```python
# Process first image (2013)
CURRENT_IMAGE = 'EO1H2020342013284110KF'
# Save, then run: python process_hyperion.py

# Process second image (2016)
CURRENT_IMAGE = 'EO1H2020342016359110KF'
# Save, then run: python process_hyperion.py
```

---

### 2️⃣ **Inundation Mapping (Compare Two Images)**

```python
# Multi-temporal inundation mapping (inundation_mapping.py)
# List all image pairs you want to analyze
# Format: (earlier_image_id, later_image_id)

INUNDATION_IMAGE_PAIRS = [
    # Example: Compare 2013 and 2016 images
    ('EO1H2020342013284110KF', 'EO1H2020342016359110KF'),  # ← EDIT THIS

    # Add more pairs as needed:
    # ('EO1H2020342014150110KF', 'EO1H2020342017150110KF'),
    # ('EO1H0370412009263110KF', 'EO1H0370412009266110PF'),
]
```

**To compare different images:**
- Edit the tuple: `('earlier_image', 'later_image')`
- The script uses the first pair in the list
- Run: `python inundation_mapping.py`

---

### 3️⃣ **Inundation Detection Threshold**

```python
# ============================================================
# INUNDATION MAPPING OPTIONS
# ============================================================

# NDWI increase threshold for inundation detection
inundation_threshold = 0.1  # ← EDIT THIS
```

**Threshold values:**
- `0.05` = Very sensitive (detects small changes)
- `0.10` = Default (balanced)
- `0.15` = Conservative (only large changes)
- `0.20` = Very conservative (permanent water only)

---

### 4️⃣ **Other Common Settings**

```python
# Google Earth Engine Project ID
GEE_PROJECT_ID = 'remote-sensing-478802'  # Your GEE ID

# Base path (where all your data is)
basePath = 'C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/'

# Processing options
destripingMethod = 'Pal'      # 'Pal' or 'Datt'
localDestriping = False       # True = slower but better
use_topo = True               # Topographic correction
use_flat_dem = False          # True = skip DEM download
run_postprocessing = True     # Generate NDVI, NDWI, quicklooks
```

---

## 🚀 Workflow

### **Step 1: Process Images (One at a Time)**

```bash
# Edit config.py
CURRENT_IMAGE = 'EO1H2020342013284110KF'

# Run
conda activate Hyperion_roger
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
python process_hyperion.py
```

```bash
# Edit config.py again
CURRENT_IMAGE = 'EO1H2020342016359110KF'

# Run again
python process_hyperion.py
```

### **Step 2: Compare Images**

```bash
# Edit config.py
INUNDATION_IMAGE_PAIRS = [
    ('EO1H2020342013284110KF', 'EO1H2020342016359110KF'),
]

# Run
python inundation_mapping.py
```

Done! All outputs in `OUT/` and `OUT/inundation/`

---

## 📂 What Changed?

### **Before (Old Way):**
```
❌ Edit process_hyperion.py line 1810
❌ Edit inundation_mapping.py lines 658-659
❌ Keep track of multiple configuration locations
```

### **After (New Way):**
```
✅ Edit config.py CURRENT_IMAGE
✅ Edit config.py INUNDATION_IMAGE_PAIRS
✅ Everything in one place!
```

---

## 🎓 Examples

### **Example 1: Process 3 Images**

Edit `config.py`:
```python
CURRENT_IMAGE = 'EO1H2020342013284110KF'
```
Run: `python process_hyperion.py`

Edit `config.py`:
```python
CURRENT_IMAGE = 'EO1H2020342016359110KF'
```
Run: `python process_hyperion.py`

Edit `config.py`:
```python
CURRENT_IMAGE = 'EO1H0370412009263110KF'
```
Run: `python process_hyperion.py`

### **Example 2: Compare Multiple Pairs**

Edit `config.py`:
```python
INUNDATION_IMAGE_PAIRS = [
    # Pair 1: Will be processed by default
    ('EO1H2020342013284110KF', 'EO1H2020342016359110KF'),

    # Pair 2: Can be processed by changing pair_index
    ('EO1H0370412009263110KF', 'EO1H0370412009266110PF'),
]
```

Run for first pair:
```bash
python inundation_mapping.py  # Uses pair_index=0 by default
```

### **Example 3: Change Threshold**

Edit `config.py`:
```python
inundation_threshold = 0.15  # More conservative
```

Run:
```bash
python inundation_mapping.py
```

Check results, adjust if needed:
```python
inundation_threshold = 0.05  # More sensitive
```

Run again:
```bash
python inundation_mapping.py
```

---

## 🔍 Configuration Validation

The scripts will show you what configuration was loaded:

```
============================================================
CONFIGURATION LOADED FROM config.py
============================================================
Current image: EO1H2020342013284110KF
Base path: C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/
Output path: C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/OUT/
============================================================
```

Check this output to verify your settings are correct!

---

## 📋 Configuration Checklist

Before running, verify in `config.py`:

### For Single Image Processing:
- [ ] `CURRENT_IMAGE` is set to the image you want to process
- [ ] `GEE_PROJECT_ID` is your actual project ID
- [ ] `basePath` points to your data directory
- [ ] Raw L1R folder exists: `L1R/{CURRENT_IMAGE}/`

### For Inundation Mapping:
- [ ] Both images in `INUNDATION_IMAGE_PAIRS[0]` are already processed
- [ ] Both reflectance files exist in `OUT/` directory
- [ ] `inundation_threshold` is appropriate for your analysis
- [ ] Images are in chronological order (earlier, later)

---

## 💡 Pro Tips

### **Tip 1: Keep a Config Backup**
```bash
cp config.py config_backup.py
```

### **Tip 2: Multiple Configurations**
Create different configs for different projects:
```bash
cp config.py config_project1.py
cp config.py config_project2.py
```

Then import the one you need:
```python
# In process_hyperion.py or inundation_mapping.py
import config_project1 as config  # Change this line
```

### **Tip 3: Batch Processing**

Edit `config.py`:
```python
ALL_IMAGES = [
    'EO1H2020342013284110KF',
    'EO1H2020342016359110KF',
    'EO1H0370412009263110KF',
]
```

Create a simple batch script:
```python
import config

for image_id in config.ALL_IMAGES:
    config.CURRENT_IMAGE = image_id
    # Run processing...
```

### **Tip 4: Check Before Running**

Quick verification:
```bash
python -c "import config; print(f'Current: {config.CURRENT_IMAGE}'); print(f'Pair: {config.INUNDATION_IMAGE_PAIRS[0]}')"
```

Output:
```
Current: EO1H2020342013284110KF
Pair: ('EO1H2020342013284110KF', 'EO1H2020342016359110KF')
```

---

## 🔧 Troubleshooting

### **Issue: "No module named 'config'"**
**Solution:** Make sure you're in the correct directory:
```bash
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
```

### **Issue: Changes not taking effect**
**Solution:** Make sure you saved `config.py` after editing

### **Issue: Wrong image being processed**
**Solution:** Check `config.CURRENT_IMAGE` value:
```bash
python -c "import config; print(config.CURRENT_IMAGE)"
```

---

## 📚 Summary

| **What You Want to Do** | **Edit in config.py** | **Then Run** |
|--------------------------|----------------------|--------------|
| Process single image | `CURRENT_IMAGE` | `python process_hyperion.py` |
| Compare two images | `INUNDATION_IMAGE_PAIRS[0]` | `python inundation_mapping.py` |
| Change threshold | `inundation_threshold` | `python inundation_mapping.py` |
| Change paths | `basePath` | Either script |
| Change GEE ID | `GEE_PROJECT_ID` | `python process_hyperion.py` |

---

**Key Benefit:** 🎉 **Change once, use everywhere!**

No more hunting through multiple files to change settings. Everything is in `config.py`.

---

**Last Updated:** 2025-12-18
**Version:** 2.0 (Unified Configuration)
