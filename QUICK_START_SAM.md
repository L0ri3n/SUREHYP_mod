# Quick Start Guide for SAM Users
## How to Use Preprocessed Hyperion Data with Spectral Angle Mapper

---

## The Problem (in 30 seconds)

Your preprocessed Hyperion data is stored with a **scale factor of 100**:
- Stored range: **0-100** (as uint16 integers)
- SAM needs: **0-1** (as floats)
- **You must divide by 100 before using SAM**

If you don't divide by 100, SAM will fail with invalid angles!

---

## The Solution (3 options)

### Option 1: Use the Helper Module (EASIEST)

```python
from sam_data_helper import load_reflectance_for_sam

# This automatically applies the scale factor
cube, wavelengths, metadata = load_reflectance_for_sam('reflectance.hdr')

# If your endmembers are also from the same preprocessed data:
endmembers = endmembers / 100.0

# Done! Now use SAM
```

### Option 2: Manual Scaling (RECOMMENDED if not using helper)

```python
import spectral.io.envi as envi
import numpy as np

# Load image
img = envi.open('reflectance.hdr')
cube = img.load()

# Apply scale factor
scale = float(img.metadata['scale factor'][0])  # Gets 100
cube = cube.astype(np.float32) / scale

# Apply to endmembers too
endmembers = endmembers / scale

# Now use SAM
```

### Option 3: Check if Already Scaled

```python
# Quick check
max_val = np.max(cube[cube > 0])

if max_val > 2.0:
    print("ERROR: Data needs scaling! Divide by 100")
    cube = cube / 100.0
else:
    print("OK: Data already in 0-1 range")
```

---

## Quick Validation

Before running SAM, always check:

```python
# Should be ~0.02 to ~0.8
print("Cube range:", np.min(cube[cube>0]), "to", np.max(cube))

# Should be ≤ 1.0
test_pixel = cube[100, 100, :]
test_endmember = endmembers['Vegetation']
cos_angle = np.dot(test_pixel, test_endmember) / (np.linalg.norm(test_pixel) * np.linalg.norm(test_endmember))
print("Test cosine:", cos_angle)

if abs(cos_angle) > 1.0:
    print("ERROR: Scale mismatch! Check your scaling.")
```

---

## Complete Example SAM Script

```python
import numpy as np
from sam_data_helper import load_reflectance_for_sam, validate_reflectance_scale

# 1. Load reflectance (automatically scaled)
cube, wavelengths, metadata = load_reflectance_for_sam('reflectance.hdr')

# 2. Load endmembers (your code here)
endmembers = {}
endmembers['Vegetation'] = np.array([...])  # Your endmember data
endmembers['Soil'] = np.array([...])

# 3. Scale endmembers if needed
# If endmembers come from spectral library (already 0-1): DON'T scale
# If endmembers come from this preprocessed image: DO scale
if endmember_source == 'preprocessed':
    for key in endmembers:
        endmembers[key] = endmembers[key] / 100.0

# 4. Validate both
validate_reflectance_scale(cube, "Image cube")
for key in endmembers:
    validate_reflectance_scale(endmembers[key], f"Endmember {key}")

# 5. Run SAM
def spectral_angle_mapper(pixel, endmember):
    """Compute SAM angle in radians"""
    cos_angle = np.dot(pixel, endmember) / (np.linalg.norm(pixel) * np.linalg.norm(endmember))
    cos_angle = np.clip(cos_angle, -1, 1)  # Safety clip
    return np.arccos(cos_angle)

# Apply SAM to whole image
sam_results = {}
for em_name, em_spectrum in endmembers.items():
    sam_map = np.zeros((cube.shape[0], cube.shape[1]))

    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            pixel = cube[i, j, :]
            if np.any(pixel > 0):  # Only process valid pixels
                sam_map[i, j] = spectral_angle_mapper(pixel, em_spectrum)

    sam_results[em_name] = sam_map

# 6. Classification (assign each pixel to closest endmember)
classification = np.zeros((cube.shape[0], cube.shape[1]), dtype=np.uint8)
min_angles = np.full((cube.shape[0], cube.shape[1]), np.inf)

for idx, (em_name, sam_map) in enumerate(sam_results.items(), start=1):
    mask = sam_map < min_angles
    classification[mask] = idx
    min_angles = np.minimum(min_angles, sam_map)

print("SAM classification complete!")
```

---

## Troubleshooting

### Problem: SAM angles > 90 degrees for everything
**Cause**: Data not scaled
**Fix**: Divide by 100

### Problem: Invalid cosine (> 1.0)
**Cause**: Cube and endmembers have different scales
**Fix**: Ensure both are scaled the same way

### Problem: All angles are very small (< 0.01 radians)
**Cause**: Data scaled twice (divided by 10000 instead of 100)
**Fix**: Multiply by 100 or reload without extra scaling

### Problem: Results look wrong but angles are valid
**Cause**: Band mismatch between image and endmembers
**Fix**: Check wavelengths match

---

## Files You Need

1. **sam_data_helper.py** - Helper functions for loading data
2. **CHANGES_SUMMARY.md** - Full documentation of changes
3. **PREPROCESSING_ISSUES_FOUND.md** - Detailed issue analysis

---

## One-Minute Checklist

Before running SAM:

- [ ] Activated hyperion conda environment
- [ ] Loaded data with helper OR manually divided by scale factor
- [ ] Checked cube range is ~0.02-0.8 (not 2-80!)
- [ ] Scaled endmembers the same way as cube
- [ ] Test SAM on one pixel gives valid cosine (≤ 1.0)
- [ ] Checked band count matches between cube and endmembers

**If all checked: You're ready for SAM!**

---

## Need Help?

1. Run the helper script to validate your data:
   ```bash
   conda activate hyperion
   python sam_data_helper.py your_reflectance.hdr
   ```

2. Check the mean spectrum plot:
   ```
   OUT/quicklooks/<image_name>_mean_spectrum.png
   ```

3. Review the detailed documentation:
   - PREPROCESSING_ISSUES_FOUND.md
   - CHANGES_SUMMARY.md
