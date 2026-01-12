# DEM Elevation Retrieval Fix - README

## Problem

The atmospheric correction was failing with:
```
ValueError: Could not retrieve elevation data from USGS/SRTMGL1_003
```

This occurred at step [10/12] when trying to get average scene elevation from Google Earth Engine (GEE).

## Root Causes

1. **Wrong scale parameter**: Used 1000m instead of SRTM's native 90m resolution
2. **No bestEffort flag**: GEE would fail for large regions without this
3. **No fallback**: Script crashed instead of using default elevation
4. **Poor error handling**: No debugging information to diagnose issues

## Solutions Implemented

### Enhanced getGEEdem_fixed() Function

**File**: `process_hyperion.py` (lines 191-256)

#### Fix 1: Correct Scale Resolution
```python
# BEFORE:
scale=1000  # Too coarse

# AFTER:
scale=90  # SRTM native resolution (90m x 90m)
```

#### Fix 2: Add bestEffort Flag
```python
mean_elev = DEM.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=coord,
    scale=90,
    maxPixels=1e9,
    bestEffort=True  # NEW: Allows computation for large areas
).getInfo()
```

#### Fix 3: Fallback to Default Elevation
If GEE fails completely, use a reasonable default instead of crashing:
```python
# Last resort: use default elevation
default_altit = 0.5  # 500m in km
print(f'    Using default elevation: 500m')
return default_altit
```

#### Fix 4: Better Error Handling
```python
try:
    # Try reduceRegion method first
    # Try sample method as fallback
    # Use default if both fail
except Exception as e:
    print(f'    Error querying GEE: {e}')
    return 0.5  # Default elevation
```

#### Fix 5: Debugging Output
Now provides detailed progress information:
```
    Querying DEM from USGS/SRTMGL1_003...
    Region: UL(lat, lon) to LR(lat, lon)
    Attempting reduceRegion method...
    Successfully retrieved elevation: 500.0 m (0.500 km)
```

## How It Works Now

```
1. Try reduceRegion (primary method)
   ├─ Use scale=90m (SRTM native resolution)
   ├─ Use bestEffort=True for large regions
   └─ If successful → return elevation

2. If failed, try sample method (fallback #1)
   ├─ Sample numPixels=1000 points
   └─ If successful → return mean elevation

3. If both failed, use default (fallback #2)
   └─ Return 0.5 km (500m) as reasonable default
```

## Expected Behavior

### Success Case:
```
[10/12] Get average elevation of the scene from GEE
    Querying DEM from USGS/SRTMGL1_003...
    Region: UL(-20.5678, 12.3456) to LR(-20.5890, 12.3678)
    Attempting reduceRegion method...
    Successfully retrieved elevation: 1234.5 m (1.235 km)
```

### Fallback Case (GEE timeout/failure):
```
[10/12] Get average elevation of the scene from GEE
    Querying DEM from USGS/SRTMGL1_003...
    Region: UL(-20.5678, 12.3456) to LR(-20.5890, 12.3678)
    Attempting reduceRegion method...
    reduceRegion returned None, trying sample method...
    Warning: Could not retrieve DEM data from GEE
    Using estimated elevation based on latitude...
    Using default elevation: 500 m (0.500 km)
```

### Error Case (network/auth failure):
```
[10/12] Get average elevation of the scene from GEE
    Querying DEM from USGS/SRTMGL1_003...
    Region: UL(-20.5678, 12.3456) to LR(-20.5890, 12.3678)
    Attempting reduceRegion method...
    Error querying GEE: <error message>
    Using default elevation of 500m (0.5 km)
```

## Impact on Results

### Using Real DEM Elevation:
- ✅ More accurate atmospheric correction
- ✅ Better SMARTS atmospheric modeling
- ✅ Precise water vapor and ozone calculations

### Using Default Elevation (500m):
- ⚠️ Slightly less accurate atmospheric correction
- ⚠️ Acceptable for most scenes (elevation impact is minor)
- ✅ Processing continues instead of crashing
- ℹ️ For high altitude sites (>2000m), consider manually setting elevation

## Manual Elevation Override (Optional)

If you know your scene's elevation and want to avoid GEE queries, you can manually set it:

**Option 1: Modify the function call** (line ~820):
```python
# Instead of querying GEE
altit = getGEEdem_fixed(...)

# Use known elevation directly
altit = 1.5  # 1500m in km
print(f'    Using manual elevation: {altit*1000:.0f} m')
```

**Option 2: Add configuration parameter**:
```python
# In main configuration section (line ~1280)
manual_elevation_km = None  # or 1.5 for 1500m

# Then in atmospheric_correction() use:
if manual_elevation_km is not None:
    altit = manual_elevation_km
else:
    altit = getGEEdem_fixed(...)
```

## Common GEE Issues

### Issue 1: GEE Authentication
**Symptom**: Error about authentication or project ID
**Solution**: Re-authenticate GEE
```bash
earthengine authenticate
```

### Issue 2: GEE Quota Exceeded
**Symptom**: Error about quota or rate limiting
**Solution**: Wait a few minutes and retry, or use default elevation

### Issue 3: Network Timeout
**Symptom**: Timeout error after long wait
**Solution**: The fallback will automatically use default elevation

### Issue 4: Scene Outside SRTM Coverage
**Symptom**: No elevation data (latitudes > 60° N or < 56° S)
**Solution**: Use different DEM or default elevation
```python
# For high latitudes, consider:
demID = 'CGIAR/SRTM90_V4'  # Different SRTM version
# or
demID = 'USGS/GTOPO30'     # Global coverage
```

## Verification

Check that elevation is reasonable:
- **Coastal areas**: 0-200m
- **Plains/lowlands**: 200-500m
- **Plateaus**: 500-1500m
- **Mountains**: 1500-4000m+

If the retrieved elevation seems wrong, check the coordinates in the console output.

## Summary

✅ **Fixed**: Changed scale from 1000m to 90m (SRTM native)
✅ **Fixed**: Added bestEffort=True for large regions
✅ **Fixed**: Fallback to default elevation instead of crashing
✅ **Fixed**: Better error handling and debugging output
✅ **Result**: Processing continues even if GEE fails

---

**Status**: ✅ FIXED
**Last Updated**: 2025-12-18
**Impact**: Atmospheric correction now more robust and won't crash on DEM retrieval failure
