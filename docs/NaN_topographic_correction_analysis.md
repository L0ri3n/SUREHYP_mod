# NaN Values in Topographic Correction - Analysis Report

## Initial Hypothesis

> The error in topographic correction comes from NaN values in albedo generation because the image was taken under completely clear sky conditions (cloud cover = 0 in metadata).

**Verdict: Not supported.** The cloud cover metadata value is never read or used anywhere in the processing pipeline. The bypass exists for a different reason entirely, and the NaN values have other upstream causes.

---

## 1. Why Is Cloud Detection Bypassed?

The cloud detection step is skipped at `process_hyperion.py:892-895`:

```python
print('\n[6/12] Cloud/shadow detection (skipped - using all pixels as clearview)')
clearview = np.ones(L.shape[:2], dtype=np.uint8)
```

### Root Cause: Outdated Installed Package

The git history contains the original comment explaining the bypass:

> `NOTE: cloudAndShadowsDetection not available in installed surehyp version`

This was confirmed by runtime inspection:

| Location | Version | Has `cloudAndShadowsDetection`? |
|----------|---------|-------------------------------|
| `site-packages/surehyp/` (installed via pip) | 1.0.1.1 | **No** |
| `src/surehyp/` (local source) | 1.0.1.2 | **Yes** (`atmoCorrection.py:933`) |

The installed pip package is outdated. The local source code includes the function, but Python imports from `site-packages`, not from the local `src/` directory.

### Original Usage

In `example.py:116-127`, cloud detection was part of the standard workflow:

```python
clearview, clouds, shadows = surehyp.atmoCorrection.cloudAndShadowsDetection(
    bands, L, latit, doy, satelliteZenith, zenith, azimuth, slope, wazim)
L[clearview == 0] = 0  # Mask non-clearview pixels
```

### How to Re-enable

1. Reinstall the local package in development mode:
   ```bash
   conda activate Hyperion_roger
   pip install -e .
   ```
2. Replace the bypass block in `process_hyperion.py:892-895` with the original function call, adapting variable names to the current script.

**Caveat**: The `cloudAndShadowsDetection` function internally calls `MM_topo_correction` on band B4 for shadow detection (`atmoCorrection.py:988`), which itself has NaN-related issues (see Section 3). For a clear-sky image this is less critical since there are no clouds to generate shadows.

---

## 2. Actual NaN Sources in the Pipeline

### 2.1 Diagnostic Results

A diagnostic was run on the existing reflectance output (`EO1H2020342013284110KF_reflectance.img`):

| Metric | Value |
|--------|-------|
| Image shape | 3281 x 881 x 195 bands |
| NaN in saved file | 0 (stored as uint16, cannot hold NaN) |
| Zero pixels in band 40 | **1,856,108** / 2,890,561 (64.2%) |
| Pixels with band 40 > 0 | 1,034,453 (35.8%) |
| NDVI NaN from 0/0 | **1,856,108 pixels** |
| All-zero bands | Band 94 |

The large number of zero pixels are background/border pixels from the georeferencing step (the Hyperion swath does not fill the rectangular output grid).

### 2.2 Three NaN Injection Points

During the processing pipeline (before the reflectance image is saved to disk), NaN values are explicitly introduced at three locations:

#### Source 1: Bad Band Masking (`process_hyperion.py:980-981`)

```python
R[:, :, bad_bands_idx] = np.nan
```

Bands with statistically extreme values (> 10 MADs above median) are set to NaN. This is intentional but these NaN values are not cleaned before reaching `writeAlbedoFile`.

#### Source 2: Water Vapor Band Masking (`process_hyperion.py:557-589`)

```python
# mask_water_vapor_bands() sets:
R[:, :, bad_bands_idx] = np.nan  # 1350-1450 nm and 1800-1950 nm ranges
```

Atmospheric water vapor absorption bands are masked with NaN. Again intentional, but not cleaned before albedo generation.

#### Source 3: Division by Zero in `computeLtoRfactor` (`atmoCorrection.py:812`)

```python
factor = np.pi / T_gs / Dgt
```

If `Dgt` (global tilted irradiance from SMARTS) is zero for any wavelength, this produces `Inf`, which then multiplies `L` to produce `Inf` values in the reflectance array.

### 2.3 NaN Propagation Chain

```
R (reflectance) contains NaN from Sources 1-3
        |
        v
writeAlbedoFile(R, bands)              [atmoCorrection.py:1609]
  - Filters by R[:,:,40] > 0           [line 1622]
  - np.nanmedian() handles NaN...       [line 1623]
  - BUT if entire bands are NaN,
    median is still NaN for those bands
        |
        v
Albedo.txt written with NaN entries
        |
        v
rho_background = interp1d(w, r)        [process_hyperion.py:1088]
  - NaN in albedo -> NaN in rho_background
        |
        v
getDemReflectance()                     [atmoCorrection.py:1421]
  - Denominator uses rho_background:
    T_sg*E*cos(betai) + Dft*(1+cos(tilt))/2
    + (T_sg*E+Dft)*rho_background*(1-cos(tilt))/2
  - NaN in rho_background -> NaN denominator -> NaN LUT
        |
        v
MM_topo_correction()                    [atmoCorrection.py:1597]
  - NDVI = (NIR - red) / (NIR + red)   [various.py:198]
  - 1.8M pixels where NIR=red=0 -> 0/0 = NaN
  - G = cos(betai) / cos(betaT)        [line 1600]
  - G[NDVI > 0.2] = power(G, b_veg)    [line 1604]
  - NaN NDVI -> NaN G -> NaN in final output
```

---

## 3. Summary of Findings

| Question | Answer |
|----------|--------|
| Is cloud metadata (cloud cover = 0) causing the NaN? | **No.** Cloud metadata is never used. |
| Why is cloud detection skipped? | Installed `surehyp` package (v1.0.1.1) lacks the function. |
| Can cloud detection be re-enabled? | Yes, by running `pip install -e .` to use local source (v1.0.1.2). |
| What causes NaN in albedo? | Bad band masking and water vapor masking explicitly set bands to NaN in R before `writeAlbedoFile` is called. |
| What causes NaN in topographic correction? | (1) NaN-contaminated albedo propagates through `getDemReflectance` LUT. (2) 64% of pixels are zero-valued background, causing 0/0 = NaN in NDVI during `MM_topo_correction`. |

## 4. Recommended Fixes

1. **Reinstall local package**: `pip install -e .` in the `Hyperion_roger` environment to make `cloudAndShadowsDetection` available.

2. **Clean NaN before `writeAlbedoFile`**: Replace NaN with 0 (or interpolate) in R before computing the albedo spectrum, so that the albedo file does not contain NaN entries.

3. **Handle zero-background pixels in `MM_topo_correction`**: Mask background pixels (where band 40 = 0) before NDVI computation, or set NDVI to 0 where the denominator is zero, and set the G correction factor to 1 for those pixels.

4. **Guard division in `getDemReflectance`**: Add a check that `rho_background` does not contain NaN before building the LUT, or replace NaN with a sensible default (e.g., 0.2).
