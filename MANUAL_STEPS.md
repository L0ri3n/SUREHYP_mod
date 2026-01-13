# Manual Steps to Reprocess (CRITICAL!)

## ⚠️ **THE PROBLEM**

Your reflectance file already exists from **BEFORE** the fix was applied.

**The script skips atmospheric correction when the reflectance file exists!**

This is why you're seeing the same results - it's **loading the old broken file** with millions in the peaks.

## ✅ **THE SOLUTION**

You **MUST** delete the old reflectance files to force reprocessing with the new fixes.

---

## Option 1: Automatic (RECOMMENDED)

### Using Windows Command Prompt:

```cmd
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
delete_and_reprocess.bat
```

This will:
1. Delete all old reflectance files
2. Delete derived products (NDVI, statistics, etc.)
3. Run processing with the new fixes
4. Show you the results

---

## Option 2: Manual Steps

### Step 1: Open File Explorer

Navigate to:
```
C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT
```

### Step 2: Delete These Files

**Find and DELETE all files starting with `EO1H2020342013284110KF_reflectance`:**
- `EO1H2020342013284110KF_reflectance.hdr`
- `EO1H2020342013284110KF_reflectance.img`
- `EO1H2020342013284110KF_reflectance.hdr.backup`
- `EO1H2020342013284110KF_reflectance_spectral_info.txt`

**Also DELETE these:**
- `EO1H2020342013284110KF_NDVI.npy`
- `EO1H2020342013284110KF_statistics.txt`
- `EO1H2020342013284110KF_valid_pixels_mask.npy`
- `EO1H2020342013284110KF_reflectance_good_bands_mask.npy`

**In the `quicklooks` subfolder, DELETE:**
- `EO1H2020342013284110KF_spectra.png`
- `EO1H2020342013284110KF_RGB.png`
- `EO1H2020342013284110KF_FalseColor.png`
- `EO1H2020342013284110KF_NDVI.png`
- Any other files starting with `EO1H2020342013284110KF_`

### Step 3: Open Anaconda Prompt

Click Start → Search for "Anaconda Prompt" → Open it

### Step 4: Activate Environment

```bash
conda activate hyperion_roger
```

### Step 5: Navigate to SUREHYP

```bash
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
```

### Step 6: Run Processing

```bash
python process_hyperion.py
```

### Step 7: Watch the Output

**Look for these critical messages during processing:**

```
STEP 2: ATMOSPHERIC CORRECTION
============================================================

[Fix 1/3] Clipping reflectance outliers...
    Detected unscaled reflectance (median max: 0.XXXX)
    ⚠️  WARNING: Found X bands with extreme max values (bad calibration)
    These bands will be masked (set to NaN):
      Band 33: max = XXXXXXX.XX  ← Should see the 750nm spike here!
      Band 195: max = XXXXXX.XX  ← Should see the 2000nm spike here!
    ✅ After clipping: max = 1.0000, mean = 0.XXXX
```

**If you DON'T see these messages**, the atmospheric correction was skipped and you're still using old files!

---

## Option 3: Using PowerShell

1. Open PowerShell as Administrator
2. Navigate to SUREHYP directory:
   ```powershell
   cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
   ```
3. Run:
   ```powershell
   .\delete_and_reprocess.ps1
   ```

---

## How to Verify It Worked

### Check 1: Look at Console Output

During processing, you should see:
```
⚠️  WARNING: Found 2 bands with extreme max values (bad calibration)
```

If you see this, **IT'S WORKING!**

### Check 2: Check File Timestamps

After processing, check when the files were created:

```bash
ls -lh "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT\EO1H2020342013284110KF_reflectance.*"
```

The files should have **TODAY'S date and time** (after you ran the script).

### Check 3: Run Validation

```bash
cd "C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\SUREHYP"
python check_reflectance_values.py
```

**Expected output:**
```
--- TOP 5 BANDS WITH HIGHEST RAW VALUES ---
Band  45 ( 874.53 nm): raw=    100, scaled=1.0000
Band  44 ( 864.35 nm): raw=     95, scaled=0.9500
...

--- CHECKING SPECIFIC WAVELENGTHS ---
~750nm (band 33, 752.43nm): raw=0, scaled=nan  ← MASKED!
~2000nm (band 195, 2385.40nm): raw=0, scaled=nan  ← MASKED!

✅ SUCCESS: Max scaled value 1.0000 in reasonable range!
```

### Check 4: Look at New Spectra Plot

Open:
```
C:\Lorien\Archivos\TUBAF\1st_Semester\Remote_Sensing\OUT\quicklooks\EO1H2020342013284110KF_spectra.png
```

**You should see:**
- ✅ NO spike at 750nm (will show as gap or flat)
- ✅ Gap at 1400nm (water vapor)
- ✅ Gap at 1800-1950nm (water vapor)
- ✅ NO spike at 2000nm (will show as gap or flat)
- ✅ Max values around 0.8-1.0 (not millions!)

---

## Troubleshooting

### "Same results as before"

**Cause:** Reflectance file still exists, atmospheric correction was skipped

**Solution:**
1. Make 100% sure you deleted the reflectance files
2. Check file timestamps to confirm new files were created
3. Look for the "WARNING: Found X bands with extreme max values" message

### "No such file or directory"

**Cause:** Wrong directory or file already deleted

**Solution:** That's OK! It means the files don't exist, which is what we want.

### "Script says 'Reflectance file already exists, skipping Step 2'"

**Cause:** YOU DIDN'T DELETE THE FILES!

**Solution:** Go back and delete the reflectance files BEFORE running the script.

---

## Why This Is Necessary

The script has this logic:

```python
if os.path.exists(pathToReflectanceImage + '.img'):
    print('Reflectance file already exists, skipping Step 2...')
    # Loads old file (with bugs!)
else:
    # Runs atmospheric correction with fixes
```

**To use the new fixes, you MUST delete the old files first!**

---

## After Successful Reprocessing

Once you see proper results (max ~1.0, bad bands masked), you can proceed with SAM classification.

Your data will be in the correct 0-1 range and comparable to your Jarosite reference spectrum!

---

**Last Updated:** 2026-01-13
**Status:** ⚠️ **ACTION REQUIRED - DELETE OLD FILES FIRST!**
