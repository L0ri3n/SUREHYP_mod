# Metadata Integration in process_hyperion.py

## Summary

The metadata creation functions from `create_metadata_EO1H2020342013284110KF.py` have been successfully integrated into `process_hyperion.py`. You can now generate the metadata CSV file directly from the main processing script.

## What Was Changed

### 1. Added Functions (Lines 192-328)
Three new functions were added to `process_hyperion.py`:

- **`parse_hyperion_xml(xml_path)`**: Parses USGS EarthExplorer XML metadata files
- **`convert_time_format(time_str)`**: Converts time format from HH:MM:SS.SSS to HH:MM:SS
- **`create_metadata_csv(xml_path, output_csv_path, append=True)`**: Creates or appends to METADATA.csv

### 2. Added Configuration Section (Lines 1234-1248)
New configuration block for optional XML metadata conversion:

```python
# ============================================================
# METADATA XML CONFIGURATION (OPTIONAL)
# ============================================================
xml_metadata_path = None
# Example: xml_metadata_path = basePath + 'Project/eo1_hyp_pub_EO1H0370412009263110KF_SG1_01.xml'
```

### 3. Integrated Metadata Conversion (Lines 1259-1284)
Added automatic metadata conversion step (Step 0) at the beginning of processing:
- Runs before preprocessing if `xml_metadata_path` is specified
- Converts XML to CSV and appends to `pathToL1Rmetadata` (METADATA.csv)
- Provides helpful error messages if XML file is not found

## How to Use

The XML path is **automatically constructed** from the `fname` variable, so you only need to change `fname` to process different datasets!

### Option 1: With Metadata Conversion (Default)
1. Download your XML metadata from USGS EarthExplorer:
   - Go to https://earthexplorer.usgs.gov/
   - Find your image (e.g., EO1H0370412009263110KF)
   - Click "Metadata" → "Export Metadata"
   - Save in the **METADATA folder** (naming is automatic from USGS)

2. The script automatically detects both naming patterns:
   ```
   eo1_hyp_pub_{fname}_SGS_01.xml  (most common)
   eo1_hyp_pub_{fname}_SG1_01.xml  (alternative)
   ```
   Examples:
   - `eo1_hyp_pub_EO1H0370412009263110KF_SGS_01.xml`
   - `eo1_hyp_pub_EO1H2020342013284110KF_SG1_01.xml`

3. Set the image ID in `process_hyperion.py`:
```python
fname = 'EO1H0370412009263110KF'  # Only change this!
process_xml_metadata = True  # Keep this True
```

4. Run the script:
```bash
python process_hyperion.py
```

The script will:
- Automatically find the XML file (tries both SGS_01 and SG1_01 patterns)
- Convert XML to CSV format (Step 0)
- Append to `METADATA/METADATA.csv`
- Continue with normal processing (Steps 1-3)

### Option 2: Without Metadata Conversion
Simply disable metadata processing:
```python
process_xml_metadata = False
```

The script will skip metadata conversion and run normal processing only.

## Example Output

When metadata conversion is enabled, you'll see:

```
============================================================
STEP 0: METADATA CONVERSION (XML TO CSV)
============================================================

Input XML: C:/Lorien/.../eo1_hyp_pub_EO1H0370412009263110KF_SG1_01.xml
Output CSV: C:/Lorien/.../METADATA/METADATA.csv

    Appended EO1H0370412009263110KF_SG1_01 to METADATA/METADATA.csv

    Metadata Summary for EO1H0370412009263110KF:
      Acquisition Date: 2009-09-20
      Sun Elevation: 45.5°
      Sun Azimuth: 145.2°
      Look Angle: 0.5°
      Center Coordinates: (34.5, -118.2)
      Cloud Cover: 10%

============================================================
METADATA CONVERSION COMPLETE!
============================================================
```

## Benefits

1. **Single Script**: No need to run separate metadata conversion scripts
2. **Automatic Path Construction**: XML path is built automatically from `fname`
3. **Smart Workflow**: Only change `fname` to switch between datasets
4. **Optional**: Can be enabled/disabled with `process_xml_metadata` flag
5. **Error Handling**: Provides clear messages if XML file is missing
6. **No Duplicates**: Checks for existing entries before appending

## Quick Start Guide

To process a new dataset:

1. **Place your files in the correct folders:**
   ```
   METADATA/
   ├── eo1_hyp_pub_EO1H0370412009263110KF_SGS_01.xml  ← Download from USGS
   └── METADATA.csv                                     ← Auto-generated

   L1R/
   └── EO1H0370412009263110KF/                         ← Unzipped L1R folder

   L1T/
   └── EO1H0370412009263110KF_1T.ZIP                   ← L1T ZIP file
   ```

   Note: XML files can be either `*_SGS_01.xml` or `*_SG1_01.xml` - both are supported!

2. **Update one line in process_hyperion.py:**
   ```python
   fname = 'EO1H0370412009263110KF'  # Change this to your image ID
   ```

3. **Run the script:**
   ```bash
   python process_hyperion.py
   ```

That's it! The script will automatically:
- Find the XML file (detects both `*_SGS_01.xml` and `*_SG1_01.xml` patterns)
- Convert it to CSV and append to `METADATA/METADATA.csv`
- Process the image through all steps

## Files Modified

- **process_hyperion.py**: Main processing script with integrated metadata functions

## Original Functionality Preserved

All original metadata creation functionality from `create_metadata_EO1H2020342013284110KF.py` is preserved:
- XML parsing with proper namespaces
- Time format conversion
- All SUREHYP-required CSV columns
- Duplicate checking
- Append mode support
- Metadata summary display
