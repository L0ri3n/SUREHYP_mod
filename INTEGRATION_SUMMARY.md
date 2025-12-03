# Metadata Integration Summary

## What Was Done

Successfully integrated the XML metadata conversion functions from `create_metadata_EO1H2020342013284110KF.py` into `process_hyperion.py` with smart automatic path detection.

## Key Features

### 1. Automatic Path Detection
The script now **automatically detects** your XML metadata file using only the `fname` variable. It supports both naming patterns:
- `eo1_hyp_pub_{fname}_SGS_01.xml` (tries this first)
- `eo1_hyp_pub_{fname}_SG1_01.xml` (fallback)

### 2. Verified Functionality
Tested and confirmed working with:
- ✓ `EO1H0370412009263110KF` → Found `*_SGS_01.xml`
- ✓ `EO1H2020342013284110KF` → Found `*_SG1_01.xml`

### 3. Simple Configuration
Only **ONE** variable to change:
```python
fname = 'EO1H0370412009263110KF'  # Change this to process different datasets
```

The script automatically:
- Constructs the XML path from `fname`
- Tries both SGS_01 and SG1_01 patterns
- Finds the correct file
- Converts to CSV format
- Appends to `METADATA/METADATA.csv`

## Implementation Details

### Code Location in process_hyperion.py:
- **Lines 191-327**: Three new functions added:
  - `parse_hyperion_xml()` - XML parsing
  - `convert_time_format()` - Time format conversion
  - `create_metadata_csv()` - CSV creation/appending

- **Lines 1234-1261**: Smart path detection logic:
  ```python
  if process_xml_metadata:
      # Try SGS_01 pattern first
      xml_metadata_path = basePath + 'METADATA/eo1_hyp_pub_' + fname + '_SGS_01.xml'
      if not os.path.exists(xml_metadata_path):
          # Try SG1_01 pattern as fallback
          xml_metadata_path = basePath + 'METADATA/eo1_hyp_pub_' + fname + '_SG1_01.xml'
  ```

- **Lines 1267-1292**: Metadata conversion step (Step 0):
  - Runs before preprocessing
  - Graceful error handling
  - Helpful messages if file not found

## Usage

### To Process a New Dataset:

1. **Download XML from USGS EarthExplorer:**
   - Go to https://earthexplorer.usgs.gov/
   - Find your image
   - Click "Metadata" → "Export Metadata"
   - Save in the `METADATA/` folder

2. **Update fname in process_hyperion.py:**
   ```python
   fname = 'EO1H0370412009263110KF'  # Only change this line!
   ```

3. **Run the script:**
   ```bash
   conda activate hyperion_roger
   python process_hyperion.py
   ```

### To Disable Metadata Conversion:
```python
process_xml_metadata = False
```

## Benefits

✅ **Automatic Detection**: No need to specify exact XML filename
✅ **Dual Pattern Support**: Works with both SGS_01 and SG1_01
✅ **Single Configuration Point**: Only change `fname` variable
✅ **Integrated Workflow**: Metadata conversion + image processing in one script
✅ **Error Handling**: Clear messages if XML file is missing
✅ **No Duplicates**: Checks for existing entries before appending

## Files Modified

- `process_hyperion.py` - Main script with integrated functions
- `METADATA_INTEGRATION_README.md` - Detailed usage documentation

## Testing

Verified with conda environment `hyperion_roger`:
```bash
conda run -n hyperion_roger python verify_xml_detection.py
```

Results:
- ✓ EO1H0370412009263110KF → SGS_01 pattern found
- ✓ EO1H2020342013284110KF → SG1_01 pattern found

## Notes

- XML files can use either `_SGS_01` or `_SG1_01` suffix (both are USGS standards)
- The script prioritizes `_SGS_01` as it's more common
- Automatic fallback to `_SG1_01` ensures compatibility with all datasets
- Original standalone script (`create_metadata_EO1H2020342013284110KF.py`) preserved for reference
