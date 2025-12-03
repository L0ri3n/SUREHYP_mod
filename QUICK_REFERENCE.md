# Quick Reference: Integrated Metadata Processing

## To Process a New Dataset

### Step 1: Place XML File
Download from USGS EarthExplorer and save to:
```
C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/METADATA/
```

File will be named automatically as:
- `eo1_hyp_pub_{ImageID}_SGS_01.xml` OR
- `eo1_hyp_pub_{ImageID}_SG1_01.xml`

### Step 2: Update ONE Variable
In `process_hyperion.py` line 1051:
```python
fname = 'EO1H0370412009263110KF'  # ← Change this!
```

### Step 3: Run Script
```bash
conda activate hyperion_roger
python process_hyperion.py
```

That's it! ✓

## What Happens Automatically

1. ✓ Script finds XML file matching `eo1_hyp_pub_{fname}_*.xml` (any USGS variation)
2. ✓ Converts XML to CSV format
3. ✓ Appends to `METADATA/METADATA.csv` (no duplicates)
4. ✓ Processes image (preprocessing, atmospheric correction, outputs)

## Toggle Metadata Processing

Enable (default):
```python
process_xml_metadata = True
```

Disable:
```python
process_xml_metadata = False
```

## File Structure Required

```
Remote_Sensing/
├── METADATA/
│   ├── eo1_hyp_pub_EO1H0370412009263110KF_SGS_01.xml  (or any suffix)
│   └── METADATA.csv
├── L1R/
│   └── EO1H0370412009263110KF/
├── L1T/
│   └── EO1H0370412009263110KF_1T.ZIP
└── OUT/
    └── (outputs generated here)
```

## Supported Patterns

**ALL** USGS naming patterns work automatically:
- ✓ `eo1_hyp_pub_*_SGS_01.xml`
- ✓ `eo1_hyp_pub_*_SG1_01.xml`
- ✓ `eo1_hyp_pub_*_<any_suffix>.xml`

Script uses glob pattern to find **any** file matching `eo1_hyp_pub_{fname}_*.xml`

## Quick Troubleshooting

**XML file not found?**
- Check file is in `METADATA/` folder
- Verify filename starts with: `eo1_hyp_pub_{fname}_` and ends with `.xml`
- Ensure `fname` variable matches your image ID exactly (e.g., 'EO1H0370412009263110KF')

**Need to skip metadata conversion?**
- Set `process_xml_metadata = False` (line 1248)

## Example: Switching Datasets

From `EO1H0370412009263110KF` to `EO1H2020342013284110KF`:

```python
# OLD:
fname = 'EO1H0370412009263110KF'

# NEW:
fname = 'EO1H2020342013284110KF'
```

Done! Script automatically finds the correct XML file.
