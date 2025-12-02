"""
Metadata XML to CSV Converter for Hyperion Image Processing
Converts USGS EarthExplorer XML metadata to CSV format required by SUREHYP

Image: EO1H2020342013284110KF
"""

import xml.etree.ElementTree as ET
import pandas as pd
import os

def parse_hyperion_xml(xml_path):
    """
    Parse Hyperion XML metadata file from USGS EarthExplorer

    Parameters:
    -----------
    xml_path : str
        Path to the XML metadata file

    Returns:
    --------
    dict : Dictionary containing all metadata fields
    """
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Define namespace
    ns = {'eemetadata': 'http://earthexplorer.usgs.gov/eemetadata.xsd'}

    # Extract metadata fields
    metadata = {}
    for field in root.findall('.//eemetadata:metadataField', ns):
        field_name = field.get('name')
        field_value = field.find('eemetadata:metadataValue', ns).text
        metadata[field_name] = field_value

    return metadata

def convert_time_format(time_str):
    """
    Convert time from HH:MM:SS.SSS to HH:MM:SS format
    Example: 2013:284:10:09:05.020 → 2013:284:10:09:05
    """
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) >= 4:
            # Remove milliseconds from seconds
            seconds = parts[3].split('.')[0]
            return "{0}:{1}:{2}:{3}".format(parts[0], parts[1], parts[2], seconds)
    return time_str

def create_metadata_csv(xml_path, output_csv_path, append=True):
    """
    Create or append to metadata CSV file from XML

    Parameters:
    -----------
    xml_path : str
        Path to the XML metadata file
    output_csv_path : str
        Path for the output CSV file
    append : bool
        If True, append to existing CSV. If False, create new CSV.
    """
    # Parse XML
    metadata = parse_hyperion_xml(xml_path)

    # Extract Scene ID from Entity ID (remove _SG1_01 suffix)
    entity_id = metadata['Entity ID']
    scene_id = entity_id[:-7] if entity_id.endswith('_SG1_01') else entity_id

    # Create DataFrame with required columns (matching SUREHYP expectations)
    df_new = pd.DataFrame({
        'Entity ID': [metadata['Entity ID']],
        'Scene ID': [scene_id],
        'Acquisition Date': [metadata['Acquisition Date']],
        'Cloud Cover': [int(metadata['Cloud Cover'])],
        'Orbit Path': [int(metadata['Orbit Path'])],
        'Orbit Row': [int(metadata['Orbit Row'])],
        'Target Path': [int(metadata['Target Path'])],
        'Target Row': [int(metadata['Target Row'])],
        'Station': [metadata['Station']],
        'Processing Level': [metadata['Processing Level']],
        'Scene Start Time': [convert_time_format(metadata['Scene Start Time'])],
        'Start Time': [metadata['Scene Start Time'].split(':')[-1].split('.')[0] if ':' in metadata['Scene Start Time'] else 'N/A'],
        'Scene Stop Time': [convert_time_format(metadata['Scene Stop Time'])],
        'Stop Time': [metadata['Scene Stop Time'].split(':')[-1].split('.')[0] if ':' in metadata['Scene Stop Time'] else 'N/A'],
        'Sun Azimuth': [float(metadata['Sun Azimuth'])],
        'Sun Elevation': [float(metadata['Sun Elevation'])],
        'Satellite Inclination': [float(metadata['Satellite Inclination'])],
        'Look Angle': [float(metadata['Look Angle'])],
        'Date Entered': [metadata['Date Entered']],
        'Center Latitude': [metadata['Center Latitude']],
        'Center Longitude': [metadata['Center Longitude']],
        'Center Latitude dec': [float(metadata['Center Latitude dec'])],
        'Center Longtude dec': [float(metadata['Center Longtude dec'])],  # Note: typo 'Longtude' matches SUREHYP
        'NW Corner Lat': [metadata['NW Corner Lat']],
        'NW Corner Long': [metadata['NW Corner Long']],
        'NE Corner Lat': [metadata['NE Corner Lat']],
        'NE Corner Long': [metadata['NE Corner Long']],
        'SE Corner Lat': [metadata['SE Corner Lat']],
        'SE Corner Long': [metadata['SE Corner Long']],
        'SW Corner Lat': [metadata['SW Corner Lat']],
        'SW Corner Long': [metadata['SW Corner Long']],
        'NW Corner Lat dec': [float(metadata['NW Corner Lat dec'])],
        'NW Corner Long dec': [float(metadata['NW Corner Long dec'])],
        'NE Corner Lat dec': [float(metadata['NE Corner Lat dec'])],
        'NE Corner Long dec': [float(metadata['NE Corner Long dec'])],
        'SE Corner Lat dec': [float(metadata['SE Corner Lat dec'])],
        'SE Corner Long dec': [float(metadata['SE Corner Long dec'])],
        'SW Corner Lat dec': [float(metadata['SW Corner Lat dec'])],
        'SW Corner Long dec': [float(metadata['SW Corner Long dec'])],
        'Satellite Azimuth': [float(metadata.get('Satellite Azimuth', 0.0))]  # Not always in XML
    })

    # Append or create new CSV
    if append and os.path.exists(output_csv_path):
        # Read existing CSV
        df_existing = pd.read_csv(output_csv_path)

        # Check if this Entity ID already exists
        if entity_id in df_existing['Entity ID'].values:
            print("Warning: {} already exists in {}".format(entity_id, output_csv_path))
            print("Skipping to avoid duplicates.")
            return

        # Append new row
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_csv_path, index=False)
        print("✓ Appended {} to {}".format(entity_id, output_csv_path))
    else:
        # Create new CSV
        df_new.to_csv(output_csv_path, index=False)
        print("✓ Created new CSV: {}".format(output_csv_path))

    # Print summary
    print("\nMetadata Summary for {}:".format(scene_id))
    print("  Acquisition Date: {}".format(metadata['Acquisition Date']))
    print("  Sun Elevation: {}°".format(metadata['Sun Elevation']))
    print("  Sun Azimuth: {}°".format(metadata['Sun Azimuth']))
    print("  Look Angle: {}°".format(metadata['Look Angle']))
    print("  Center Coordinates: ({}, {})".format(metadata['Center Latitude dec'], metadata['Center Longtude dec']))
    print("  Cloud Cover: {}%".format(metadata['Cloud Cover']))


if __name__ == '__main__':
    # ============================================================
    # CONFIGURATION
    # ============================================================

    # Base path
    basePath = 'C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/'

    # Path to XML metadata file (downloaded from USGS EarthExplorer)
    xml_path = basePath + 'Project/eo1_hyp_pub_EO1H2020342013284110KF_SG1_01.xml'

    # Output CSV path (will append to existing METADATA.csv)
    output_csv = basePath + 'METADATA/METADATA.csv'

    # ============================================================
    # RUN CONVERSION
    # ============================================================

    print('=' * 60)
    print('HYPERION METADATA XML TO CSV CONVERTER')
    print('=' * 60)
    print('\nInput XML: {}'.format(xml_path))
    print('Output CSV: {}'.format(output_csv))
    print()

    # Check if XML file exists
    if not os.path.exists(xml_path):
        print("ERROR: XML file not found: {}".format(xml_path))
        print("\nPlease download the metadata XML file from USGS EarthExplorer:")
        print("1. Go to https://earthexplorer.usgs.gov/")
        print("2. Find your image (EO1H2020342013284110KF)")
        print("3. Click 'Metadata' → 'Export Metadata'")
        print("4. Save as eo1_hyp_pub_EO1H2020342013284110KF_SG1_01.xml")
        exit(1)

    # Convert XML to CSV
    create_metadata_csv(xml_path, output_csv, append=True)

    print('\n' + '=' * 60)
    print('METADATA CONVERSION COMPLETE!')
    print('=' * 60)
    print('\nNext steps:')
    print('1. Update process_hyperion.py line 1051:')
    print("   fname = 'EO1H2020342013284110KF'")
    print('2. Run: python process_hyperion.py')
    print('=' * 60)
