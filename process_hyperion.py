"""
Hyperion Image Processing Script
Image: EO1H2020342016359110KF
Processing: L1R/L1T to Surface Reflectance with Topographic Correction
"""

import numpy as np
import os
import sys
import ee
import pandas as pd
from scipy import interpolate
from zipfile import ZipFile
from pathlib import Path
import rasterio
import subprocess

import surehyp.preprocess
import surehyp.atmoCorrection


def getGEEdem_fixed(UL_lat, UL_lon, UR_lat, UR_lon, LL_lat, LL_lon, LR_lat, LR_lon,
                    demID='USGS/SRTMGL1_003', elevationName='elevation', numPixels=1000):
    """
    Fixed version of getGEEdem that handles SRTM as Image instead of ImageCollection.
    Gets the average elevation of the scene from Google Earth Engine.
    """
    # Define the bounding box
    coord = ee.Geometry.Polygon([
        [UL_lon, UL_lat],
        [UR_lon, UR_lat],
        [LR_lon, LR_lat],
        [LL_lon, LL_lat],
        [UL_lon, UL_lat]
    ])

    # SRTM is an Image, not an ImageCollection
    DEM = ee.Image(demID)

    # Sample the DEM and get mean elevation
    result = DEM.sample(region=coord, numPixels=numPixels, scale=1000).getInfo()

    if result['features']:
        elevations = [f['properties'][elevationName] for f in result['features'] if elevationName in f['properties']]
        if elevations:
            altit = np.mean(elevations) / 1000.0  # Convert to km
            return altit

    # Fallback: try reduceRegion
    mean_elev = DEM.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=coord,
        scale=1000,
        maxPixels=1e9
    ).getInfo()

    if elevationName in mean_elev and mean_elev[elevationName] is not None:
        return mean_elev[elevationName] / 1000.0  # Convert to km

    raise ValueError(f"Could not retrieve elevation data from {demID}")


def processImage_fixed(fname, pathToImages, pathToImagesFiltered):
    """
    Fixed version of processImage that doesn't use tiled output
    (fixes the "TileWidth must be multiple of 16" error)
    """
    Path(pathToImages + 'tmp/').mkdir(parents=True, exist_ok=True)
    Path(pathToImagesFiltered).mkdir(parents=True, exist_ok=True)

    fpath = pathToImages + "/" + fname + '_1T.ZIP'
    with ZipFile(fpath, 'r') as zip:
        namelist = zip.namelist()
        namelist = [string for string in namelist if 'TIF' in string]
        zip.extractall(path=pathToImages + 'tmp/')
    namelist = [string for string in namelist if 'TIF' in string]

    # Read all bands files separately
    arrays = []
    for name in namelist:
        tmpImg = pathToImages + 'tmp/' + name
        src = rasterio.open(tmpImg, driver='GTiff', dtype=rasterio.int16)
        arrays.append(src.read(1))

    # Write all bands to a new file (without tiling to avoid dimension issues)
    profile = src.profile
    profile.update(count=len(arrays), nodata=0, tiled=False)  # Changed: tiled=False

    with rasterio.open(pathToImagesFiltered + fname + '.TIF', 'w', compress='lzw', **profile) as dst:
        k = 1
        for array in arrays:
            dst.write(array.astype(rasterio.int16), k)
            k += 1

    # Clean up tmp folder
    import shutil
    shutil.rmtree(pathToImages + 'tmp/', ignore_errors=True)


def preprocess_radiance(fname, pathToL1Rmetadata, pathToL1Rimages, pathToL1Timages,
                        pathToL1TimagesFiltered, pathOut, nameOut,
                        destripingMethod='Pal', localDestriping=False,
                        smileCorrectionOrder=2, checkSmile=False):
    """
    Preprocess L1R radiance data: desmiling, destriping, georeferencing
    """
    print('=' * 60)
    print('STEP 1: PREPROCESSING RADIANCE')
    print('=' * 60)

    print('\n[1/10] Concatenate the L1T image')
    processImage_fixed(fname, pathToL1Timages, pathToL1TimagesFiltered)

    print('\n[2/10] Read the L1R image')
    arrayL1R = surehyp.preprocess.readL1R(pathToL1Rimages + fname + '/', fname)

    print('\n[3/10] Get the L1R image parameters')
    metadata, bands, fwhms = surehyp.preprocess.getImageMetadata(pathToL1Rimages + fname + '/', fname)

    print('\n[4/10] Separate VNIR and SWIR')
    VNIR, VNIRb, VNIRfwhm, SWIR, SWIRb, SWIRfwhm = surehyp.preprocess.separating(arrayL1R, bands, fwhms)

    print('\n[5/10] Convert DN to radiance')
    VNIR, SWIR = surehyp.preprocess.DN2Radiance(VNIR, SWIR)

    print('\n[6/10] Align VNIR and SWIR, part 1')
    VNIR, SWIR = surehyp.preprocess.alignSWIR2VNIRpart1(VNIR, SWIR)

    print('\n[7/10] Desmiling')
    VNIR = surehyp.preprocess.smileCorrectionAll(VNIR, smileCorrectionOrder, check=checkSmile)
    SWIR = surehyp.preprocess.smileCorrectionAll(SWIR, smileCorrectionOrder, check=checkSmile)

    if destripingMethod == 'Datt':
        print('\n[8/10] Destriping - Datt (2003)')
        VNIR = surehyp.preprocess.destriping(VNIR, 'VNIR', 0.11)
        SWIR = surehyp.preprocess.destriping(SWIR, 'SWIR', 0.11)
    elif destripingMethod == 'Pal':
        print('\n[8/10] Destriping - Pal et al. (2020)')
        VNIR, nc = surehyp.preprocess.destriping_quadratic(VNIR)
        if localDestriping:
            VNIR = surehyp.preprocess.destriping_local(VNIR, nc)
        SWIR, nc = surehyp.preprocess.destriping_quadratic(SWIR)
        if localDestriping:
            SWIR = surehyp.preprocess.destriping_local(SWIR, nc)
    else:
        print('\n[8/10] No destriping method selected -> skipping')

    print('\n[9/10] Assemble VNIR and SWIR')
    arrayL1R, wavelengths, fwhms = surehyp.preprocess.concatenateImages(VNIR, VNIRb, VNIRfwhm, SWIR, SWIRb, SWIRfwhm)

    print('\n[9b/10] Smooth the cirrus bands for later thin cirrus removal')
    arrayL1R = surehyp.preprocess.smoothCirrusBand(arrayL1R, wavelengths)

    print('\n[10/10] Georeference the corrected L1R data using L1T data')
    arrayL1Rgeoreferenced, metadataGeoreferenced = surehyp.preprocess.georeferencing(arrayL1R, pathToL1TimagesFiltered, fname)

    print('\nSaving the processed image as an ENVI file...')
    surehyp.preprocess.savePreprocessedL1R(arrayL1Rgeoreferenced, wavelengths, fwhms, metadataGeoreferenced,
                                            pathToL1Rimages, pathToL1Rmetadata, metadata, fname, pathOut + nameOut)

    # Cleanup temporary files
    for f in os.listdir(pathOut):
        if (fname in f) and ('_tmp' in f):
            os.remove(os.path.join(pathOut, f))

    print('\n' + '=' * 60)
    print('PREPROCESSING COMPLETE!')
    print(f'Output saved to: {pathOut + nameOut}')
    print('=' * 60)

    return pathOut + nameOut


def atmospheric_correction(pathToRadianceImage, pathToOutImage, stepAltit=1, stepTilt=15,
                           stepWazim=30, demID='USGS/SRTMGL1_003', elevationName='elevation',
                           topo=True, smartsAlbedoFilePath=None):
    """
    Perform atmospheric correction with optional topographic correction
    """
    print('\n' + '=' * 60)
    print('STEP 2: ATMOSPHERIC CORRECTION')
    print('=' * 60)

    print('\n[1/12] Open processed radiance image')
    L, bands, fwhms, processing_metadata, metadata = surehyp.atmoCorrection.getImageAndParameters(pathToRadianceImage)

    # Extract metadata for clearer visualization
    longit = processing_metadata['longit']
    latit = processing_metadata['latit']
    datestamp1 = processing_metadata['datestamp1']
    zenith = processing_metadata['zenith']
    azimuth = processing_metadata['azimuth']
    satelliteZenith = np.abs(processing_metadata['satelliteZenith'])
    satelliteAzimuth = processing_metadata['satelliteAzimuth']

    UL_lat = processing_metadata['UL_lat']
    UL_lon = processing_metadata['UL_lon']
    UR_lat = processing_metadata['UR_lat']
    UR_lon = processing_metadata['UR_lon']
    LL_lat = processing_metadata['LL_lat']
    LL_lon = processing_metadata['LL_lon']
    LR_lat = processing_metadata['LR_lat']
    LR_lon = processing_metadata['LR_lon']

    year = processing_metadata['year']
    doy = processing_metadata['doy']

    print(f'    Image center: ({latit:.4f}, {longit:.4f})')
    print(f'    Year: {year}, DOY: {doy}')
    print(f'    Sun zenith: {zenith:.2f}, Sun azimuth: {azimuth:.2f}')

    if topo:
        print('\n[2/12] Download DEM images from GEE')
        path_to_dem = surehyp.atmoCorrection.getDEMimages(UL_lon, UL_lat, UR_lon, UR_lat,
                                                          LR_lon, LR_lat, LL_lon, LL_lat,
                                                          demID=demID, elevationName=elevationName)

        print('\n[3/12] Reproject DEM images')
        path_to_reprojected_dem = surehyp.atmoCorrection.reprojectDEM(pathToRadianceImage, path_elev=path_to_dem)

        print('\n[4/12] Resampling DEM')
        path_elev = surehyp.atmoCorrection.matchResolution(pathToRadianceImage, path_elev=path_to_reprojected_dem)

        print("\n[5/12] Extract DEM data for Hyperion image pixels")
        elev, slope, wazim = surehyp.atmoCorrection.extractDEMdata(pathToRadianceImage, path_elev=path_elev)
    else:
        slope = None
        wazim = None

    # NOTE: cloudAndShadowsDetection not available in installed surehyp version
    # Skipping cloud/shadow detection - all pixels treated as clear
    print('\n[6/12] Cloud/shadow detection (skipped - not available in installed version)')
    clearview = np.ones(L.shape[:2], dtype=np.uint8)
    print('    All pixels marked as clearview')

    print('\n[7/12] Get haze spectrum (dark object subtraction)')
    L, Lhaze = surehyp.atmoCorrection.darkObjectDehazing(L, bands)

    print('\n[8/12] Mask non-clearview pixels (skipped)')
    # L[clearview == 0] = 0  # Skipped since all pixels are clearview

    print('\n[9/12] Removal of thin cirrus')
    L, cirrus_cloudMask = surehyp.atmoCorrection.cirrusRemoval(bands, L, latit, doy, satelliteZenith, zenith, azimuth)

    print('\n[10/12] Get average elevation of the scene from GEE')
    # Use fixed version that handles SRTM as Image (not ImageCollection)
    altit = getGEEdem_fixed(UL_lat, UL_lon, UR_lat, UR_lon,
                            LL_lat, LL_lon, LR_lat, LR_lon,
                            demID=demID, elevationName=elevationName)
    print(f'    Average altitude: {altit*1000:.1f} m ({altit:.3f} km)')

    print('\n[11/12] Get atmospheric parameters')
    try:
        wv, o3, flag_no_o3 = surehyp.atmoCorrection.getAtmosphericParameters(bands, L, datestamp1, year, doy,
                                                                              longit, latit, altit,
                                                                              satelliteZenith, zenith, azimuth)
        print(f'    Water vapor: {wv:.2f} cm')
        print(f'    Ozone: {o3:.3f} atm-cm')
        if flag_no_o3:
            IO3 = 1
        else:
            IO3 = 0
    except (ValueError, Exception) as e:
        print(f'    WARNING: Could not compute atmospheric parameters from image: {e}')
        print('    Using default values for mid-latitude atmosphere')
        # Use typical mid-latitude winter values (Dec 24)
        # Water vapor ~ 1.5 cm for winter, Ozone ~ 0.35 atm-cm
        wv = 1.5  # precipitable water vapor in cm
        o3 = 0.35  # ozone column in atm-cm
        IO3 = 0  # Use the specified ozone value (not SMARTS default)
        print(f'    Water vapor (default): {wv:.2f} cm')
        print(f'    Ozone (default): {o3:.3f} atm-cm')

    print('\n[12/12] Obtain radiative transfer outputs from SMARTS')
    # Get atmosphere parameters for sun-ground section
    df = surehyp.atmoCorrection.runSMARTS(ALTIT=altit, LATIT=latit, IMASS=0, ZENITH=zenith,
                                           AZIM=azimuth, SUNCOR=surehyp.atmoCorrection.get_SUNCOR(doy),
                                           IH2O=0, WV=wv, IO3=IO3, IALT=0, AbO3=o3)

    # Get atmosphere parameters for ground-satellite section
    df_gs = surehyp.atmoCorrection.runSMARTS(ALTIT=altit, LATIT=0, LONGIT=0, IMASS=0,
                                              SUNCOR=surehyp.atmoCorrection.get_SUNCOR(doy),
                                              ITURB=5, ZENITH=satelliteZenith, AZIM=0,
                                              IH2O=0, WV=wv, IO3=IO3, IALT=0, AbO3=o3)

    print('\nComputing radiance to reflectance conversion...')
    R = surehyp.atmoCorrection.computeLtoR(L, bands, df, df_gs)

    if not topo:
        print('\nSaving the reflectance image (flat surface)...')
        surehyp.atmoCorrection.saveRimage(R, metadata, pathToOutImage)
    else:
        print('\n--- TOPOGRAPHIC CORRECTION ---')

        if smartsAlbedoFilePath is None:
            smartsAlbedoFilePath = os.environ['SMARTSPATH'] + 'Albedo/Albedo.txt'

        print('\nWriting Albedo.txt file for SMARTS')
        pathToAlbedoFile = surehyp.atmoCorrection.writeAlbedoFile(R, bands, pathOut=smartsAlbedoFilePath)

        print('\nGetting scene background reflectance')
        sp = pd.read_csv(pathToAlbedoFile, header=3, sep=r'\s+')
        w = sp.values[:, 0]
        r = sp.values[:, 1]
        f = interpolate.interp1d(w, r, bounds_error=False, fill_value='extrapolate')
        rho_background = f(df['Wvlgth'] * 1E-3)

        print('\nComputing LUT for rough terrain correction')
        R = surehyp.atmoCorrection.getDemReflectance(altitMap=elev, tiltMap=slope, wazimMap=wazim,
                                                      stepAltit=stepAltit, stepTilt=stepTilt,
                                                      stepWazim=stepWazim, latit=latit,
                                                      IH2O=0, WV=wv, IO3=IO3, IALT=0, AbO3=o3,
                                                      doy=doy, zenith=zenith, azimuth=azimuth,
                                                      satelliteZenith=satelliteZenith,
                                                      satelliteAzimuth=satelliteAzimuth,
                                                      L=L, bands=bands, IALBDX=1,
                                                      rho_background=rho_background)

        print('\nApplying Modified-Minnaert topography correction')
        R = surehyp.atmoCorrection.MM_topo_correction(R, bands, slope * np.pi / 180,
                                                       wazim * np.pi / 180, zenith * np.pi / 180,
                                                       azimuth * np.pi / 180)

        print('\nSaving the reflectance image (with topographic correction)...')
        surehyp.atmoCorrection.saveRimage(R, metadata, pathToOutImage)

    print('\n' + '=' * 60)
    print('ATMOSPHERIC CORRECTION COMPLETE!')
    print(f'Output saved to: {pathToOutImage}')
    print('=' * 60)

    return pathToOutImage


if __name__ == '__main__':

    # ============================================================
    # CONFIGURATION
    # ============================================================

    # ============================================================
    # GOOGLE EARTH ENGINE PROJECT ID
    # ============================================================
    # IMPORTANT: You MUST set your GEE project ID below!
    #
    # To get a project ID:
    # 1. Go to https://code.earthengine.google.com/
    # 2. If prompted, register for Earth Engine
    # 3. Your project ID will be shown in the URL or in your account settings
    #    It typically looks like: 'ee-yourusername' or 'your-project-name'
    # 4. Replace 'YOUR_PROJECT_ID' below with your actual project ID
    # ============================================================

    GEE_PROJECT_ID = 'remote-sensing-478802'  # Your GEE Project ID

    print('Initializing Google Earth Engine...')

    if GEE_PROJECT_ID == 'YOUR_PROJECT_ID':
        print('\n' + '=' * 60)
        print('ERROR: GEE Project ID not configured!')
        print('=' * 60)
        print('\nPlease edit process_hyperion.py and set your GEE project ID.')
        print('Look for the line: GEE_PROJECT_ID = "YOUR_PROJECT_ID"')
        print('\nTo get a project ID:')
        print('1. Go to https://code.earthengine.google.com/')
        print('2. Register/sign in with your Google account')
        print('3. Find your project ID (e.g., "ee-yourusername")')
        print('=' * 60)
        raise ValueError('GEE Project ID not configured. Please edit the script.')

    ee.Initialize(project=GEE_PROJECT_ID)
    print(f'GEE initialized with project: {GEE_PROJECT_ID}')

    # SMARTS configuration
    smartsPath = 'C:/Program Files/SMARTS_295_PC/'
    os.environ['SMARTSPATH'] = smartsPath

    # Add SMARTS to PATH so the executable can be found
    if smartsPath not in os.environ['PATH']:
        os.environ['PATH'] = smartsPath + os.pathsep + os.environ['PATH']

    surehyp.atmoCorrection.smartsVersion = 'smarts295'
    surehyp.atmoCorrection.smartsExecutable = 'smarts295bat.exe'
    print(f"SMARTS path: {smartsPath}")

    # ============================================================
    # PATHS CONFIGURATION
    # ============================================================

    basePath = 'C:/Lorien/Archivos/TUBAF/1st_Semester/Remote_Sensing/'

    # Path to USGS Hyperion metadata (optional, can be empty file)
    pathToL1Rmetadata = basePath + 'METADATA/METADATA.csv'

    # Path to L1R images (uncompressed folders)
    pathToL1Rimages = basePath + 'L1R/'

    # Path to L1T images (TIF files)
    pathToL1Timages = basePath + 'L1T/'

    # Path for filtered/processed L1T images
    pathToL1TimagesFiltered = basePath + 'L1T/filteredImages/'

    # Output folder
    pathOut = basePath + 'OUT/'

    # ============================================================
    # IMAGE CONFIGURATION
    # ============================================================

    # Hyperion image ID (folder name)
    fname = 'EO1H2020342016359110KF'

    # Output names
    nameOut_radiance = fname + '_preprocessed'
    nameOut_reflectance = fname + '_reflectance'

    # ============================================================
    # PROCESSING OPTIONS
    # ============================================================

    # Destriping method: 'Pal' (recommended) or 'Datt'
    destripingMethod = 'Pal'

    # Local destriping refinement (slower but better results)
    localDestriping = False

    # Topographic correction
    # NOTE: Disabled due to geetools version incompatibility with SUREHYP
    # To enable, install geetools==0.6.14: pip install geetools==0.6.14
    use_topo = False

    # DEM source from Google Earth Engine
    # Options: 'USGS/SRTMGL1_003' (global), 'NRCan/CDEM' (Canada), etc.
    demID = 'USGS/SRTMGL1_003'
    elevationName = 'elevation'

    # ============================================================
    # RUN PROCESSING
    # ============================================================

    print('\n' + '=' * 60)
    print('HYPERION IMAGE PROCESSING')
    print(f'Image ID: {fname}')
    print('=' * 60)

    # STEP 1: Preprocess radiance
    # Check if preprocessed file already exists
    pathToRadianceImage = pathOut + nameOut_radiance
    if os.path.exists(pathToRadianceImage + '.bip'):
        print('\nPreprocessed radiance file already exists, skipping Step 1...')
        print(f'Using: {pathToRadianceImage}')
    else:
        pathToRadianceImage = preprocess_radiance(
            fname,
            pathToL1Rmetadata,
            pathToL1Rimages,
            pathToL1Timages,
            pathToL1TimagesFiltered,
            pathOut,
            nameOut_radiance,
            destripingMethod=destripingMethod,
            localDestriping=localDestriping,
            checkSmile=False
        )

    # STEP 2: Atmospheric correction with topographic correction
    atmospheric_correction(
        pathToRadianceImage,
        pathOut + nameOut_reflectance,
        topo=use_topo,
        demID=demID,
        elevationName=elevationName,
        smartsAlbedoFilePath=os.environ['SMARTSPATH'] + 'Albedo/Albedo.txt'
    )

    print('\n' + '=' * 60)
    print('ALL PROCESSING COMPLETE!')
    print('=' * 60)
    print(f'\nOutputs:')
    print(f'  - Preprocessed radiance: {pathOut + nameOut_radiance}.bip')
    print(f'  - Surface reflectance:   {pathOut + nameOut_reflectance}.bip')
    print(f'  - Cloud mask:            {pathOut + nameOut_reflectance}_cloud_mask.npy')
    print(f'  - Shadow mask:           {pathOut + nameOut_reflectance}_shadows_mask.npy')
    print(f'  - Clearview mask:        {pathOut + nameOut_reflectance}_clearview_mask.npy')
