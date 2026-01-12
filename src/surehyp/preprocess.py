import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import os
import time
import cv2
from pathlib import Path
from zipfile import ZipFile
from pyhdf.SD import SD, SDC
from scipy.ndimage.filters import median_filter,uniform_filter1d, uniform_filter
from scipy.signal import savgol_filter, find_peaks
from sklearn.decomposition import PCA
import spectral.io.envi as envi
import surehyp.various
np.seterr(invalid='ignore')


def processImage(fname,pathToImages,pathToImagesFiltered):
    '''
    compiles all TIF bands of the Hyperion L1T image fname into a single TIF image and saves it
        Parameters:
            :class:`string`:
                - fname: ID of the hyperion image, e.g. EO1H0110262016254110KF
                - pathToImages: path to the folder containing the image folder
                - pathToImagesFiltered: destination folder for the processed images
        Returns:
            nothing
    '''
    
    Path(pathToImages+'tmp/').mkdir(parents=True, exist_ok=True)
    Path(pathToImagesFiltered).mkdir(parents=True, exist_ok=True)

    fpath=pathToImages+"/"+fname+'_1T.ZIP'
    with ZipFile(fpath,'r') as zip:
        namelist=zip.namelist()
        namelist = [string for string in namelist if 'TIF' in string]
        zip.extractall(path=pathToImages+'tmp/')
    namelist = [string for string in namelist if 'TIF' in string]
    #read all bands files separately
    arrays=[]
    for name in namelist:
        tmpImg=pathToImages+'tmp/'+name
        src=rasterio.open(tmpImg,driver='GTiff',dtype=rasterio.int16)
        arrays.append(src.read(1))
    #write all bands to a new file
    profile=src.profile
    profile.update(count=len(arrays),nodata=0,tiled=True)
    with rasterio.open(pathToImagesFiltered+fname+'.TIF','w',compress='lzw',predictor=2,**profile) as dst:
        k=1
        for array in arrays:
            dst.write(array.astype(rasterio.int16),k)
            k+=1
        dst.close()
    src.close()
    #cleans tmp
    for f in os.listdir(pathToImages+'tmp/'):
        if fname in f:
            os.remove(os.path.join(pathToImages+'tmp/',f))

def getAcquisitionsProperties(pathToL1Rmetadata,fname=None):
    '''
    reads the metadata file exported from https://earthexplorer.usgs.gov/ to get the acquisition properties of each Hyperion image necessary for the processing
    
        Parameters:
            :class:`string`:
                pathToL1Rmetadata: path to the Hyperion image metadata csv as downloaded from the usgs website
                fname: Hyperion image name e.g. EO1H0110262016254110KF

        Returns:
            :class:`np.float`:
                - sunZeniths: sun zenith angle in degrees
                - sunAzimuth: sun azimuth angle in degrees
                - satelliteZeniths: satellite zenith angle in degrees
                - satelliteAzimuth: satellite azimuth angle in degrees
                - centerLon, centerLat: center longitude and latitude of the Hyperion image in decimal degrees
            :class:`string`:
                - acquisition date in YYYY-MM-DD or YYYY/MM/DD format
                - acquisitionTimeStart/acquisitionTimeStop: start and end times of the acquisition in YYYY:DOY:HH:mm:ss format
    '''

    df=pd.read_csv(pathToL1Rmetadata,encoding='unicode_escape')
    sunZeniths=-(df['Sun Elevation'].values.astype(float)-90)
    sunAzimuths=df['Sun Azimuth'].values.astype(float)
    satelliteZeniths=df['Look Angle'].values.astype(float)
    satelliteAzimuths=df['Satellite Inclination'].values.astype(float)
    centerLons=df['Center Longtude dec'].values.astype(float)
    centerLats=df['Center Latitude dec'].values.astype(float)
    acquisitionDates=df['Acquisition Date'].values
    acquisitionTimeStarts=df['Scene Start Time'].values
    acquisitionTimeStops=df['Scene Stop Time'].values
    IDs=np.asarray(df['Entity ID'].map(lambda x: x[:-7]).values.tolist())
    if fname==None:
        return sunZeniths,sunAzimuths,satelliteZeniths,satelliteAzimuths,centerLons,centerLats,acquisitionDates,acquisitionTimeStarts,acquisitionTimeStops,IDs
    else:
        return sunZeniths[IDs==fname][0],sunAzimuths[IDs==fname][0],satelliteZeniths[IDs==fname][0],satelliteAzimuths[IDs==fname][0],centerLons[IDs==fname][0],centerLats[IDs==fname][0],acquisitionDates[IDs==fname][0],acquisitionTimeStarts[IDs==fname][0],acquisitionTimeStops[IDs==fname][0],IDs[IDs==fname][0]

def getImageCorners(pathToL1Rimages,fname):
    '''
    Gets coordinates of the image corners
        Parameters:
            :class:`string`:
                - pathToL1Rimages: path to the L1R hyperion images folder
                - fname: ID of the Hyperion image

        Returns: 
            :class:`np.float`:
                - Latitudes/longitudes of the corners of the Hyperion image in clockwise order from upper left
    '''

    with open(pathToL1Rimages+fname+'/'+fname+'.MET') as file:
        lines=file.readlines()
        UL_lat=lines[-9][24:-1]
        UL_lon=lines[-8][24:-1]
        UR_lat=lines[-7][24:-1]
        UR_lon=lines[-6][24:-1]
        LL_lat=lines[-5][24:-1]
        LL_lon=lines[-4][24:-1]
        LR_lat=lines[-3][24:-1]
        LR_lon=lines[-2][24:-1]
    return UL_lat, UL_lon, UR_lat, UR_lon, LL_lat, LL_lon, LR_lat, LR_lon

def readL1R(path,fname):
    '''
    Reads a L1R image file
        Parameters:
            :class:`string`:
                - path: path to the L1R image folder
                - fname: ID of the hyperion image
        Returns:
            :class:`np.array`:
                - (m,n,b) array containing the L1R data, with b the number of bands
    '''

    #reads the L1R HDF4 file and reorganizes the axes so that the bands are on axis 2
    #returns the hyperspectral array
    file = SD(path+'/'+fname+'.L1R', SDC.READ)
    img = file.select(fname+'.L1R',) # select sds
    array=img[:,:,:].copy().astype(float)
    array=np.swapaxes(array,1,2)
    return array

def getImageMetadata(path,fname):
    '''
    Retrieves the image metadata
        Parameters:
            :class:`string`:
                - path: path to the L1R image folder
                - fname: ID of the hyperion image

        Returns:
            :class:`dict`:
                - the image metadata
            :class:`np.array`:
                - (b,) array containing the bands wavelength
                - (b,) array containing the bands Full-Widths at Half Maximum
    '''

    #returns the metadata of the image
    img=envi.open(path+'/'+fname+'.hdr',path+'/'+fname+'.L1R') #!! problem when reading with spectral, I suspect it fills it considers that bands and samples are swapped. would probably need to reshape the array to correct it
    return img.metadata.copy(), np.asarray(img.metadata['wavelength']).astype(float), np.asarray(img.metadata['fwhm']).astype(float)\

def separating(data3D,bands,fwhms):
    '''
    separates the Hyperion array into VNIR and SWIR, removes unused bands
        Parameters:
            :class:`np.array`:
                - data3D: L1R data -- (m,n,b) array
                - bands: wavelengths of the image bands -- (b,) array
                - fwhms: FWHMs of the image bands -- (b,) array
        Returns:
            :class:`np.array`:
                - VNIR: array corresponding to the Hyperion VNIR data
                - VNIRb: spectral bands of the VNIR data
                - VNIRfwhm: full width at half maximum for each band of the VNIR data
                - SWIR: array corresponding to the Hyperion SWIR data
                - SWIRb: spectral bands of the SWIR data
                - SWIRfwhm: full width at half maximum for each band of the SWIR data
    '''

    # Bad bands removed:
    # - Bands 1-7: Uncalibrated VNIR
    # - Bands 58-77: Overlap region and water vapor absorption
    # - Bands 224-242: Uncalibrated SWIR with low SNR (FIXED: was 223, now 224)
    VNIR=data3D[:,:,7:56]
    VNIRb=bands[7:56]
    VNIRfwhm=fwhms[7:56]
    SWIR=data3D[:,:,77:224]  # FIXED: Changed from 223 to 224 to properly remove bands 224-242
    SWIRb=bands[77:224]
    SWIRfwhm=fwhms[77:224]
    return VNIR,VNIRb,VNIRfwhm,SWIR,SWIRb,SWIRfwhm

def DN2Radiance(VNIR,SWIR):
    '''
    converts digital numbers to radiance
    40 for VNIR, 80 for SWIR
        Parameters:
            :class:`np.array`:
                - VNIR,SWIR: (m,n,b) DN arrays
        Returns:
            :class:`np.array`:
                - VNIR,SWIR: (m,n,b) arrays in radiance
    '''

    return VNIR/40, SWIR/80

def alignSWIR2VNIRpart1(VNIR,SWIR):
    '''
    first part of the alignment between VNIR and SWIR: moves the right side of the SWIR one pixel up (Khurshid et al., 2006)
        Parameters:
            :class:`np.array`:
                - VNIR,SWIR: (m,n,b) arrays
        Returns: 
            :class:`np.array`:
                - aligned VNIR and SWIR arrays (m-2,n-2,b) to account for the bad bands on the sides
    '''

    SWIR[0:-1,128:,:]=SWIR[1:,128:,:]
    SWIR=SWIR[:-1,1:-1,:]
    VNIR=VNIR[:-1,1:-1,:]
    return VNIR,SWIR

def smileCorrectionAll(array,degree,check=False):
    '''
    desmiles the images using the method presented by San and Suzen (2011)
    while in several papers states the smile is in the first band of the MNF (band 0), it is not always the case. The present algorithm searches for the smile band by fitting a `degree` order polynomial function on the column-averaged MNF band. The bands for which the coefficients of order `degree` are above mean+3*std of the coefficients of order `degree` for all bands are assumed to be the smiled bands.
    This finding is empirical and led to correct selection of the smiled bands over images EO1H0430332015166110KF, EO1H0430332015288110KF, EO1H0430332014136110P3, EO1H0430332015166110KF, EO1H0430332013149110K4 and EO1H0190262011062110K3 using order 2.
        Parameters:
            :class:`np.array`:
                - array: (m,n,b) array to desmile
            :class:`float`:
                - degree: degree of the polynomial function used for the desmiling
            :class:`bool`:
                - check: boolean flag  to ask for a figure showing the before/after desmiling

        Returns:
            :class:`np.array`:
                - Desmiled (m,n,b) array
    '''
    
    pca = PCA(whiten=True)
    h, w, numBands = array.shape
    X = np.reshape(array, (w*h, numBands))
    pca.fit(X)
    transformed_X = pca.transform(X)
    mnfArray=np.reshape(transformed_X, (h, w, numBands))

    if check==True:
        plotCheckSmile(mnfArray)
    colMean=np.mean(mnfArray[:,:,:],axis=0)
    colMean=savgol_filter(colMean,21,0,axis=0)

    bandMean=np.mean(colMean,axis=0)
    x=np.arange(colMean.shape[0])
    coefs,regOut=np.polynomial.polynomial.polyfit(x,colMean,degree,full=True)

    coefsStd=np.std(coefs[-1,:])
    coefsMean=np.mean(coefs[-1,:])
    b=np.argwhere(np.abs(coefs[-1,:])>coefsMean+3*coefsStd).squeeze()

    x=np.tile(np.arange(colMean.shape[0]),(mnfArray.shape[2],1)).T
    fit=fpoly(coefs,x)
    corrMnfArray=mnfArray.copy()
    bandMean=np.tile(np.tile(bandMean,(fit.shape[0],1)),(corrMnfArray.shape[0],1,1))
    fit=np.tile(fit,(corrMnfArray.shape[0],1,1))
    corrMnfArray[:,:,b]=corrMnfArray[:,:,b]+np.squeeze(bandMean[:,:,b]-fit[:,:,b])
    corrTransformedX = np.reshape(corrMnfArray, (w*h, numBands))
    corrX=pca.inverse_transform(corrTransformedX)
    corrArray=np.reshape(corrX, (h, w, numBands))

    return corrArray

def fpoly(c,x):
    '''
    returns the polynoms' values over each point x for all of the n polynoms
        Parameters:
            :class:`np.array`:
                - c: array containing the k coefficients of each of the n polymoms fitting the n functions -- (k,m) array
                - x: x values of the polynoms -- (n) array
        Returns:
            :class:`np.array`:
                - out: (m,n) array
    '''

    out=0
    for k in np.arange(c.shape[0]):
        try:
            out+=c[k]*x**k
        except:
            out+=c[k,:]*x**k
    return out

def destriping(array,srange,threshold):
    '''
    destripes the images according to the local method described by Datt et al. (2003)
    iterates the local neighbourhoods to remove a maximum of stripes
        Parameters: 
            :class:`np.array`:
                - array: the radiance array to destripe -- (m,n,b) array
            :class:`string`
                - srange: 'VNIR' or 'SWIR' depending on the part of the hyperion image the array if from
            :class:`float`
                - threshold: threshold value to detect outliers
        Returns:
            :class:`np.array`:
                - a destriped array -- (m,n,b) array
    '''

    ngbrh={'VNIR':21,'SWIR':41}
    mik=np.nanmedian(array,axis=0)
    sik=np.nanstd(array,axis=0)
    for i in np.arange(ngbrh[srange],5,-2):
        outlier=getLocalOutlier3D(array,mik,sik,i,threshold)
        array=localDestriping3D(array,mik,sik,i,outlier)
    return array

def getLocalOutlier3D(mik,sik,ngbrh,thres):
    '''
    return the columns identified as outliers
        Parameters:
            :class:`np.array`:
                - mik: median value of each column of the array -- (n,b) array
                - sik: std value of each column of the array -- (n,b) array
            :class:`int`
                - ngbrh: neighborhood to use for the outlier destection
            :class:`float`
                - threshold: threshold value to detect outliers
        Returns:
            :class:`np.array`:
                - outlier -- (n,b) array
    '''

    lmedmik=median_filter(mik,footprint=np.ones((ngbrh,1)),mode='reflect')
    lmedsik=median_filter(sik,footprint=np.ones((ngbrh,1)),mode='reflect')
    test=np.abs(mik-lmedmik)/lmedsik

    outlier=np.zeros(test.shape,dtype=bool)
    outlier[(test-np.nanmin(test,axis=0))>=thres]=True
    return outlier

def localDestriping3D(img,mik,sik,ngbrh,outlier):
    '''
    returns the image after correction of the columns marked as outliers
        Parameters:
            :class:`np.array`:
                - img: array to destripe -- (m,n,b) array
                - mik: median value of each column of the array -- (n,b) array
                - sik: std value of each column of the array -- (n,b) array
            :class:`int`
                - ngbrh: neighborhood to use for the outlier destection
            :class:`np.array`:
                - outlier -- (n,b) array
        Returns: 
            :class:`np.array`:
                - img: destriped (m,n,b) array
    '''
    mmik=uniform_filter1d(mik,ngbrh,axis=0)
    msik=uniform_filter1d(sik,ngbrh,axis=0)
    alpha=msik/sik
    beta=np.tile(mmik-alpha*mik,(img.shape[0],1,1))
    alpha=np.tile(msik/sik,(img.shape[0],1,1))
    outlier=np.tile(outlier,(img.shape[0],1,1))
    tmp=alpha*img+beta
    img[outlier==True]=tmp[outlier==True]
    img[img<=0]=np.nan
    return img



def destriping_quadratic(array):
    '''
    spanning-image destriping based on a method by Pal et al. (2020)
        Parameters: 
            :class:`np.array`:
                - array: input radiance image to destripe -- (m,n,b) array

        Returns:
            :class:`np.array`:
                - array+diff: radiance with the image destriped of the spanning-column stripes -- (m,n,b) array
                - ncs: width of the largest peaks/crests for each band -- (b,) array
    '''

    Pca=np.nanmedian(array,axis=0)
    Pfit=[]
    ncs=[]
    for b in np.arange(Pca.shape[1]):
        peaks,_=find_peaks(Pca[:,b])
        trough,_=find_peaks(-Pca[:,b])
        width=np.sort(np.append(peaks,trough))
        nc=int(np.ceil(np.median(np.sort(np.diff(width))[-int(width.size/5):])))
        ncs.append(nc)
        Pfit.append(savgol_filter(Pca[:,b],10*nc+1,2))
    Pfit=np.asarray(Pfit).T
    diff=Pfit-Pca
    diff=np.tile(diff,(array.shape[0],1,1))
    return array+diff, ncs

def destriping_local(array,ncs):
    '''
    non-spanning-image destriping based on a method by Pal et al. (2020)
        Parameters:
            :class:`np.array`:
                - array: input radiance image to destripe -- (m,n,b) array
                - ncs: width of the largest peaks/crests for each band -- (b,) array
        Returns:
            :class:`np.array`:
                - array:radiance with the image destriped of the non-spanning-column stripes -- (m,n,b) array
    '''
    #based on a method by Pal et al. (2020)
    for b in np.arange(array.shape[2]):
        size=3*ncs[b]
        if not surehyp.various.is_odd(size):
            size+=1

        Icorr_mean=uniform_filter(array[:,:,b],size=(size,size))
        Icorr_std=surehyp.various.window_stdev(array[:,:,b],(size,size))
        Icorr_diff=array[:,:,b]-Icorr_mean
        bad_pixels=np.zeros(array[:,:,b].shape)
        bad_pixels[Icorr_diff>Icorr_std]=1

        bad_columns=uniform_filter(bad_pixels,size=(size,1))

        idx=np.nonzero(bad_columns>0.9) #get faulty columns's centers
        idx0=idx[0]
        idx1=idx[1]
        tmp=np.arange(-(size-1)/2,(size-1)/2+1) #get surrounding pixels of the column
        tmp=np.tile(tmp,len(idx0))
        idx0=np.repeat(idx0,size)
        idx1=np.repeat(idx1,size)
        idx0=idx0+tmp
        idx1=idx1[idx0>=0]
        idx0=idx0[idx0>=0]
        idx1=idx1[idx0<array.shape[0]]
        idx0=idx0[idx0<array.shape[0]]
        idx0=idx0.astype(int)
        array[idx0,idx1,b]=Icorr_mean[idx0,idx1]
    return array


def alignSWIR2VNIRpart2(VNIR,VNIRb,SWIR,SWIRb):
    '''
    second step of the alignment of VNIR and SWIR: instead of a set rotation, looks for matching features in the images and apply an homography on the SWIR so that SWIR and VNIR are aligned
    use bands VNIR and SWIR bands at 925.41 amd 922.54
    or could have used bands VNIR and SWIR bands at 935.58 amd 932.64 (Thenkabail et al 2018)
        Parameters:
            :class:`np.array`:
                - VNIR,SWIR: (m,n,b1) and (m,n,b2) arrays
                - VNIRb,SWIRb: (b1,) and (b2,) arrays
        Returns:
            :class:`np.array`:
                - aligned VNIR and SWIR arrays -- (m,n,b1) and (m,n,b2) arrays
    '''

    VNIR=np.pad(VNIR,((4,4),(4,4),(0,0)))
    SWIR=np.pad(SWIR,((4,4),(4,4),(0,0)))
    MAX_FEATURES = 10000
    GOOD_MATCH_PERCENT = 0.15
    # Convert images to grayscale
    SWIRGray=cv2.normalize(src=SWIR[:,:,np.argmin(np.abs(SWIRb-922.54))], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    VNIRGray=cv2.normalize(src=VNIR[:,:,np.argmin(np.abs(VNIRb-925.41))], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES,edgeThreshold=4)
    keypoints1, descriptors1 = orb.detectAndCompute(SWIRGray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(VNIRGray, None)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    try:
        matches=sorted(matches, key=lambda x: x.distance, reverse=False)
    except:
        matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width, bands = VNIR.shape
    SWIRAligned = cv2.warpPerspective(SWIR, h, (width, height),flags=cv2.INTER_NEAREST)
    VNIR=VNIR[4:-4,4:-4,:]
    SWIR=SWIRAligned[4:-4,4:-4,:]
    return VNIR,SWIR

def concatenateImages(VNIR,VNIRb,VNIRfwhm,SWIR,SWIRb,SWIRfwhm):
    '''
    reassembles VNIR and SWIR
        Parameters:
            :class:`np.array`:
                - VNIR,SWIR: radiance arrays -- (m,n,b1) and (m,n,b2) arrays
                - VNIRb,SWIRb: wavelengths for each band -- (b1,) and (b2,) arrays
                - VNIRfwhm,SWIRfwhm: fwhm for each band -- (b1,) and (b2,) arrays
        Returns:
            :class:`np.array`:
                - imout: radiance image containing all VNIR and SWIR data -- (m,n,b1+b2) array
                - wavelengths: wavelength for each band -- (b1+b2,) array
                - fwhms: fwhm for each band -- (b1+b2,) array
    '''

    imout=np.concatenate((VNIR,SWIR),axis=2)
    wavelengths=np.concatenate((VNIRb,SWIRb))
    fwhms=np.concatenate((VNIRfwhm,SWIRfwhm))
    return imout, wavelengths, fwhms

def georeferencing(imgL1R,pathToGeoreferencedImage,fname):
    '''
    georeferences the L1R images using matching features between L1R and a georeferenced image (L1t or L1Gst)
        Parameters: 
            :class:`np.array`:
                - imgL1R: radiance array to georeference -- (m,n,b) array
            :class:`string`:
                - pathTOGeoreferencedImage: path to the georeferenced image
                - fname: ID of the georefenced hyperion image
        Returns:
            :class:`np.array`:
                - arrayL1RGeoreferenced: the georeferenced radiance array -- (o,p,b) array
            :class:`dict`:
                - imgL1Georeferenced.meta: the georefenced image's metadata
    '''

    try:
        imgL1Georeferenced=rasterio.open(pathToGeoreferencedImage+fname+'.TIF')
        arrayL1Georeferenced=imgL1Georeferenced.read()
        arrayL1Georeferenced=np.moveaxis(arrayL1Georeferenced,0,2)
    except:
        print('could not open the georeferenced file')
        raise
    #Georeference the L1R data using L1Georeferenced
    try:
        arrayL1RGeoreferenced,h,srcPoints,dstPoints=alignImages(imgL1R,arrayL1Georeferenced,bandIm1=33,bandIm2=40)
    except:
        print('could not align the images')
        raise
    return arrayL1RGeoreferenced, imgL1Georeferenced.meta

def alignImages(im1, im2,MAX_FEATURES=10000,GOOD_MATCH_PERCENT=0.25,bandIm1=33,bandIm2=40):
    '''
    align images using ORB features
        Parameters:
            :class:`np.array`:
                - im1: array to transform (source) -- (m,n,b1) array
                - im2: reference array (target) -- (o,p,b2) array
            :class:`int`:
                - MAX_FEATURES: maximum number of features to identify in each image
            :class:`float`:
                - GOOD_MATCH_PERCENT: percent of matches (0-1) considered as good, e.g. only the first 25% are considered good
            :class:`int`:
                - bandIm1, bandIm2: band of each image considered when looking for features. They must allow for the identification of similar features
        Returns:
            :class:`np.array`:
                - im1Reg: the aligned im1 array -- (o,p,b1) array
            :class:`opencv homography`:
                - h: the homography used
            :class:`list`:
                - points1: the features points of image 1
                - points2: the features points of image 2
    '''

    # Convert images to grayscale
    im1Gray=cv2.normalize(src=im1[:,:,bandIm1], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    im2Gray=cv2.normalize(src=im2[:,:,bandIm2], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES,edgeThreshold=0)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    try:
        matches=sorted(matches,key=lambda x: x.distance, reverse=False)
    except:
        matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    # Use homography
    height, width,bands = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height),flags=cv2.INTER_NEAREST)
    #check if registration was ok
    printIm2Gray=im2[:,:,bandIm2].copy()
    printIm2Gray[printIm2Gray>0]=1
    printIm1Reg=im1Reg[:,:,bandIm1].copy()
    printIm1Reg[printIm1Reg>0]=1
    verif=printIm1Reg+printIm2Gray
    return im1Reg, h, points1, points2

def smoothCirrusBand(array,bands,npass=2,size=5):
    '''
    smoothes the cirrus band
        Parameters:
            :class:`np.array`:
                - array: the radiance array -- (m,n,b) array
                - bands: wavelength associated to each band -- (b,) array
            :class:`int`:
                - npass: the number of passes to make with the smoothing filter
                - size: width of the smoothing filter
        Returns:
            :class:`np.array`:
                - array: the radiance array with a spatially smoother band at 1380 nm, e.g. the cirrus band -- (m,n,b) array
    '''
    for p in np.arange(npass):
        array[:,:,np.argmin(np.abs(bands-1380))]=median_filter(array[:,:,np.argmin(np.abs(bands-1380))],size=size)
    return array

def savePreprocessedL1R(arrayL1RGeoreferenced,wavelengths,fwhms,kwargs,pathToL1Rimages,pathToL1Rmetadata,metadataL1R,fname,pathOut,scaleFactor=1e3):
    '''
    Saves the corrected L1 data.
    Did not found a way to directly save the ENVI file with all the desired metadata (georeferencement and image acquisition properties) so first a temporary image is saved with Rasterio to pre-generate part of the header, then it is open and the actual image with complete header is saved using Spectral.io
        Parameters:
            :class:`np.array`:
                - arrayL1RGeoreferenced: array to save -- (m,n,b) array
                - wavelengths: wavelength associated to each band -- (b,) array
            :class:`dict`:
                - kwargs: rasterio arguments to use when saving the image using rasterio
            :class:`string`:
                - pathToL1Rimages: path to the L1R data
                - pathToL1Rmetadata: path to the Hyperion image metadata as downloaded from the USGS website
            :class:`dict`:
                - metadataL1R: medata of the Hyperion L1R data
            :class:`string`:
                - fname: Hyperion image ID
                - pathOut: path of the image to save
            :class:`int`:
                - scaleFactor: factor by which to multiply the image before saving as an unsigned int16 to save space
    '''
    
    #conversion to FLAASH units
    #L1R product in W/(m2 um sr) after scaling
    #FLAASH requires uW/(cm2 nm sr)
    arrayL1RGeoreferenced*=0.1
    arrayL1RGeoreferenced[np.isnan(arrayL1RGeoreferenced)]=0
    #save image as ENVI file for FLAASH
    arrayL1RGeoreferenced*=scaleFactor
    arrayL1RGeoreferenced[arrayL1RGeoreferenced<0]=0
    arrayL1RGeoreferenced[arrayL1RGeoreferenced>1e7]=1e7
    arrayL1RGeoreferenced=arrayL1RGeoreferenced.astype(rasterio.uint16)
    arrayL1RGeoreferenced=np.moveaxis(arrayL1RGeoreferenced,2,0)
    kwargs.update({
        'driver': 'ENVI',
        'interleave':'bsq',
        'dtype': rasterio.uint16,
        'count': 1,
        'width': arrayL1RGeoreferenced.shape[2],
        'height': arrayL1RGeoreferenced.shape[1],
        'tiled': True,
        'compress': 'lzw',
        'predictor': 2
    })

    with rasterio.open(pathOut+'_tmp','w',**kwargs) as out:
        out.write(arrayL1RGeoreferenced[0,:,:],1)
        out.close()

    arrayL1RGeoreferenced=np.moveaxis(arrayL1RGeoreferenced,0,2)
    bandNames=np.asarray(metadataL1R['band names'].copy())[np.r_[7:56,77:223]]
    sunZenith,sunAzimuth,satelliteZenith,satelliteAzimuth,centerLon,centerLat,acquisitionDate,acquisitionTimeStart,acquisitionTimeStop,ID=getAcquisitionsProperties(pathToL1Rmetadata,fname)

    UL_lat, UL_lon, UR_lat, UR_lon, LL_lat, LL_lon, LR_lat, LR_lon=getImageCorners(pathToL1Rimages,fname)

    img=envi.open(pathOut+'_tmp.hdr',pathOut+'_tmp')
    metadata=img.metadata.copy()
    metadata['data type']= rasterio.uint16
    metadata['interleave']='bip'
    metadata['wavelength']=wavelengths.tolist()
    metadata['fwhm']=fwhms.tolist()
    metadata['band names']=bandNames.tolist()
    metadata['scale factor']=scaleFactor
    metadata['sun zenith']=sunZenith
    metadata['sun azimuth']=sunAzimuth
    metadata['satellite zenith']=satelliteZenith
    metadata['satellite azimuth']=satelliteAzimuth
    metadata['center lon']=centerLon
    metadata['center lat']=centerLat
    metadata['acquisition date']=acquisitionDate
    metadata['acquisition time start']=acquisitionTimeStart
    metadata['acquisition time stop']=acquisitionTimeStop
    metadata['id']=ID

    metadata['ul_lat']=UL_lat
    metadata['ul_lon']=UL_lon
    metadata['ur_lat']=UR_lat
    metadata['ur_lon']=UR_lon
    metadata['ll_lat']=LL_lat
    metadata['ll_lon']=LL_lon
    metadata['lr_lat']=LR_lat
    metadata['lr_lon']=LR_lon
    envi.save_image(pathOut+'.hdr',arrayL1RGeoreferenced[:,:,:],metadata=metadata,force=True)

def plotCheckSmile(mnfArray):
    '''
    plots the first 10 bands of the MNF array
        Parameters:
            :class:`array`:
            mnfArray: the MNF array of the radiance image -- (m,n,b) array
    '''

    fig,ax=plt.subplots(2,5)
    ax[0,0].imshow(mnfArray[1500:2000,:,0],cmap='binary')
    ax[0,1].imshow(mnfArray[1500:2000,:,1],cmap='binary')
    ax[0,2].imshow(mnfArray[1500:2000,:,2],cmap='binary')
    ax[0,3].imshow(mnfArray[1500:2000,:,3],cmap='binary')
    ax[0,4].imshow(mnfArray[1500:2000,:,4],cmap='binary')

    ax[0,0].set_title('0')
    ax[0,1].set_title('1')
    ax[0,2].set_title('2')
    ax[0,3].set_title('3')
    ax[0,4].set_title('4')

    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[0,2].axis('off')
    ax[0,3].axis('off')
    ax[0,4].axis('off')
    ax[1,0].imshow(mnfArray[1500:2000,:,5],cmap='binary')
    ax[1,1].imshow(mnfArray[1500:2000,:,6],cmap='binary')
    ax[1,2].imshow(mnfArray[1500:2000,:,7],cmap='binary')
    ax[1,3].imshow(mnfArray[1500:2000,:,8],cmap='binary')
    ax[1,4].imshow(mnfArray[1500:2000,:,9],cmap='binary')
    ax[1,0].set_title('5')
    ax[1,1].set_title('6')
    ax[1,2].set_title('7')
    ax[1,3].set_title('8')
    ax[1,4].set_title('9')
    ax[1,0].axis('off')
    ax[1,1].axis('off')
    ax[1,2].axis('off')
    ax[1,3].axis('off')
    ax[1,4].axis('off')

    plt.show()
