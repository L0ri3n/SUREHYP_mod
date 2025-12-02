import matplotlib.pyplot as plt
import spectral.io.envi as envi
import numpy as np
import ee
import geetools
import os
from pathlib import Path
import skimage.measure
import skimage.morphology
import skimage.transform
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy import interpolate
try:
    from interp3d import interp_3d
    flag_interp3d=True
except:
    flag_interp3d=False
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
import richdem as rd
from tqdm.auto import tqdm
import pandas as pd
import subprocess
import surehyp.various
import surehyp.preprocess


np.seterr(invalid='ignore')

def runSMARTS(ALTIT=0.3,LATIT=48.1,LONGIT=-79.3,YEAR=2013,MONTH=9,DAY=26,HOUR=10,ITILT=0,TILT=None,WAZIM=None,TAU550=None,IMASS=0,ZENITH=None,AZIM=None,SUNCOR=1,doy=269,ITURB=5,VISI=None,IH2O=1,WV=None,IO3=1,IALT=None,AbO3=None,IALBDX=1,RHOX=None):
    '''
    returns a dataframe containing the radiative transfer parameters along the optical path
        Parameters:
            see the SMARTS documentation for details regarding the inputs and outputs
        Returns:
            a dataframe containing the parameters
    '''

    if doy<46:
        col=1
    elif doy<105:
        col=2
    elif doy<166:
        col=3
    elif doy<227:
        col=4
    elif doy<288:
        col=5
    else:
        col=6
    dico_aero={ #From FLAASH user manual
            (8,1):'SAW',(8,2):'SAW',(8,3):'SAW',(8,4):'MLW',(8,5):'MLW',(8,5):'SAW',
            (7,1):'SAW',(7,2):'SAW',(7,3):'MLW',(7,4):'MLW',(7,5):'MLW',(7,5):'SAW',
            (6,1):'MLW',(6,2):'MLW',(6,3):'MLW',(6,4):'SAS',(6,5):'SAS',(6,5):'MLW',
            (5,1):'MLW',(5,2):'MLW',(5,3):'SAS',(5,4):'SAS',(5,5):'SAS',(5,5):'SAS',
            (4,1):'SAS',(4,2):'SAS',(4,3):'SAS',(4,4):'MLS',(4,5):'MLS',(4,5):'SAS',
            (3,1):'MLS',(3,2):'MLS',(3,3):'MLS',(3,4):'TRL',(3,5):'TRL',(3,5):'MLS',
            (2,1):'TRL',(2,2):'TRL',(2,3):'TRL',(2,4):'TRL',(2,5):'TRL',(2,5):'TRL',
            (1,1):'TRL',(1,2):'TRL',(1,3):'TRL',(1,4):'TRL',(1,5):'TRL',(1,5):'TRL',
            (0,1):'TRL',(0,2):'TRL',(0,3):'TRL',(0,4):'TRL',(0,5):'TRL',(0,5):'TRL',
            (-1,1):'TRL',(-1,2):'TRL',(-1,3):'TRL',(-1,4):'TRL',(-1,5):'TRL',(-1,5):'TRL',
            (-2,1):'TRL',(-2,2):'TRL',(-2,3):'TRL',(-2,4):'MLS',(-2,5):'MLS',(-2,5):'TRL',
            (-3,1):'MLS',(-3,2):'MLS',(-3,3):'MLS',(-3,4):'MLS',(-3,5):'MLS',(-3,5):'MLS',
            (-4,1):'SAS',(-4,2):'SAS',(-4,3):'SAS',(-4,4):'SAS',(-4,5):'SAS',(-4,5):'SAS',
            (-5,1):'SAS',(-5,2):'SAS',(-5,3):'SAS',(-5,4):'MLW',(-5,5):'MLW',(-5,5):'SAS',
            (-6,1):'MLW',(-6,2):'MLW',(-6,3):'MLW',(-6,4):'MLW',(-6,5):'MLW',(-6,5):'MLW',
            (-7,1):'MLW',(-7,2):'MLW',(-7,3):'MLW',(-7,4):'MLW',(-7,5):'MLW',(-7,5):'MLW',
            (-8,1):'MLW',(-8,2):'MLW',(-8,3):'MLW',(-8,4):'SAW',(-8,5):'MLW',(-8,5):'MLW',
            }

    CMNT='test'
    ISPR='2'
    SPR=None
    if ALTIT!=None:
        ALTIT=str(np.round(ALTIT,3))
    HEIGHT='0'
    IATMOS='1'
    ATMOS=dico_aero[(np.round(LATIT/10,0),col)]
    LATIT=str(LATIT)
    RH=None
    TAIR=None
    SEASON=None
    TDAY=None
    IH2O=str(IH2O)
    W=str(WV)
    IO3=str(IO3)
    IALT=str(IALT)
    if AbO3!=None:
        AbO3=str(np.round(AbO3,5))
    IGAS='1'
    ILOAD=None
    ApCH2O=None
    ApCH4=None
    ApCO=None
    ApHNO2=None
    ApHNO3=None
    ApNO=None
    ApNO2=None
    ApNO3=None
    ApO3=None
    ApSO2=None
    qCO2='370'
    ISPCTR='9'
    AEROS='S&F_RURAL'
    ALPHA1=None
    ALPHA2=None
    OMEGL=None
    GG=None
    ITURB=str(ITURB)
    TAU5=None
    BETA=None
    BCHUEP=None
    RANGE=None
    VISI=str(VISI)
    if TAU550==None:
        TAU550=str(np.round(np.exp(-3.2755-0.15078*float(ALTIT)),5))
    IALBDX=str(IALBDX)
    RHOX=str(RHOX)
    ITILT=str(ITILT)
    IALBDG=IALBDX
    TILT=str(TILT)
    WAZIM=str(WAZIM)
    RHOG=None
    WLMN='350'
    WLMX='2600'
    SUNCOR=str(np.round(SUNCOR,5))
    SOLARC='1367'
    IPRT='2'
    WPMN='350'
    WPMX='2600'
    INTVL='1'
    IOUT='1 2 3 4 5 6 7 8 15 16 17 18 19 20 21 27 28 32'
    ICIRC='0'
    SLOPE=None
    APERT=None
    LIMIT=None
    ISCAN='0'
    IFILT=None
    WV1=None
    WV2=None
    STEP=None
    FWHM=None
    ILLUM='0'
    IUV='0'
    IMASS=str(IMASS)
    if ZENITH!=None:
        ZENITH=str(np.round(ZENITH,3))
    AZIM=str(AZIM)
    ELEV=None
    AMASS=None
    YEAR=str(int(YEAR))
    MONTH=str(int(MONTH))
    DAY=str(int(DAY))
    ZONE=(np.abs(LONGIT)-7.5)//15
    if (np.abs(LONGIT)-7.5)<0:
        ZONE=ZONE
    else:
        ZONE+=1
    ZONE=int(np.sign(LONGIT)*ZONE)
    HOUR=float(HOUR)+ZONE
    HOUR=str(np.round(HOUR,3))
    ZONE=str(ZONE)
    LONGIT=str(LONGIT)
    DSTEP=None

    output=smartsAll_original(CMNT, ISPR, SPR, ALTIT, HEIGHT, LATIT, IATMOS, ATMOS, RH, TAIR, SEASON, TDAY, IH2O, W, IO3, IALT, AbO3, IGAS, ILOAD, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO,ApNO2, ApNO3, ApO3, ApSO2, qCO2, ISPCTR, AEROS, ALPHA1, ALPHA2, OMEGL, GG, ITURB, TAU5, BETA, BCHUEP, RANGE, VISI, TAU550, IALBDX, RHOX, ITILT, IALBDG,TILT, WAZIM,  RHOG, WLMN, WLMX, SUNCOR, SOLARC, IPRT, WPMN, WPMX, INTVL, IOUT, ICIRC, SLOPE, APERT, LIMIT, ISCAN, IFILT, WV1, WV2, STEP, FWHM, ILLUM,IUV, IMASS, ZENITH, AZIM, ELEV, AMASS, YEAR, MONTH, DAY, HOUR, LONGIT, ZONE, DSTEP)

    return output

def smartsAll_original(CMNT, ISPR, SPR, ALTIT, HEIGHT, LATIT, IATMOS, ATMOS, RH, TAIR, SEASON, TDAY, IH2O, W, IO3, IALT, AbO3, IGAS, ILOAD, ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO,ApNO2, ApNO3, ApO3, ApSO2, qCO2, ISPCTR, AEROS, ALPHA1, ALPHA2, OMEGL, GG, ITURB, TAU5, BETA, BCHUEP, RANGE, VISI, TAU550, IALBDX, RHOX, ITILT, IALBDG,TILT, WAZIM,  RHOG, WLMN, WLMX, SUNCOR, SOLARC, IPRT, WPMN, WPMX, INTVL, IOUT, ICIRC, SLOPE, APERT, LIMIT, ISCAN, IFILT, WV1, WV2, STEP, FWHM, ILLUM,IUV, IMASS, ZENITH, AZIM, ELEV, AMASS, YEAR, MONTH, DAY, HOUR, LONGIT, ZONE, DSTEP):

    # Check if SMARTSPATH environment variable exists and change working
    # directory if it does.
    original_wd = None
    if 'SMARTSPATH' in os.environ:
        original_wd = os.getcwd()
        os.chdir(os.environ['SMARTSPATH'])

    try:
        os.remove(smartsVersion+'.inp.txt')
    except:
        pass
    try:
        os.remove(smartsVersion+'.out.txt')
    except:
        pass
    try:
        os.remove(smartsVersion+'.ext.txt')
    except:
        pass
    try:
        os.remove(smartsVersion+'.scn.txt')
    except:
        pass

    f = open(smartsVersion+'.inp.txt', 'w')

    IOTOT = len(IOUT.split())

    ## Card 1: Comment.
    if len(CMNT)>62:
        CMNT = CMNT[0:61]

    CMNT = CMNT.replace(" ", "_")
    CMNT = "'"+CMNT+"'"
    print('{}' . format(CMNT), file=f)

    ## Card 2: Site Pressure
    print('{}'.format(ISPR), file=f)

    ##Card 2a:
    if ISPR=='0':
       # case '0' #Just input pressure.
        print('{}'.format(SPR), file=f)
    elif ISPR=='1':
        # case '1' #Input pressure, altitude and height.
        print('{} {} {}'.format(SPR, ALTIT, HEIGHT), file=f)
    elif ISPR=='2':
        #case '2' #Input lat, alt and height
        print('{} {} {}'.format(LATIT, ALTIT, HEIGHT), file=f)
    else:
        print("ISPR Error. ISPR should be 0, 1 or 2. Currently ISPR = ", ISPR)

    ## Card 3: Atmosphere model
    print('{}'.format(IATMOS), file=f)

    ## Card 3a:
    if IATMOS=='0':
        #case '0' #Input TAIR, RH, SEASON, TDAY
        print('{} {} {} {}'.format(TAIR, RH, SEASON, TDAY), file=f)
    elif IATMOS=='1':
        #case '1' #Input reference atmosphere
        ATMOS = "'"+ATMOS+"'"
        print('{}'.format(ATMOS), file=f)

    ## Card 4: Water vapor data
    print('{}'.format(IH2O), file=f)

    ## Card 4a
    if IH2O=='0':
        #case '0'
        print('{}'.format(W), file=f)
    elif IH2O=='1':
        #case '1'
        #The subcard 4a is skipped
        pass  #      print("")

    ## Card 5: Ozone abundance
    print('{}'.format(IO3), file=f)

    ## Card 5a
    if IO3=='0':
        #case '0'
        print('{} {}'.format(IALT, AbO3), file=f)
    elif IO3=='1':
        #case '1'
        #The subcard 5a is skipped and default values are used from selected
        #reference atmosphere in Card 3.
        pass #      print("")

    ## Card 6: Gaseous absorption and atmospheric pollution
    print('{}'.format(IGAS), file=f)

    ## Card 6a:  Option for tropospheric pollution
    if IGAS=='0':
        # case '0'
        print('{}'.format(ILOAD), file=f)

        ## Card 6b: Concentration of Pollutants
        if ILOAD=='0':
            #case '0'
            print('{} {} {} {} {} {} {} {} {} {} '.format(ApCH2O, ApCH4, ApCO, ApHNO2, ApHNO3, ApNO, ApNO2, ApNO3, ApO3, ApSO2), file=f)
        elif ILOAD=='1':
            #case '1'
                #The subcard 6b is skipped and values of PRISTINE
                #ATMOSPHERIC conditions are assumed
            pass #     print("")
        elif ILOAD=='2' or ILOAD =='3' or ILOAD == '4':
            #case {'2', '3', '4'}
            #The subcard 6b is skipped and value of ILOAD will be used
            #as LIGHT POLLUTION (ILOAD = 2), MODERATE POLLUTION (ILOAD = 3),
            #and SEVERE POLLUTION (ILOAD = 4).
            pass #     print("")

    elif IGAS=='1':
        #case '1'
        #The subcard 6a is skipped, and values are for default average
        #profiles.
        pass

    ## Card 7:  CO2 columnar volumetric concentration (ppmv)
    print('{}'.format(qCO2), file=f)

    ## Card 7a: Option of proper extraterrestrial spectrum
    print('{}'.format(ISPCTR), file=f)

    ## Card 8: Aerosol model selection out of twelve
    AEROS = "'"+AEROS+"'"

    print('{}'.format(AEROS), file=f)

    ## Card 8a: If the aerosol model is 'USER' for user supplied information
    if AEROS=="'USER'":
        print('{} {} {} {}'.format(ALPHA1, ALPHA2, OMEGL, GG), file=f)
    else:
        #The subcard 8a is skipped
        pass #     print("")

    ## Card 9: Option to select turbidity model
    print('{}'.format(ITURB), file=f)

    ## Card 9a
    if ITURB=='0':
        #case '0'
        print('{}'.format(TAU5), file=f)
    elif ITURB=='1':
        #case '1'
        print('{}'.format(BETA), file=f)
    elif ITURB=='2':
        #case '2'
        print('{}'.format(BCHUEP), file=f)
    elif ITURB=='3':
        #case '3'
        print('{}'.format(RANGE), file=f)
    elif ITURB=='4':
        #case '4'
        print('{}'.format(VISI), file=f)
    elif ITURB=='5':
        #case '5'
        print('{}'.format(TAU550), file=f)
    else:
        print("Error: Card 9 needs to be input. Assign a valid value to ITURB = ", ITURB)

    ## Card 10:  Select zonal albedo
    print('{}'.format(IALBDX), file=f)

    ## Card 10a: Input fix broadband lambertial albedo RHOX
    if IALBDX == '-1':
        print('{}'.format(RHOX), file=f)
    else:
        pass #     print("")
        #The subcard 10a is skipped.

    ## Card 10b: Tilted surface calculation flag
    print('{}'.format(ITILT), file=f)

    ## Card 10c: Tilt surface calculation parameters
    if ITILT == '1':
        print('{} {} {}'.format(IALBDG, TILT, WAZIM), file=f)

        ##Card 10d: If tilt calculations are performed and zonal albedo of
        ##foreground.
        if IALBDG == '-1':
            print('{}'.format(RHOG), file=f)
        else:
            pass #     print("")
            #The subcard is skipped


    ## Card 11: Spectral ranges for calculations
    print('{} {} {} {}'.format(WLMN, WLMX, SUNCOR, SOLARC), file=f)

    ## Card 12: Output selection.
    print('{}'.format(IPRT), file=f)

    ## Card 12a: For spectral results (IPRT >= 1)
    if float(IPRT) >= 1:
        print('{} {} {}'.format(WPMN, WPMX, INTVL), file=f)

        ## Card 12b & Card 12c:
        if float(IPRT) == 2 or float(IPRT) == 3:
            print('{}'.format(IOTOT), file=f)
            print('{}'.format(IOUT), file=f)
        else:
            pass #     print("")
            #The subcards 12b and 12c are skipped.
    else:
        pass #     print("")
        #The subcard 12a is skipped

    ## Card 13: Circumsolar calculations
    print('{}'.format(ICIRC), file=f)

    ## Card 13a:  Simulated radiometer parameters
    if ICIRC == '1':
        print('{} {} {}'.format(SLOPE, APERT, LIMIT), file=f)
    else:
        pass #     print("")
        #The subcard 13a is skipped since no circumsolar calculations or
        #simulated radiometers have been requested.


    ## Card 14:  Scanning/Smoothing virtual filter postprocessor
    print('{}'.format(ISCAN), file=f)

    ## Card 14a:  Simulated radiometer parameters
    if ISCAN == '1':
        print('{} {} {} {} {}'.format(IFILT, WV1, WV2, STEP, FWHM), file=f)
    else:
        pass #     print("")
        #The subcard 14a is skipped since no postprocessing is simulated.

    ## Card 15: Illuminace, luminous efficacy and photosythetically active radiarion calculations
    print('{}'.format(ILLUM), file=f)

    ## Card 16: Special broadband UV calculations
    print('{}'.format(IUV), file=f)

    ## Card 17:  Option for solar position and air mass calculations
    print('{}'.format(IMASS), file=f)

    ## Card 17a: Solar position parameters:
    if IMASS=='0':
        #case '0' #Enter Zenith and Azimuth of the sun
        print('{} {}'.format(ZENITH, AZIM), file=f)
    elif IMASS=='1':
        #case '1' #Enter Elevation and Azimuth of the sun
        print('{} {}'.format(ELEV, AZIM), file=f)
    elif IMASS=='2':
        #case '2' #Enter air mass directly
        print('{}'.format(AMASS), file=f)
    elif IMASS=='3':
        #case '3' #Enter date, time and latitude
        print('{} {} {} {} {} {} {}'.format(YEAR, MONTH, DAY, HOUR, LATIT, LONGIT, ZONE), file=f)
    elif IMASS=='4':
        #case '4' #Enter date and time and step in min for a daily calculation.
        print('{}, {}, {}'.format(MONTH, LATIT, DSTEP), file=f)

    ## Input Finalization
    print('', file=f)
    f.close()

    ## Run SMARTS 2.9.5
    #dump = os.system('smarts295bat.exe')
    commands = [smartsExecutable]
    command = None
    for cmd in commands:
        if os.path.exists(cmd):
            command = cmd
            break

    if not command:
        print('Could not find SMARTS2 executable.')
        data = None
    else:
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=open("output.txt", "w"), shell=True)
        p.wait()

        ## Read SMARTS 2.9.5 Output File
        data = pd.read_csv(smartsVersion+'.ext.txt', delim_whitespace=True)

    try:
        os.remove(smartsVersion+'.inp.txt')
    except:
        pass #     print("")
    try:
        os.remove(smartsVersion+'.out.txt')
    except:
        pass #     print("")
    try:
        os.remove(smartsVersion+'.ext.txt')
    except:
        pass #     print("")
    try:
        os.remove(smartsVersion+'.scn.txt')
    except:
        pass #     print("")

    # Return to original working directory.
    if original_wd:
        os.chdir(original_wd)

    return data

def getGEEdate(datestamp1,year,doy,longit,latit):
    '''
    Retrieves water vapor and ozone concentrations from GEE using the image metadata 
        Parameters:
            :class:`string`:
                - datestamp1: date in YYYY/MM/DD or YYYY-MM-DD format
            :class:`int`:
                - year: YYYY
                - doy: Day Of Year
            :class:`float`:
                - longit: decimal longitude
                - latit: decimal latitude
        Returns:
            :class:`float`:
                - average water vapor in g.cm-2 
                - ozone concentration in atm-cm over the site
            :class:`bool`:
                - a flag indicating whether or not ozone could be retrieved
    '''


    numPixels=1

    WTR = ee.ImageCollection('NCEP_RE/surface_wv') #kg/m2
    WTR=WTR.select('pr_wtr').filterDate(datestamp1.replace('/','-')+'T17:30',datestamp1.replace('/','-')+'T18:30')
    coord=ee.Geometry.Point(longit,latit)
    wv=WTR.first().sample(region=coord,numPixels=numPixels).getInfo()['features'][0]['properties']['pr_wtr']#.get('air').getInfo()
    wv=wv*1E-1 #g.cm-2

    year=int(year)
    doy=int(doy)
    i=1
    while i<3:
        try:
            O3=ee.ImageCollection('TOMS/MERGED') #dobson
            O3=O3.select('ozone').filter(ee.Filter.calendarRange(year,year,'year'))
            O3=O3.filter(ee.Filter.dayOfYear(doy-i,doy+i))
            coord=ee.Geometry.Point(longit,latit)
            o3=O3.first().sample(region=coord,numPixels=numPixels).getInfo()['features'][0]['properties']['ozone']#.get('air').getInfo()
            flag_no_o3=False
            break
        except:
            flag_no_o3=True
            i+=1
    o3=o3*1E-3 # atm-cm

    return wv, o3,flag_no_o3

def getImageAndParameters(path):
    '''
    Extracts the image array and its associated acquisition parameters from the corrected radiance file
    returns the array as well as all the metadata required for the atmospheric correction phase
        Parameters:
            :class:`string`:
                - path: path to the image saved by surehyp.preprocess.savePreprocessedL1R
        Returns:
            :class:`array`:
                - L: radiance array -- (m,n,b) array
                - bands: wavelengths of each band -- (b,) array
            :class:`dict`:
                - processing_metadata: metadata corresponding to the acquisition parameters, used for the atmospheric correciton
                - metadata: metadata of the ENVI file
    '''

    img=envi.open(path+'.hdr',path+'.img') #img in uW.cm-2.nm-1.sr-1, with a scaleFactor

    bands=np.asarray(img.metadata['wavelength']).astype(float)
    fwhms=np.asarray(img.metadata['fwhm']).astype(float)

    processing_metadata={}
    processing_metadata['longit']=float(img.metadata['center lon'])
    processing_metadata['latit']=float(img.metadata['center lat'])
    processing_metadata['datestamp1']=img.metadata['acquisition date']
    processing_metadata['datestamp2']=img.metadata['acquisition time start']
    processing_metadata['zenith']=float(img.metadata['sun zenith'])
    processing_metadata['azimuth']=float(img.metadata['sun azimuth'])

    if float(img.metadata['satellite zenith'])<0:
        processing_metadata['satelliteAzimuth']=float(img.metadata['satellite azimuth'])
    else:
        processing_metadata['satelliteAzimuth']=float(img.metadata['satellite azimuth'])+180
    processing_metadata['satelliteZenith']=np.abs(float(img.metadata['satellite zenith']))
    processing_metadata['scaleFactor']=float(img.metadata['scale factor'])

    processing_metadata['UL_lat']=float(img.metadata['ul_lat'])
    processing_metadata['UL_lon']=float(img.metadata['ul_lon'])
    processing_metadata['UR_lat']=float(img.metadata['ur_lat'])
    processing_metadata['UR_lon']=float(img.metadata['ur_lon'])
    processing_metadata['LL_lat']=float(img.metadata['ll_lat'])
    processing_metadata['LL_lon']=float(img.metadata['ll_lon'])
    processing_metadata['LR_lat']=float(img.metadata['lr_lat'])
    processing_metadata['LR_lon']=float(img.metadata['lr_lon'])

    processing_metadata['year']=processing_metadata['datestamp1'][:4]
    processing_metadata['month']=processing_metadata['datestamp1'][5:7]
    processing_metadata['day']=processing_metadata['datestamp1'][8:]
    processing_metadata['hour']=processing_metadata['datestamp2'][9:11]
    processing_metadata['minute']=processing_metadata['datestamp2'][12:14]
    processing_metadata['doy']=processing_metadata['datestamp2'][5:8]

    processing_metadata['thetaZ']=processing_metadata['zenith']*np.pi/180
    processing_metadata['thetaV']=processing_metadata['satelliteZenith']*np.pi/180


    metadata=img.metadata.copy()
    metadata['satellite azimuth']=processing_metadata['satelliteAzimuth']
    metadata['satellite zenith']=processing_metadata['satelliteZenith']


    L=img[:,:,:].astype(np.float32)/processing_metadata['scaleFactor']
    L=L.astype(np.float32) #image in uW.cm-2.nm-1.sr-1 with scalefactor

    return L,bands,fwhms,processing_metadata,metadata

def getGEEdem(UL_lat,UL_lon,UR_lat,UR_lon,LL_lat,LL_lon,LR_lat,LR_lon,demID='JAXA/ALOS/AW3D30/V3_2',elevationName='DSM'):
    '''
    returns the average site elevation in km using data from GEE
        Parameters:
            :class:`float`:
                - UL, UR, LL, LR: Upper left, Upper right, Lower left, Lower right; lon, lat: longitude, latitude; units in decimal degrees
            :class:`string`:
                - demID: name of the GEE dataset
                - elevationName: name of the band corresponding to elevation
        Returns:
            :class:`float`:
                - average site altitude in kilometers (assumes GEE provides altitude in meters)
    '''

    numPixels=1

    DEM = ee.ImageCollection(demID) #kg/m2
    DEM=DEM.select(elevationName)
    coord=ee.Geometry.Point(np.mean([UL_lon,UR_lon,LL_lon,LR_lon]),np.mean([UL_lat,UR_lat,LL_lat,LR_lat]))
    try:
        altit=DEM.mosaic().sample(region=coord,numPixels=numPixels,scale=1000).getInfo()['features'][0]['properties'][elevationName]
    except:
        raise
    return altit*1e-3

def getWaterVapor(bands,L,altit,latit,zenith,azimuth,doy,satelliteZenith,imass=0,io3=0,ialt=0,o3=3):
    '''
    returns the site average water vapor content using the water absorption bands at 940 and 1120 and comparing the absorption depth over land with the absorption depths over a LUT generated with SMARTS for the same optical path
        Parameters:
            :class:`array`:
                - bands: wavelengths of each band -- (b,) array
                - L: radiance array -- (m,n,b) array
            :class:`float`:
                - altit: site altitude in km
                - latit, longit: site latitude and longitude in decimal degrees
            :class:`array`:
                - year: YYYY
                - month: MM
                - day: DD
                - doy: Day Of Year
            :class:`float`:
                - satelliteZenith: satellite zenith angle in degrees
            io3,ialt,o3,imass: see SMARTS documentation
        Returns:
            :class:`float`:
                - wvImg: site average water vapor content
    '''

    #remove water pixels to keep only land surfaces
    L=L[L[:,:,0]!=0]
    L=L[L[:,np.argmin(np.abs(bands-940))]>0.25]
    L=np.mean(L,axis=0)

    #bands at 820,940 and 1120
    l940=[860,880]
    r940=[990,1120]
    l1120=[1060,1090]
    r1120=[1170,1200]

    ratios940=[]
    ratios1120=[]
    wvOut=[]

    for wv in np.concatenate((np.arange(0,1.5,0.2),np.arange(1.5,4,0.5),np.arange(4,12,1))):
        df=runSMARTS(ALTIT=altit,LATIT=latit,IMASS=imass,ZENITH=zenith,AZIM=azimuth,SUNCOR=get_SUNCOR(doy),IH2O=0,WV=wv,IO3=io3,IALT=ialt,AbO3=o3)
        df_gs=runSMARTS(ALTIT=altit,LATIT=0,IMASS=0,SUNCOR=get_SUNCOR(doy),ITURB=5,ZENITH=satelliteZenith,AZIM=0,IH2O=0,WV=wv,IO3=io3,IALT=ialt,AbO3=o3)

        W=df['Wvlgth']
        E=df['Extraterrestrial_spectrm']
        try:
            Dft=df['Difuse_tilted_irradiance']
        except:
            Dft=df['Total_Diffuse_tilt_irrad']
        T=df['Direct_rad_transmittance']
        T_gs=df_gs['Direct_rad_transmittance']

        Lout=T_gs*(T*E+Dft)
        Lout[W<=1700]=gaussian_filter(Lout[W<=1700],surehyp.various.fwhm2sigma(10))
        Lout[W>1700]=gaussian_filter(Lout[W>1700],surehyp.various.fwhm2sigma(2))
        f=interpolate.interp1d(W,Lout)
        Lout=f(bands)

        lShoulder940=np.mean(Lout[np.logical_and(bands>=l940[0],bands<=l940[1])])
        rShoulder940=np.mean(Lout[np.logical_and(bands>=r940[0],bands<=r940[1])])
        btm940=Lout[np.argmin(np.abs(bands-940))]
        ratios940.append(btm940/np.mean([lShoulder940,rShoulder940]))

        lShoulder1120=np.mean(Lout[np.logical_and(bands>=l1120[0],bands<=l1120[1])])
        rShoulder1120=np.mean(Lout[np.logical_and(bands>=r1120[0],bands<=r1120[1])])
        btm1120=Lout[np.argmin(np.abs(bands-1120))]
        ratios1120.append(btm1120/np.mean([lShoulder1120,rShoulder1120]))

        wvOut.append(wv)
    ratios940=np.asarray(ratios940)
    ratios1120=np.asarray(ratios1120)
    wvOut=np.asarray(wvOut)

    #for image
    lShoulder940=np.mean(L[np.logical_and(bands>=l940[0],bands<=l940[1])])
    rShoulder940=np.mean(L[np.logical_and(bands>=r940[0],bands<=r940[1])])
    btm940=L[np.argmin(np.abs(bands-940))]
    ratio940=btm940/np.mean([lShoulder940,rShoulder940])

    lShoulder1120=np.mean(L[np.logical_and(bands>=l1120[0],bands<=l1120[1])])
    rShoulder1120=np.mean(L[np.logical_and(bands>=r1120[0],bands<=r1120[1])])
    btm1120=L[np.argmin(np.abs(bands-1120))]
    ratio1120=btm1120/np.mean([lShoulder1120,rShoulder1120])

    if btm1120>=0.15:
        f=interpolate.interp1d(ratios1120,wvOut)
        wvImg=f(ratio1120)
    else:
        f=interpolate.interp1d(ratios940,wvOut)
        wvImg=f(ratio940)

    return wvImg

def darkObjectDehazing(L,bands, cloudMask=None):
    '''
    Estimates the average haze radiance in the image and removes it from the image
        Parameters:
            :class:`array`:
                - bands: wavelengths of each band -- (b,) array
                - L: radiance array -- (m,n,b) array
                - cloudMask: cloud mask, 0 if not clouds, 1 if clouds -- (m,n) array
        Returns:
            :class:`array`:
                - L: the dehazed radiance array usign the Dark object substraction method by Chavez (1988)  -- (m,n,b) array
                - Lhaze: the haze radiance spectrum -- (b,) array
    '''

    Ltmp=L.copy()
    if cloudMask is not None:
        Ltmp[cloudMask==1,:]=0

    Ltmp=Ltmp[Ltmp[:,:,0]>0,:]

    DOBJ=np.amin(Ltmp,axis=0)
    DOBJ=smoothing(DOBJ,5,1)
    bands_dobj=bands
    c=-0.5
    bref=bands[np.logical_and(bands>410,bands<480)][np.argmax(DOBJ[np.logical_and(bands>410,bands<480)])]
    deepB=np.amax(DOBJ[np.logical_and(bands>410,bands<480)])
    Lhaze=(bands*1E-3)**c
    offset=deepB-Lhaze[np.argmin(np.abs(bands-bref))]
    Lhaze+=offset

    while (np.amin(DOBJ[np.logical_and(bands_dobj>bref,bands<1500)]-Lhaze[np.logical_and(bands_dobj>bref,bands<1500)])<0) or (Lhaze[np.argmin(np.abs(bands-2150))]>0.05):
        c-=0.001
        Lhaze=(bands*1E-3)**c
        offset=deepB-Lhaze[np.argmin(np.abs(bands-bref))]
        Lhaze+=offset
    Lhaze[Lhaze<0]=0
    L=L-Lhaze
    L[L<=0]=0

    return L,Lhaze

def get_SUNCOR(doy):
    '''
    Computes the sun-earth distance correction factor
        Parameters:
            :class:`int`:
                - doy: Day of Year
        Returns:
            :class:`float`:
                - the sun-earth distance correction factor
    '''

    #compute the sun-earth distance correction factor depending on the day of year
    return 1-0.0335*np.sin(360*(int(doy)-94)/365*np.pi/180)

def smoothing(R,width=3,order=1):#,processO2=False,bands=None):
    '''
    Filters the array over its last axis with a Savitzky Golay filter
        Parameters:
            :class:`array`:
                - R: array -- (...,b) array
            :class:`int`:
                - width: width of the savitzky golay filter
                - order: order of the polynom for the filter
        Returns:
            :class:`array`:
                - smoothed array
    '''
    #if processO2==True: #remove band values from ]750,780[ and replaces them by values interpolated used the two previous and next R values
    #    argb=np.zeros(bands.shape).astype(int)
    #    argb[np.logical_and(bands>=730,bands<=750)]=1
    #    argb[np.logical_and(bands>=780,bands<=800)]=1
    #    Rtmp=R[R[:,:,0]>0,:]
    #    f=interp1d(bands[argb==1],Rtmp[...,argb==1], kind='cubic')
    #    Rtmp[:,:,argb]=f(Rtmp[...,argb==1])
    #    R[R[:,:,0]>0,:]=Rtmp

    return savgol_filter(R,width,order)

def computeLtoRfactor(df,df_gs):
    '''
    Calculates the factors to convert TOA radiance to BOA reflectance considering a flat terrain correction
        Parameters:
            :class:`pd.DataFrame`:
                - df: dataframe containing the outputs of the SMARTS simulation for the sun-ground optical path
                - df_gs: dataframe containing the outputs of the SMARTS simulation for the ground-sensor optical path
        Returns:
            :class:`array`:
                - the correction factors for each wavelength
    '''

    #compute the factor to convert TOA radiance to surface reflectance
    W=df['Wvlgth'].values
    E=df['Extraterrestrial_spectrm']
    T=df['RayleighScat_trnsmittnce']*df['Ozone_totl_transmittance']*df['Trace_gas__transmittance']*df['WaterVapor_transmittance']*df['Mixed_gas__transmittance']*df['Aerosol_tot_transmittnce']
    T_gs=df_gs['RayleighScat_trnsmittnce']*df_gs['Ozone_totl_transmittance']*df_gs['Trace_gas__transmittance']*df_gs['WaterVapor_transmittance']*df_gs['Mixed_gas__transmittance']*df_gs['Aerosol_tot_transmittnce']
    Dnt=df['Direct_normal_irradiance']
    Dtt=df['Direct_tilted_irradiance']
    try:
        Dft=df['Difuse_tilted_irradiance']
    except:
        Dft=df['Total_Diffuse_tilt_irrad']
    Dgt=df['Global_tilted_irradiance']

    factor=np.pi/T_gs/Dgt
    factor[W<=1700]=gaussian_filter(factor[W<=1700],surehyp.various.fwhm2sigma(10))
    factor[W>1700]=gaussian_filter(factor[W>1700],surehyp.various.fwhm2sigma(2))

    return factor

def getAtmosphericParameters(bands,L,datestamp1,year,doy,longit,latit,altit,satelliteZenith,zenith,azimuth,imass=0,io3=0,ialt=0,o3=3):
    '''
    Returns atmopsheric ozone and water vapor content for the study site from GEE or (for water, if possible) directly from the image
        Parameters:
            :class:`array`:
                - bands: wavelengths of each band -- (b,) array
                - L: radiance array -- (m,n,b) array
            :class:`string`:
                - datestamp1: date in YYYY/MM/DD or YYYY-MM-DD format
            :class:`int`:
                - year: YYYY
                - month: MM
                - minute: mm
                - day: DD
                - doy: Day Of Year
            :class:`float`:
                - latit, longit: site latitude and longitude in decimal degrees
                - altit: site altitude in km
                - satelliteZenith: satellite zenith angle in degrees
            io3,ialt,o3,imass: see SMARTS documentation
        Returns:
            :class:`float`:
                - average water vapor in g.cm-2 
                - ozone concentration in atm-cm over the site
            :class:`bool`:
                - a flag indicating whether or not ozone could be retrieved
    '''

    #obtain some atmospheric parameters using GEE
    #print('get water vapor and ozone from GEE')
    wvGEE,o3,flag_no_o3=getGEEdate(datestamp1,year,doy,longit,latit)

    #print('get water vapor from the radiance image')
    wv=getWaterVapor(bands,L,altit,latit,zenith,azimuth,doy,satelliteZenith,imass,io3,ialt,o3)

    if (wv>12) or (np.isnan(wv)):
        wv=wvGEE
    return wv,o3,flag_no_o3

def computeLtoR(L,bands,df,df_gs):
    '''
    Calculates the BOA reflectance from the TOA radiance considering a flat terrain correction
        Parameters:
            :class:`array`:
                - bands: wavelengths of each band -- (b,) array
                - L: at satellite radiance array -- (m,n,b) array
            :class:`pd.DataFrame`:
                - df: dataframe containing the outputs of the SMARTS simulation for the sun-ground optical path
                - df_gs: dataframe containing the outputs of the SMARTS simulation for the ground-sensor optical path
        Returns:
            :class:`array`:
                - R: the BOA reflectance  -- (m,n,b) array
    '''

    #get the factor to convert TOA radiance to surface reflectance and return the reflectance
    factor=computeLtoRfactor(df,df_gs)
    W=df['Wvlgth'].values
    fun=interpolate.interp1d(W,factor)
    factor=fun(bands)
    R=factor*L
    return R

def saveRimage(R,metadata,pathOut,scaleFactor=100):
    '''
    Saves the reflectance image to the given path
        Parameters:
            :class:`array`:
                - R: array to save -- (m,n,b) array
            :class:`dict`:
                - metadata: image metadata (ENVI format for the Spectral library)
            :class:`string`:
                - pathOut: pathout (must end with .hdr)
            :class:`int`:
                - scaleFactor: scaling factor to multiply the reflectance with. Allows for saving the array in unsigned int16 format to save space
    '''

    scale=scaleFactor*np.ones(R.shape[2]).astype(int)
    metadata['scale factor']=scale.tolist()

    # Print warning about scale factor for SAM users
    print('\n' + '='*70)
    print('IMPORTANT: Reflectance Scale Factor Information')
    print('='*70)
    print('Reflectance data is being saved with scale factor = {}'.format(scaleFactor))
    print('Data range BEFORE scaling: {:.4f} to {:.4f}'.format(np.min(R[R>0]), np.max(R)))

    R=R*scaleFactor

    print('Data range AFTER scaling: {:.1f} to {:.1f}'.format(np.min(R[R>0]), np.max(R)))
    print('')
    print('NOTE: For Spectral Angle Mapper (SAM), you MUST divide by {} to get 0-1 range:'.format(scaleFactor))
    print('      cube = cube.astype(np.float32) / {}'.format(scaleFactor))
    print('      endmembers = endmembers / {}'.format(scaleFactor))
    print('='*70 + '\n')

    R[R>65535]=65535
    R[R<0]=0
    R=R.astype(np.uint16)
    envi.save_image(pathOut+'.hdr',R,metadata=metadata,force=True)

def getTOAreflectanceFactor(bands,latit,doy,satelliteZenith,zenith,azimuth):
    '''
    Calculates the factors to get TOA reflectance from at satellite radiance
        Parameters:
            :class:`array`:
                - bands: wavelengths of each band -- (b,) array
            :class:`float`:
                - latit, longit: site latitude and longitude in decimal degrees
            :class:`int`:
                - year: YYYY
                - month: MM
                - day: DD
                - doy: Day Of Year
            :class:`float`:
                - satelliteZenith: satellite zenith angle in degrees
        Returns:
            :class:`array`:
                - the conversion factors for each wavelength
    '''

    #compute TOA reflectance
    df=runSMARTS(ALTIT=0,LATIT=latit,IMASS=0,ZENITH=zenith,AZIM=azimuth,SUNCOR=get_SUNCOR(doy),IH2O=0,WV=0,IO3=0,IALT=0,AbO3=0)
    W=df['Wvlgth'].values
    E=df['Extraterrestrial_spectrm'].values
    factor=np.pi/(E*np.cos(satelliteZenith*np.pi/180))
    factor[W<=1700]=gaussian_filter(factor[W<=1700],surehyp.various.fwhm2sigma(10))
    factor[W>1700]=gaussian_filter(factor[W>1700],surehyp.various.fwhm2sigma(2))
    f=interpolate.interp1d(W,factor)
    factor=f(bands)
    return factor

def cloudAndShadowsDetection(bands,A,latit,doy,satelliteZenith,zenith,azimuth,slope,aspect,pixelSize=30):
    '''
    Retrieves clear view, cloud, and cloud shadows masks from the image. Cloud and shadow detection adapted from Braaten et al 2015
        Parameters:
            :class:`array`:
                - bands: wavelengths of each band -- (b,) array
                - A: radiance array -- (m,n,b) array
            :class:`float`:
                - latit, longit: site latitude and longitude in decimal degrees
            :class:`int`:
                - year: YYYY
                - month: MM
                - day: DD
                - doy: Day Of Year
            :class:`float`:
                - satelliteZenith: satellite zenith angle in degrees
        Returns:
            :class:`array`:
                - masks for clearview, shadows and clouds -- (m,n) arrays
    '''

    factor=getTOAreflectanceFactor(bands,latit,doy,satelliteZenith,zenith,azimuth).astype(np.float32)
    A=factor*A
    A[A<=0]=0
    A=A.astype(np.float32)

    B1,_=surehyp.various.getBLandsat(A,bands,'1')
    B2,_=surehyp.various.getBLandsat(A,bands,'2')
    NDGR=(B1-B2)/(B1+B2)

    CL=np.zeros((A.shape[0],A.shape[1])).astype(int)
    CL[((B1>17.5) & (NDGR>-0.1)) | (B1>39)] = 1

    CL=skimage.measure.label(CL,background=0)
    for label in np.unique(CL):
        if (label>0):
            if np.size(CL[CL==label])<9:
                CL[CL==label]=0
    CL[CL>0]=1
    CL=skimage.morphology.binary_dilation(CL,footprint=np.ones((5,5)))

    NDWI,_,_=surehyp.various.NDWI_water(A,bands)
    WL=np.zeros((A.shape[0],A.shape[1])).astype(int)
    WL[NDWI>0.3]=1

    WL=skimage.measure.label(WL,background=0)
    for label in np.unique(WL):
        if (label>0):
            if np.size(WL[WL==label])<6:
                WL[WL==label]=0
    WL[WL>0]=1
    WL=skimage.morphology.binary_dilation(WL,footprint=np.ones((5,5)))

    B4,_=surehyp.various.getBLandsat(A,bands,'4')
    if slope is not None:
        B4c=MM_topo_correction(B4,np.array([950]),slope,aspect,zenith,azimuth)
    else:
        B4c=B4

    X1=np.nanmean(B4c[CL!=1])
    Y=0.4*X1+0.0248*1e2  # *1.2 as Braaten et al 2015 assumes reflectance to be between 0-1 while with its default parameters SUREHYP has 0-100
    X2=np.nanmean(B4c[(B4c>Y)&(CL!=1)])
    ST=0.47*X2+0.0073*1e2 # same here
    CCSL=np.zeros((A.shape[0],A.shape[1])).astype(int)
    CCSL[(B4c<ST)&(WL!=1)]=1

    tmp=skimage.morphology.binary_dilation(WL,footprint=np.ones((15,15)))
    CCPL=np.zeros((A.shape[0],A.shape[1])).astype(int)
    d1=int(1e3/np.tan((90-zenith)*np.pi/180)/pixelSize)
    d2=int(7e3/np.tan((90-zenith)*np.pi/180)/pixelSize)
    ds=np.arange(d1,d2,1).astype(int)
    for d in ds:
        tform=skimage.transform.SimilarityTransform(translation=[np.sin(azimuth*np.pi/180)*d,-np.cos(azimuth*np.pi/180)*d])
        CCPL+=skimage.transform.warp(CL,tform)
    CCPL[CCPL>1]=1

    SHADOWS=CCPL+CCSL
    SHADOWS[SHADOWS<2]=0
    SHADOWS=skimage.measure.label(SHADOWS,background=0)
    for label in np.unique(SHADOWS):
        if (label>0):
            if np.size(SHADOWS[SHADOWS==label])<9:
                SHADOWS[SHADOWS==label]=0
    SHADOWS[SHADOWS>0]=1
    SHADOWS=skimage.morphology.binary_dilation(SHADOWS,footprint=np.ones((5,5)))

    CLEARVIEW=np.zeros((A.shape[0],A.shape[1])).astype(int)
    CLEARVIEW[A[:,:,0]>0]=1
    CLEARVIEW[CL==1]=0
    CLEARVIEW[SHADOWS==1]=0

    return CLEARVIEW,CL,SHADOWS

def cirrusRemoval(bands,A,latit,doy,satelliteZenith,zenith,azimuth,cirrusReflectanceThreshold=1,cloudReflectance=30):
    '''
    Attemps a thin cirrus correction using the method presented by Gao et al 1997, 2017 
        Parameters:
            :class:`array`:
                - bands: wavelengths of each band -- (b,) array
                - A: radiance array -- (m,n,b) array
            :class:`float`:
                - latit, longit: site latitude and longitude in decimal degrees
            :class:`int`:
                - year: YYYY
                - month: MM
                - day: DD
                - doy: Day Of Year
            :class:`float`:
                - satelliteZenith: satellite zenith angle in degrees
                - cirrusReflectanceThreshold: TOA reflectance below which pixels at 1380 um  are considered to be 0 even though they are not
                - cloudReflectance: reflectance threshold to determine if pixel is a cloud
        Returns:
            :class:`array`:
                - cirrus-corrected radiance array -- (m,n,b) array
    '''

    #uses the method presented by Gao et al 1997, 2017 to remove cirrus effects
    #compute TOA reflectance, performs the cirrus removal, and retransforms to TOA radiance
    #returns the cirrus-removed TOA radiance
    factor=getTOAreflectanceFactor(bands,latit,doy,satelliteZenith,zenith,azimuth).astype(np.float32)
    A=factor*A
    A[A<=0]=0
    A=A.astype(np.float32)

    Rcirrus=A[:,:,np.argmin(np.abs(bands-1380))]
    r1380=Rcirrus[Rcirrus>=cirrusReflectanceThreshold] #if Reflectance TOA [0-100]
    Rcirrus[Rcirrus<cirrusReflectanceThreshold]=0

    if r1380.size>50: #if not enough points, considers that the computation can not take place
        rlambda=A[Rcirrus>cirrusReflectanceThreshold,:]

        xs=[]
        ys=[]
        for i in np.arange(np.floor(np.amin(r1380)),np.ceil(np.amax(r1380)),0.25):
            cond=np.logical_and(r1380>=i,r1380<i+1)
            tmp1380=r1380[cond]
            tmpLambda=rlambda[cond,:]
            if not len(tmpLambda)==0:
                xs.append(np.amin(tmpLambda,axis=0))
                ys.append(tmp1380[np.argmin(tmpLambda,axis=0)])
        xs=np.asarray(xs)
        ys=np.asarray(ys)

        Ka=[]
        for b in np.arange(xs.shape[1]):
            xs_trim=xs[:,b].copy()
            ys_trim=ys[:,b].copy()
            if xs_trim.size>6:
                diff=np.diff(xs_trim)
                ys_trim=ys_trim[1:]
                xs_trim=xs_trim[1:]
                ys_trim=ys_trim[diff>0.5]
                xs_trim=xs_trim[diff>0.5]

                if xs_trim.size>5:
                    p=np.polyfit(xs_trim,ys_trim,1)
                    Ka.append(p[0])
                else:
                    Ka.append(10000)
            else:
                Ka.append(10000)
        Ka=np.asarray(Ka).astype(np.float32)
        Rcirrus=np.moveaxis(np.tile(Rcirrus,(xs.shape[1],1,1)),0,2)
        A=A-Rcirrus/Ka
        A=A/factor
        A[A<=0]=0
    else:
        A=A/factor
        A[A<=0]=0
    return A

def splitDEMdownload(UL_lon,UL_lat,UR_lon,UR_lat,LR_lon,LR_lat,LL_lon,LL_lat,elev,prefix='elev'):
    '''
    if the GEE image to download is too large, divide it in four and download each subimage
    recursive function if the subimage is still too large
        Parameters:
            :class:`float`:
                - UL, UR, LL, LR: Upper left, Upper right, Lower left, Lower right for ; lon, lat: longitude, latitude; units in decimal degrees
            :class:`string`:
                - elev: GEE image to download
                - prefix: folder names (they will appear as 'prefix_*')
        Returns:
            :class:`list`:
                - a list of the folders containing the subimages
    '''
    #print('splitting download of '+prefix)


    lon_min=np.amin([UL_lon-0.05,UR_lon-0.05,LR_lon-0.05,LL_lon-0.05])
    lon_max=np.amax([UL_lon+0.05,UR_lon+0.05,LR_lon+0.05,LL_lon+0.05])
    lat_min=np.amin([UL_lat-0.05,UR_lat-0.05,LR_lat-0.05,LL_lat-0.05])
    lat_max=np.amax([UL_lat+0.05,UR_lat+0.05,LR_lat+0.05,LL_lat+0.05])

    lon_mid=(lon_min+lon_max)/2
    lat_mid=(lat_min+lat_max)/2

    folders=[]
    try:
        region=ee.Geometry.Polygon([[[lon_min,lat_max],[lon_mid,lat_max],[lon_mid, lat_mid],[lon_min,lat_mid]]],None,False)
        elev0=elev.clip(region)
        geetools.batch.image.toLocal(elev0,prefix+'_0',scale=20,region=region)
        folders.append(prefix+'_0')
    except:
        folders.extend(splitDEMdownload(lon_min,lat_max,lon_mid,lat_max,lon_mid,lat_mid,lon_min,lat_mid,elev,prefix=prefix+'_0'))

    try:
        region=ee.Geometry.Polygon([[[lon_mid,lat_max],[lon_max,lat_max],[lon_max, lat_mid],[lon_mid,lat_mid]]],None,False)
        elev1=elev.clip(region)
        geetools.batch.image.toLocal(elev1,prefix+'_1',scale=20,region=region)
        folders.append(prefix+'_1')
    except:
        folders.extend(splitDEMdownload(lon_mid,lat_max,lon_max,lat_max,lon_max,lat_mid,lon_mid,lat_mid,elev,prefix=prefix+'_1'))

    try:
        region=ee.Geometry.Polygon([[[lon_mid,lat_mid],[lon_max,lat_mid],[lon_max, lat_min],[lon_mid,lat_min]]],None,False)
        elev2=elev.clip(region)
        geetools.batch.image.toLocal(elev2,prefix+'_2',scale=20,region=region)
        folders.append(prefix+'_2')
    except:
        folders.extend(splitDEMdownload(lon_mid,lat_mid,lon_max,lat_mid,lon_max,lat_min,lon_mid,lat_min,elev,prefix=prefix+'_2'))

    try:
        region=ee.Geometry.Polygon([[[lon_min,lat_mid],[lon_mid,lat_mid],[lon_mid, lat_min],[lon_min,lat_min]]],None,False)
        elev3=elev.clip(region)
        geetools.batch.image.toLocal(elev3,prefix+'_3',scale=20,region=region)
        folders.append(prefix+'_3')
    except:
        folders.extend(splitDEMdownload(lon_min,lat_mid,lon_mid,lat_mid,lon_mid,lat_min,lon_min,lat_min,elev,prefix=prefix+'_3'))
    return folders

def getDEMimages(UL_lon,UL_lat,UR_lon,UR_lat,LR_lon,LR_lat,LL_lon,LL_lat,demID='JAXA/ALOS/AW3D30/V3_2',elevationName='DSM'):
    '''
    downloads the DEM image for the region delimited by the latitudes and longitudes from GEE
        Parameters:
            :class:`float`:
                - UL, UR, LL, LR: Upper left, Upper right, Lower left, Lower right for; lon, lat: longitude, latitude; units in decimal degrees
            :class:`string`:
                - demID: name of the GEE dataset
                - elevationName: name of the band correspond to elevation
        Returns:
            :class:`string`:
                - path to the downloaded DEM image
    '''
    try:
        elev = ee.Image(demID);
        elev= elev.select(elevationName)
        elev.getInfo() #will fail if not an image
    except:
        dem = ee.ImageCollection(demID);
        dem = dem.select(elevationName)
        elev=dem.mosaic()

    try: # download can fail if image is too large, so it may be necessary to download it by parts and reassemble everything
        region=ee.Geometry.Polygon([[[UL_lon-0.05,UL_lat+0.05],[UR_lon+0.05,UR_lat+0.05],[LR_lon+0.05,LR_lat-0.05],[LL_lon-0.05,LL_lat-0.05]]],None,False)
        elev=elev.clip(region)
        geetools.batch.image.toLocal(elev,'elev',region=region,scale=20)
    except:
        try:
            elev = ee.Image(demID);
            elev= elev.select(elevationName)
            elev.getInfo() #will fail if not an image
        except:
            dem = ee.ImageCollection(demID);
            dem = dem.select(elevationName)
            elev=dem.mosaic()

        folders=splitDEMdownload(UL_lon,UL_lat,UR_lon,UR_lat,LR_lon,LR_lat,LL_lon,LL_lat,elev)

        src_files_to_mosaic=[]
        for elev in folders:
            for f in os.listdir('./'+elev+'/'):
                if 'tif' in f:
                    src=rasterio.open('./'+elev+'/'+f)
                    src_files_to_mosaic.append(src)
        mosaic, out_trans=merge(src_files_to_mosaic)
        out_meta=src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans,
                        }
                    )

        Path('./elev/').mkdir(parents=True, exist_ok=True)

        with rasterio.open('./elev/'+f, "w", **out_meta) as dest:
            dest.write(mosaic)

    for f in os.listdir('.'):
        if '.zip' in f:
            os.remove(os.path.join('.',f))
    for f in os.listdir('./elev/'):
        if ('.tif' in f) and ('tmp' not in f):
            fileName=f
    return './elev/'+fileName

def reprojectImage(im,dst_crs,pathOut):
    '''
    Reprojects an image to another CRS
        Parameters:
            :class:`rasterio image`:
                - im: rasterio image to reproject
            :class:`string`:
                - dst_crs: crs to reproject img to
                - pathOut: path to save the reprojected image
        Returns:
            :class:`string`:
                - path of the saved reprojected image
    '''

    with im as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'count': 1
        })

        with rasterio.open(pathOut, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                num_threads=10,
                warp_mem_limit=1024)
    return pathOut

def get_target_rows_cols(im1,imSecondary, maskBand=40):
    '''
    finds the position of non null matching-location pixels in im1 and imSecondary
        Parameters:
            :class:`rasterio image`:
                - im1: rasterio reference image
                - imSecondary: rasterio image
            :class:`int`:
                - maskBand: band used to keep pixels containing values: pixels with negative values for this band will not be considered
        Returns:
            :class:`array`:
                - rows, cols: indexes of pixels containing values in im1
                - rowsSecondary, colsSecondary: indexes of the pixels in imSecondary that are associated with rows, cols in im1
    '''

    T0=im1.transform
    array1=im1.read()
    array1=np.moveaxis(array1,0,2)

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(array1.shape[1]), np.arange(array1.shape[0]))

    cols=cols[array1[:,:,maskBand]>=0]
    rows=rows[array1[:,:,maskBand]>=0]
    xs,ys=rasterio.transform.xy(T0,rows,cols)

    #get coordinates row/cols in the landcover map
    rowsSecondary, colsSecondary = rasterio.transform.rowcol(imSecondary.transform, xs,ys)

    rowsSecondary=np.asarray(rowsSecondary)
    colsSecondary=np.asarray(colsSecondary)
    return rows,cols,rowsSecondary, colsSecondary

def extractSecondaryData(array1,array2,rows,cols,rowsSecondary,colsSecondary):
    '''
    Trims array2 so that its empty pixels match those of array1
        Parameters:
            :class:`array`:
                - array1: reference array -- (m,n,...) array
                - array2: (m,n) array
                - rows, cols: indexes of pixels containing values in array1
                - rowsSecondary, colsSecondary: indexes of the pixels in array2 that are associated with rows, cols in array1
        Returns:
            :class:`array`:
                - array with shape (m,n) containing the values of array2
    '''

    #remove OOB values
    rows=rows[rowsSecondary>0]
    cols=cols[rowsSecondary>0]
    colsSecondary=colsSecondary[rowsSecondary>0]
    rowsSecondary=rowsSecondary[rowsSecondary>0]
    rows=rows[colsSecondary>0]
    cols=cols[colsSecondary>0]
    rowsSecondary=rowsSecondary[colsSecondary>0]
    colsSecondary=colsSecondary[colsSecondary>0]
    rows=rows[rowsSecondary<array2.shape[0]]
    cols=cols[rowsSecondary<array2.shape[0]]
    colsSecondary=colsSecondary[rowsSecondary<array2.shape[0]]
    rowsSecondary=rowsSecondary[rowsSecondary<array2.shape[0]]
    rows=rows[colsSecondary<array2.shape[1]]
    rowsSecondary=rowsSecondary[colsSecondary<array2.shape[1]]
    cols=cols[colsSecondary<array2.shape[1]]
    colsSecondary=colsSecondary[colsSecondary<array2.shape[1]]

    arraySecondaryNew=np.zeros((array1.shape[0],array1.shape[1]))
    arraySecondaryNew[rows,cols]=array2[rowsSecondary,colsSecondary].copy()
    arraySecondaryNew=np.reshape(arraySecondaryNew,(array1.shape[0],array1.shape[1]))
    return arraySecondaryNew

def getDemReflectance(altitMap,tiltMap,wazimMap,stepAltit,stepTilt,stepWazim,latit,WV,AbO3,doy,zenith,azimuth,satelliteZenith,satelliteAzimuth,L,bands,IH2O=0,IO3=0,IALT=0,IALBDX=1,RHOX=0,rho_background=0):
    '''
    Calculates the BOA reflectance from the radiance image, using a rough terrain atmospheric correction
        Parameters:
            :class:`array`:
                - altitMap: elevation map of the study site (km) -- (m,n) array
                - tiltMap: slope tilt angle map (degree) -- (m,n) array
                - wazimMap: slope aspect map (degree) -- (m,n) array
            :class:`float`:
                - stepAltit, stepTilt, stepWazim: steps for the sampling scheme over the LUT over altitude, slope angle and asspect
                - longit, latit: longitude, latitude
            :class:`int`:
                - year: YYYY
                - month: MM
                - day: DD
                - doy: Day Of Year
            :class:`float`:
                - satelliteZenith: satellite zenith angle in degrees
                - satelliteAzimuth: satellite azimuth angle in degrees (may be set to 0 here as per SMARTS doc)
            :class:`array`:
                - bands: wavelengths of each band -- (b,) array
                - L: at satellite radiance array -- (m,n,b) array
            :class:`float`:
                - WV: site water vapor
                - AbO3: site ozone
            IH2O, IO3, IALT: see SMARTS documentation, leave untouched unless you want to edit the function
        Returns:
            :class:`array`:
                - The BOA reflectance array
    '''

    #prepare the iteration vectors for the LUT building
    ALTITS=np.arange(np.maximum(0,np.floor(np.nanmin(altitMap))),np.maximum(np.ceil(np.nanmax(altitMap))+stepAltit,np.floor(np.nanmin(altitMap))+2*stepAltit),stepAltit)
    TILTS=np.arange(np.maximum(0,np.floor(np.nanmin(tiltMap))),np.maximum(np.ceil(np.nanmax(tiltMap))+stepTilt,np.floor(np.nanmin(tiltMap))+2*stepTilt),stepTilt) #decimal degree
    WAZIMS=np.arange(np.maximum(0,np.floor(np.nanmin(wazimMap))),np.maximum(np.ceil(np.nanmax(wazimMap))+stepWazim,np.floor(np.nanmin(wazimMap))+stepWazim),stepWazim)
    TILTS=TILTS[TILTS<=90]
    WAZIMS=WAZIMS[WAZIMS<=360]
    ALTITS=ALTITS[ALTITS<=9]
    TILTS=TILTS[TILTS>=0]
    WAZIMS=WAZIMS[WAZIMS>=0]
    ALTITS=ALTITS[ALTITS>=0]

    #first dry run to get W
    df=runSMARTS(ALTIT=0,ITILT='1',TILT=0,WAZIM=0,LATIT=latit,IMASS=0,SUNCOR=get_SUNCOR(doy),ZENITH=zenith,AZIM=azimuth)
    W=df['Wvlgth'].values

    points = (ALTITS, TILTS, WAZIMS)
    xv,yv,zv=np.meshgrid(*points,indexing='ij')
    data=np.zeros((xv.shape[0],xv.shape[1],xv.shape[2],len(W)))

    #print(np.unique(ALTITS))
    #print(np.unique(TILTS))
    #print(np.unique(WAZIMS))

    for i in tqdm(np.arange(xv.shape[0]),desc='ALTITS'):
        df=runSMARTS(ALTIT=xv[i,0,0],ITILT='1',TILT=0,WAZIM=0,LATIT=latit,IMASS=0,ZENITH=np.abs(zenith),AZIM=azimuth,SUNCOR=get_SUNCOR(doy),IH2O=IH2O,WV=WV,IO3=IO3,IALT=IALT,AbO3=AbO3,IALBDX=1,RHOX=0.2)
        df_gs=runSMARTS(ALTIT=xv[i,0,0],ITILT='1',TILT=0,WAZIM=0,LATIT=0,IMASS=0,SUNCOR=get_SUNCOR(doy),ITURB=5,ZENITH=np.abs(satelliteZenith),AZIM=np.abs(satelliteAzimuth),IH2O=IH2O,WV=WV,IO3=IO3,IALT=IALT,AbO3=AbO3,IALBDX=1,RHOX=0.2)

        E=df['Extraterrestrial_spectrm'].astype(np.float)
        T_sg=df['Direct_rad_transmittance'].astype(np.float)
        T_gs=df_gs['Direct_rad_transmittance'].astype(np.float)
        try:
            Dft=df['Difuse_tilted_irradiance'].astype(np.float)
        except:
            Dft=df['Total_Diffuse_tilt_irrad'].astype(np.float)

        for j in tqdm(np.arange(xv.shape[1]),desc='TILTS '):
            for k in tqdm(np.arange(xv.shape[2]),desc='WAZIMS'):
                #compute solar illumination angles betai
                tilt=yv[i,j,k]*np.pi/180
                wazim=zv[i,j,k]*np.pi/180
                Vslope=np.asarray([np.sin(tilt)*np.cos(wazim),np.sin(tilt)*np.sin(wazim),np.cos(tilt)])
                Vsun=np.asarray([np.sin(zenith*np.pi/180)*np.cos(azimuth*np.pi/180),np.sin(zenith*np.pi/180)*np.sin(azimuth*np.pi/180),np.cos(zenith*np.pi/180)])
                betai=np.arccos(np.dot(Vslope,Vsun)/np.linalg.norm(Vslope)/np.linalg.norm(Vsun))
                betai=betai.astype(np.float32)
                if np.abs(betai)>np.pi/2:
                    betai=np.pi/2

                if betai<45*np.pi/180:
                    '''
                    for low angles where direct illumination is considerably more important than diffuse illumination
                    this is not the exact formula used by SMARTS, however difference is negligible for angles <45 degrees
                    '''
                    tmp= ( np.pi/T_gs/(T_sg*E*np.cos(betai) + Dft*(1+np.cos(tilt))/2 + (T_sg*E+Dft)*rho_background*(1-np.cos(tilt))/2) )
                else: #for high angles, e.g. indirect illumination, compute with SMARTS
                    df=runSMARTS(ALTIT=xv[i,0,0],ITILT='1',TILT=yv[i,j,k],WAZIM=zv[i,j,k],LATIT=latit,IMASS=0,ZENITH=np.abs(zenith),AZIM=azimuth,SUNCOR=get_SUNCOR(doy),IH2O=IH2O,WV=WV,IO3=IO3,IALT=IALT,AbO3=AbO3,IALBDX=1,RHOX=0.2)
                    df_gs=runSMARTS(ALTIT=xv[i,0,0],ITILT='1',TILT=yv[i,j,k],WAZIM=zv[i,j,k],LATIT=0,IMASS=0,SUNCOR=get_SUNCOR(doy),ITURB=5,ZENITH=np.abs(satelliteZenith),AZIM=np.abs(satelliteAzimuth),IH2O=IH2O,WV=WV,IO3=IO3,IALT=IALT,AbO3=AbO3,IALBDX=1,RHOX=0.2)

                    Dgt=df['Global_tilted_irradiance']
                    T_gs=df_gs['Direct_rad_transmittance']

                    tmp=np.pi/T_gs/(Dgt)

                tmp[W<=1700]=gaussian_filter1d(tmp[W<=1700],surehyp.various.fwhm2sigma(10))
                tmp[W>1700]=gaussian_filter1d(tmp[W>1700],surehyp.various.fwhm2sigma(2))
                data[i,j,k,:]=tmp
    W=df['Wvlgth'].values

    if flag_interp3d==True: #if the cython library interp3d has been installed, use it
        interpFunction = interp_3d.Interp3D(data,ALTITS,TILTS,WAZIMS)
    else: #otherwise use scipy
        interpFunction = interpolate.RegularGridInterpolator((ALTITS, TILTS, WAZIMS), data)

    data=None

    L[L<0]=0
    L[np.isnan(L)]=0
    idx=np.argwhere(L[:,:,40].flatten()>0).squeeze()

    R=np.zeros((L.shape[0]*L.shape[1],L.shape[2])).squeeze()
    Lflat=L.reshape((L.shape[0]*L.shape[1],L.shape[2]))
    for idxx in np.array_split(idx,100):
        values=np.stack((altitMap.flatten()[idxx],tiltMap.flatten()[idxx],wazimMap.flatten()[idxx]),axis=-1)
        M=interpFunction(values)
        f=interpolate.interp1d(W,M)
        R[idxx,:]=f(bands)*Lflat[idxx,:]
    R=R.reshape(L.shape)

    return R


def reprojectDEM(path_im1,path_elev='./elev/SRTMGL1_003.elevation.tif',path_elev_out='./elev/tmp.tif',extension='.img'):
    '''
    reprojects an image in the CRS of im1
        Parameters:
            path_im1: rasterio reference image
            path_elev: path of the image to reproject
            path_elev_out: path to save the reprojected image to
            extension: extension of im1
        Returns:
            path_elev_out: path of the reprojected image
    '''

    im1=rasterio.open(path_im1+extension)
    im2=rasterio.open(path_elev)
    try:
        os.remove(path_elev_out)
    except:
        pass
    path_elev_out = reprojectImage(im2,im1.profile['crs'],path_elev_out)
    im1.close()
    im2.close()
    return path_elev_out

def matchResolution(pathToIm1,path_elev='./elev/tmp.tif',path_out='./elev/tmp_blurred.tif',extension='.img') :
    '''
    smoothes the image stored in path_elev so that the values in each pixel correspond to the values that a sensor with a different spatial FWHM would have acquired
        Parameters:
            :class:`rasterio image`:
                - pathToIm1: path to the reference image for which the DEM data needs to be extracted
            :class:`string`:
                - path_elev: path to the DEM image
                - path_out: path to the degraded image
                - extension: extension of im1
        Returns:
            :class:`string`:
                - path_out: path to the degraded image
    '''

    im1=rasterio.open(pathToIm1+extension)
    im2=rasterio.open(path_elev)
    smoothing_std=im1.transform[0]/im2.transform[0]
    elev=im2.read()
    elev=gaussian_filter(elev,smoothing_std)

    with rasterio.open(path_out,'w',**im2.meta) as out:
        out.write(elev)
    return path_out

def extractDEMdata(pathToIm1,path_elev='./elev/tmp.tif',extension='.img',maskBand=40):
    '''
    Computes elevation, slope, and aspect data for the area corresponding to the hyperion strip
        Parameters:
            :class:`string`:
                - pathToIm1: path to the reference image for which the DEM data needs to be extracted
                - path_elev: path to the DEM image
                - extension: extension of im1
            :class:`int`:
                - maskBand: Hyperion band used to mask the DEM-related pixels that do not need to contain data
        Returns:
            :class:`array`:
                - elevation: (degree) -- (m,n) arrays
                - slope: (degree) -- (m,n) arrays 
                - aspect (km) -- (m,n) arrays
    '''

    im1=rasterio.open(pathToIm1+extension)
    ar1=im1.read()
    ar1=np.moveaxis(ar1,0,2)
    im2=rasterio.open(path_elev)
    meta=im2.meta
    rows1,cols1,rows2, cols2=get_target_rows_cols(im1,im2,maskBand=maskBand) #elev, slope and aspect all have the same projection

    elev=np.squeeze(im2.read())
    elev=extractSecondaryData(ar1,elev,rows1,cols1,rows2, cols2)

    elev[ar1[:,:,maskBand]<=0]=np.nan
    elev=elev*1e-3
    elev[elev<0]=np.nan
    elev[elev>9]=np.nan

    slope=rd.TerrainAttribute(rd.LoadGDAL(path_elev),attrib='slope_degrees')
    slope=extractSecondaryData(ar1,slope,rows1,cols1,rows2, cols2)

    slope[ar1[:,:,maskBand]<=0]=np.nan
    slope[slope<0]=np.nan
    slope[slope>90]=np.nan

    wazim=rd.TerrainAttribute(rd.LoadGDAL(path_elev),attrib='aspect')
    wazim=extractSecondaryData(ar1,wazim,rows1,cols1,rows2, cols2)
    wazim[ar1[:,:,maskBand]<=0]=np.nan
    wazim[wazim<0]=np.nan
    wazim[wazim>360]=np.nan

    return elev, slope, wazim

def MM_topo_correction(R,bands,tiltMap,wazimMap,zenith,azimuth,correction='weak',g=0.2):
    '''
    Performs a MM correction as presented in Richter et al. (2009), with the parameters suggested in the ATCOR Theoretical background document v.9.1.1
        Parameters:
            :class:`numpy.ndarray`:
                - R: reflectance image -- (m,n,b) array
                - bands: wavelengths associated to the bands -- (b,) array
                - altitMap: elevation map of the study site (km) -- (m,n) array
                - tiltMap: slope tilt angle map (degree) -- (m,n) array
                - wazimMap: slope aspect map (degree) -- (m,n) array
            :class:`np.float`
                - zenith: sun zenith angle in degrees
                - azimuth: sun azimuth angle in degrees
                - correction: strength of the correction
                - g: minimum value for the correction factor
        Returns:
            :class:`numpy.ndarray`:
                - np.squeeze(R)*np.squeeze(G): The corrected BOA reflectance array
    '''
    b_vals={'weak':[0.75,0.33],'strong':[0.75,1]}

    Vslope=np.asarray([np.sin(tiltMap)*np.cos(wazimMap),np.sin(tiltMap)*np.sin(wazimMap),np.cos(tiltMap)])
    Vslope=np.moveaxis(Vslope,0,2)

    Vsun=np.asarray([np.sin(zenith)*np.cos(azimuth),np.sin(zenith)*np.sin(azimuth),np.cos(zenith)])

    betai=np.arccos(np.dot(Vslope,Vsun)/np.linalg.norm(Vslope,axis=2)/np.linalg.norm(Vsun))

    zenith=zenith*180/np.pi
    if zenith<45:
        betaT=zenith+20
    elif (zenith>=45) and (zenith<=55):
        betaT=zenith+15
    else:
        betaT=zenith+10
    betaT=betaT*np.pi/180

    b_veg=np.ones(bands.size)
    b_veg[bands<=720]=b_vals[correction][0]
    b_veg[bands>720]=b_vals[correction][1]

    b_soil=0.5

    NDVI,_,_=surehyp.various.getNDVI(R,bands)


    G=(np.cos(betai)/np.cos(betaT))
    G[betai<betaT]=1
    G=np.tile(G,(bands.size,1,1))
    G=np.moveaxis(G,0,2)
    G[NDVI>0.2]=np.power(G[NDVI>0.2],b_veg)
    G[NDVI<0.2]=np.power(G[NDVI<0.2],b_soil)
    G[G<g]=g
    return np.squeeze(R)*np.squeeze(G)

def writeAlbedoFile(R,bands,pathOut='./SMARTS2981-PC_Package/Albedo/Albedo.txt'):
    '''
    Writes a text file containing the average scene BOA reflectance in a format usable by SMARTS
        Parameters:
            :class:`numpy.ndarray`:
                - R: reflectance [0-100] -- (b,) array
                - bands: wavelengths associated to the bands -- (b,) array
            :class:`string`:
                - pathOut: path of the SMARTS Albedo.txt file
        Returns:
            :class:`string`:
                - pathOut: path of the SMARTS Albedo.txt file
    '''
    rho=R[R[:,:,40]>0,:]
    rho=np.round(np.nanmedian(rho,axis=0),3)
    col1=[bands[0]*1E-3,bands[-1]*1E-3,'Albedo']
    b1=np.round(bands*1E-3,3)
    col1.extend(b1.tolist())
    col2=['','','']
    r=rho*1e-2
    r[r>1]=1
    r=np.round(r,4)
    col2.extend(r.tolist())
    dico={'col1':col1,'col2':col2}
    df=pd.DataFrame.from_dict(dico)
    df.to_csv(pathOut,sep=' ',header=False,index=False)
    return pathOut
