import wradlib as wrl
import datetime as dt
import numpy as np


def read_gpm(filename):

    pr_data = wrl.io.read_generic_hdf5(filename)

    lon = pr_data['NS/Longitude']['data']
    lat = pr_data['NS/Latitude']['data']

    year = pr_data['NS/ScanTime/Year']['data']
    month = pr_data['NS/ScanTime/Month']['data']
    dayofmonth = pr_data['NS/ScanTime/DayOfMonth']['data']
    dayofyear = pr_data['NS/ScanTime/DayOfYear']['data']
    hour = pr_data['NS/ScanTime/Hour']['data']
    minute = pr_data['NS/ScanTime/Minute']['data']
    second = pr_data['NS/ScanTime/Second']['data']
    secondofday = pr_data['NS/ScanTime/SecondOfDay']['data']
    millisecond = pr_data['NS/ScanTime/MilliSecond']['data']
    date_array = zip(year, month, dayofmonth,
                     hour, minute, second,
                     millisecond.astype(np.int32) * 1000)
    pr_time = np.array(
        [dt.datetime(d[0], d[1], d[2], d[3], d[4], d[5], d[6]) for d in
         date_array])

    sfc = pr_data['NS/PRE/landSurfaceType']['data']
    pflag = pr_data['NS/PRE/flagPrecip']['data']

    bbflag = pr_data['NS/CSF/flagBB']['data']
    zbb = pr_data['NS/CSF/heightBB']['data']
    bbwidth = pr_data['NS/CSF/widthBB']['data']
    qbb = pr_data['NS/CSF/qualityBB']['data']
    qtype = pr_data['NS/CSF/qualityTypePrecip']['data']
    ptype = pr_data['NS/CSF/typePrecip']['data']

    quality = pr_data['NS/scanStatus/dataQuality']['data']
    refl = pr_data['NS/SLV/zFactorCorrected']['data']

    # Check for bad data
    if max(quality) != 0:
        raise ValueError('GPM contains Bad Data')

    pflag = pflag.astype(np.int8)

    # Determine the dimensions
    ndim = refl.ndim
    if ndim != 3:
        raise ValueError('GPM Dimensions do not match! Needed 3, given {0}'.format(ndim))

    tmp = refl.shape
    nscan = tmp[0]
    nray = tmp[1]
    nbin = tmp[2]

    # Reverse direction along the beam
    # TODO: Why is this reversed?
    refl = refl[::-1]

    # Change pflag=1 to pflag=2 to be consistent with 'Rain certain' in TRMM
    pflag[pflag == 1] = 2

    # Simplify the precipitation types
    ptype = (ptype/1e7).astype(np.int16)

    # Simplify the surface types
    imiss = (sfc == -9999)
    sfc = (sfc/1e2).astype(np.int16) + 1
    sfc[imiss] = 0

    # Set a quality indicator for the BB and precip type data
    # TODO: Why is the `quality` variable overwritten?

    quality = np.zeros((nscan, nray), dtype=np.uint8)

    i1 = ((qbb == 0) | (qbb == 1)) & (qtype == 1)
    quality[i1] = 1

    i2 = ((qbb > 1) | (qtype > 2))
    quality[i2] = 2

    gpm_data = {}
    gpm_data.update({'nscan': nscan, 'nray': nray, 'nbin': nbin,
                     'date': pr_time, 'lon': lon, 'lat': lat,
                     'pflag': pflag, 'ptype': ptype, 'zbb': zbb,
                     'bbwidth': bbwidth, 'sfc': sfc, 'quality': quality,
                     'refl': refl})

    return gpm_data





