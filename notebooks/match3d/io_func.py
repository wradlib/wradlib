import wradlib as wrl
import datetime as dt
import numpy as np
from osgeo import gdal
# flake8: noqa


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
    print(zbb.dtype)
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
        raise ValueError('GPM Dimensions do not match! '
                         'Needed 3, given {0}'.format(ndim))

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


def read_trmm(filename1, filename2):

    # trmm 2A23 and 2A25 data is hdf4
    # it can be read with `read_generic_netcdf`
    pr_data1 = wrl.io.read_generic_netcdf(filename1)
    pr_data2 = wrl.io.read_generic_netcdf(filename2)

    lon = pr_data1['variables']['Longitude']['data']
    lat = pr_data1['variables']['Latitude']['data']

    year = pr_data1['variables']['Year']['data']
    month = pr_data1['variables']['Month']['data']
    dayofmonth = pr_data1['variables']['DayOfMonth']['data']
    dayofyear = pr_data1['variables']['DayOfYear']['data']
    hour = pr_data1['variables']['Hour']['data']
    minute = pr_data1['variables']['Minute']['data']
    second = pr_data1['variables']['Second']['data']
    secondofday = pr_data1['variables']['scanTime_sec']['data']
    millisecond = pr_data1['variables']['MilliSecond']['data']
    date_array = zip(year, month, dayofmonth,
                     hour, minute, second,
                     millisecond.astype(np.int32) * 1000)
    pr_time = np.array(
        [dt.datetime(d[0], d[1], d[2], d[3], d[4], d[5], d[6]) for d in
         date_array])

    pflag = pr_data1['variables']['rainFlag']['data']
    ptype = pr_data1['variables']['rainType']['data']

    status = pr_data1['variables']['status']['data']
    zbb = pr_data1['variables']['HBB']['data'].astype(np.float32)
    bbwidth = pr_data1['variables']['BBwidth']['data'].astype(np.float32)

    quality = pr_data2['variables']['dataQuality']['data']
    refl = pr_data2['variables']['correctZFactor']['data'] / 100.
    print(refl.dtype, refl)

    # Ground clutter
    refl[refl == -8888.] = np.nan
    # Misssing data
    refl[refl == -9999.] = np.nan
    # Scaling
    refl /= 100.

    # Check for bad data
    if max(quality) != 0:
        raise ValueError('TRMM contains Bad Data')

    # Determine the dimensions
    ndim = refl.ndim
    if ndim != 3:
        raise ValueError('TRMM Dimensions do not match!'
                         'Needed 3, given {0}'.format(ndim))

    tmp = refl.shape
    nscan = tmp[0]
    nray = tmp[1]
    nbin = tmp[2]

    # Reverse direction along the beam
    # TODO: Why is this reversed?
    refl = refl[::-1]

    # Simplify the precipitation flag
    ipos = (pflag >= 10) & (pflag <= 20)
    icer = (pflag >= 20)
    pflag[ipos] = 1
    pflag[icer] = 2

    # Simplify the precipitation types
    istr = (ptype >= 100) & (ptype <= 200)
    icon = (ptype >= 200) & (ptype <= 300)
    ioth = (ptype >= 300)
    inone = (ptype == -88)
    imiss = (ptype == -99)
    ptype[istr] = 1
    ptype[icon] = 2
    ptype[ioth] = 3
    ptype[inone] = 0
    ptype[imiss] = -1

    # Extract the surface type
    sfc = np.zeros((nscan, nray), dtype=np.uint8)
    i0 = (status == 168)
    sfc[i0] = 0
    i1 = (status % 10 == 0)
    sfc[i1] = 1
    i2 = ((status - 1) % 10 == 0)
    sfc[i2] = 2
    i3 = ((status - 3) % 10 == 0)
    sfc[i3] = 3
    i4 = ((status - 4) % 10 == 0)
    sfc[i4] = 4
    i5 = ((status - 5) % 10 == 0)
    sfc[i5] = 5
    i9 = ((status - 9) % 10 == 0)
    sfc[i9] = 9

    # Extract 2A23 quality
    # TODO: Why is the `quality` variable overwritten?
    quality = np.zeros((nscan, nray), dtype=np.uint8)
    i0 = (status == 168)
    quality[i0] = 0
    i1 = (status < 50)
    quality[i1] = 1
    i2 = ((status >= 50) & (status < 109))
    quality[i2] = 2

    trmm_data = {}
    trmm_data.update({'nscan': nscan, 'nray': nray, 'nbin': nbin,
                      'date': pr_time, 'lon': lon, 'lat': lat,
                      'pflag': pflag, 'ptype': ptype, 'zbb': zbb,
                      'bbwidth': bbwidth, 'sfc': sfc, 'quality': quality,
                      'refl': refl})

    return trmm_data


# Hidden subdatasets in TRMM HDF4 files
sds_hidden_2a23 = {
    "0": {"name": "Year",         "units": "years",   "dtype": np.integer},
    "1": {"name": "Month",        "units": "months",  "dtype": np.integer},
    "2": {"name": "DayOfMonth",   "units": "days",    "dtype": np.integer},
    "3": {"name": "Hour",         "units": "hours",   "dtype": np.integer},
    "4": {"name": "Minute",       "units": "minutes", "dtype": np.integer},
    "5": {"name": "Second",       "units": "s",       "dtype": np.integer},
    "6": {"name": "MilliSecond",  "units": "ms",      "dtype": np.integer}
}
# NOT USED
#    "7": {"name": "DayOfYear",    "units": "days",    "dtype": np.integer},
#    "8": {"name": "scanTime_sec", "units": "s",       "dtype": np.float},

sds_hidden_2a25 = {
    "8": {"name": "dataQuality", "units": None, "dtype": np.integer}
}
# NOT USED
# "0": {"name": "Year",         "units": "years",   "dtype": np.integer},
# "1": {"name": "Month", "units": "months", "dtype": np.integer},
# "2": {"name": "DayOfMonth", "units": "days", "dtype": np.integer},
# "3": {"name": "Hour", "units": "hours", "dtype": np.integer},
# "4": {"name": "Minute", "units": "minutes", "dtype": np.integer},
# "5": {"name": "Second", "units": "s", "dtype": np.integer},
# "6": {"name": "MilliSecond", "units": "ms", "dtype": np.integer}
# "7": {"name": "DayOfYear",    "units": "days",    "dtype": np.integer},
# "9": {"name": "scanTime_sec", "units": "s",       "dtype": np.float},

trmm_sd_prefix = "HDF4_SDS:UNKNOWN:"


def read_trmm_gdal(f2a23, f2a25):
    """
    TRMM 2A23 and 2A25 data comes in hdf4. It can be read with
    `read_generic_netcdf`, but only if netcDF4 was compiled with hdf4 support.
    If not, we need to use gdal.

    The use of this function is discouraged in case read_trmm works. This
    is because the mapping is fragile since GDAL is unable to access all
    subdatasets by name. Unfortunately, we have to assume that the order
    of subdatasets remains the same. However, this might not be true depending
    on how the TRMM file was generated from the STORM database!

    """
    vars2a23 = {"Latitude": "lat", "Longitude": "lon", "rainFlag": "pflag",
                "rainType": "ptype", "status": "status", "HBB": "zbb",
                "BBwidth": "bbwidth"}
    vars2a25 = {"correctZFactor": "refl"}

    # Extract "visible" 2A23 subdatasets into output dictionary
    trmm = {}
    hdf = gdal.Open(f2a23)
    sds = hdf.GetSubDatasets()
    for var in vars2a23.keys():
        for sd in sds:
            if (var in sd[1]):
                trmm[vars2a23[var]] = gdal.Open(sd[0]).ReadAsArray()
    # TODO: What about the flags (-1111: no bb, -8888: no rain, -9999: no data)
    trmm["zbb"] = trmm["zbb"].astype(np.float32)
    trmm["bbwidth"] = trmm["bbwidth"].astype(np.float32)

    # Extract "hidden" 2A23 subdatasets into output dictionary
    d = {}
    for sd in sds_hidden_2a23.keys():
        sdstr = "%s%s:%s" % (trmm_sd_prefix, hdf.GetDescription(), sd)
        tmp = gdal.Open(sdstr)
        d[sds_hidden_2a23[sd]["name"]] = tmp.ReadAsArray()
        # Check consistency
        #   of units
        if not sds_hidden_2a23[sd]["units"] == tmp.GetMetadata_Dict()["units"]:
            raise ValueError("Unexpected TRMM HDF4 subdataset units: %s"
                             "instead of %s" % (sds_hidden_2a23[sd]["units"],
                                                tmp.GetMetadata_Dict()["units"]))
        #   of data type
        if not np.issubdtype(d[sds_hidden_2a23[sd]["name"]].dtype,
                             sds_hidden_2a23[sd]["dtype"]):
            raise ValueError("Unexpected TRMM HDF4 subdataset dtype:"
                             "%s instead of %s" %
                             (d[sds_hidden_2a23[sd]["name"]].dtype,
                              sds_hidden_2a23[sd]["dtype"]))

    date_array = zip(d["Year"].astype(np.int32).ravel(),
                     d["Month"].astype(np.int32).ravel(),
                     d["DayOfMonth"].astype(np.int32).ravel(),
                     d["Hour"].astype(np.int32).ravel(),
                     d["Minute"].astype(np.int32).ravel(),
                     d["Second"].astype(np.int32).ravel(),
                     # Millisecond to Microsecond
                     d["MilliSecond"].astype(np.int32).ravel() * 1000)

    trmm["date"] = np.array(
        [dt.datetime(d[0], d[1], d[2], d[3], d[4], d[5], d[6]) for d in
         date_array]
    )

    # Extract "visible" 2A25 subdatasets into output dictionary
    hdf = gdal.Open(f2a25)
    sds = hdf.GetSubDatasets()
    for var in vars2a25.keys():
        for sd in sds:
            if (var in sd[1]):
                trmm[vars2a25[var]] = np.swapaxes(
                    gdal.Open(sd[0]).ReadAsArray(), 0, 1)
    trmm["refl"] = trmm["refl"].astype(np.float32)
    # Ground clutter
    trmm["refl"][trmm["refl"] == -8888.] = np.nan
    # Misssing data
    trmm["refl"][trmm["refl"] == -9999.] = np.nan
    # Scaling
    trmm["refl"] = trmm["refl"] / 100.

    # Extract "hidden" 2A25 subdatasets into output dictionary
    d2 = {}
    for sd in sds_hidden_2a25.keys():
        sdstr = "%s%s:%s" % (trmm_sd_prefix, hdf.GetDescription(), sd)
        tmp = gdal.Open(sdstr)
        d2[sds_hidden_2a25[sd]["name"]] = tmp.ReadAsArray()
        # Check consistency
        #   of units
        if "units" not in tmp.GetMetadata_Dict().keys():
            # Quality flag does not have a units keyword
            if not sds_hidden_2a25[sd]["units"] is None:
                raise ValueError("Unexpected TRMM HDF4 subdataset units: %s"
                                 "is not expected to have units." %
                                 sds_hidden_2a25[sd]["name"])
        elif not sds_hidden_2a25[sd]["units"] == tmp.GetMetadata_Dict()["units"]:
            raise ValueError("Unexpected TRMM HDF4 subdataset units: %s"
                             "instead of %s" % (sds_hidden_2a25[sd]["units"],
                                                tmp.GetMetadata_Dict()["units"]))
        #   of data type
        if not np.issubdtype(d2[sds_hidden_2a25[sd]["name"]].dtype,
                             sds_hidden_2a25[sd]["dtype"]):
            raise ValueError("Unexpected TRMM HDF4 subdataset dtype:"
                             "%s instead of %s" %
                             (d2[sds_hidden_2a25[sd]["name"]].dtype,
                              sds_hidden_2a25[sd]["dtype"]))

    # Check for bad data
    if d2["dataQuality"].max() != 0:
        raise ValueError('TRMM contains Bad Data.')

    # Check dimensions
    ndim = trmm["refl"].ndim
    if ndim != 3:
        raise ValueError("TRMM Dimensions do not match!"
                         "Needed 3, given {0}".format(ndim))

    tmp = trmm["refl"].shape
    nscan = tmp[0]
    nray = tmp[1]
    nbin = tmp[2]

    # Reverse direction along the beam
    # TODO: Why is this reversed?
    trmm["refl"] = trmm["refl"][::-1]

    # Simplify the precipitation flag
    ipos = (trmm["pflag"] >= 10) & (trmm["pflag"] <= 20)
    icer = (trmm["pflag"] >= 20)
    trmm["pflag"][ipos] = 1
    trmm["pflag"][icer] = 2

    # Simplify the precipitation types
    istr = (trmm["ptype"] >= 100) & (trmm["ptype"] <= 200)
    icon = (trmm["ptype"] >= 200) & (trmm["ptype"] <= 300)
    ioth = (trmm["ptype"] >= 300)
    inone = (trmm["ptype"] == -88)
    imiss = (trmm["ptype"] == -99)
    trmm["ptype"][istr] = 1
    trmm["ptype"][icon] = 2
    trmm["ptype"][ioth] = 3
    trmm["ptype"][inone] = 0
    trmm["ptype"][imiss] = -1

    # Extract the surface type
    sfc = np.zeros((nscan, nray), dtype=np.uint8)
    i0 = (trmm["status"] == 168)
    sfc[i0] = 0
    i1 = (trmm["status"] % 10 == 0)
    sfc[i1] = 1
    i2 = ((trmm["status"] - 1) % 10 == 0)
    sfc[i2] = 2
    i3 = ((trmm["status"] - 3) % 10 == 0)
    sfc[i3] = 3
    i4 = ((trmm["status"] - 4)  % 10 == 0)
    sfc[i4] = 4
    i5 = ((trmm["status"] - 5) % 10 == 0)
    sfc[i5] = 5
    i9 = ((trmm["status"] - 9) % 10 == 0)
    sfc[i9] = 9

    # Extract 2A23 quality
    # TODO: Why is the `quality` variable overwritten?
    trmm["quality"] = np.zeros((nscan, nray), dtype=np.uint8)
    i0 = (trmm["status"] == 168)
    sfc[i0] = 0
    i1 = (trmm["status"] < 50)
    sfc[i1] = 1
    i2 = ((trmm["status"] >= 50) & (trmm["status"] < 109))
    sfc[i2] = 2

    # Update output dictionary
    trmm["sfs"] = sfc
    del trmm["status"]
    trmm["nscan"] = nscan
    trmm["nray"] = nray
    trmm["nbin"] = nbin

    return trmm


def _get_tilts(dic):
    i = 0
    for k in dic.keys():
        if 'dataset' in k:
            i += 1
    return i


def read_gr(filename, loaddata=True):

    gr_data = wrl.io.read_generic_netcdf(filename)
    dat = gr_data['what']['date']
    tim = gr_data['what']['time']
    date = dt.datetime.strptime(dat + tim, "%Y%d%m%H%M%S")
    source = gr_data['what']['source']

    lon = gr_data['where']['lon']
    lat = gr_data['where']['lat']
    alt = gr_data['where']['height']

    if gr_data['what']['object'] == 'PVOL':
        ntilt = _get_tilts(gr_data)
    else:
        raise ValueError('GR file is no PPI/Volume File')

    ngate = np.zeros(ntilt, dtype=np.int16)
    nbeam = np.zeros(ntilt)
    elang = np.zeros(ntilt)
    r0 = np.zeros(ntilt)
    dr = np.zeros(ntilt)
    a0 = np.zeros(ntilt)

    for i in range(0, ntilt):
        dset = gr_data['dataset{0}'.format(i+1)]
        a0[i] = dset['how']['astart']
        elang[i] = dset['where']['elangle']
        ngate[i] = dset['where']['nbins']
        r0[i] = dset['where']['rstart']
        dr[i] = dset['where']['rscale']
        nbeam[i] = dset['where']['nrays']

    if ((len(np.unique(r0)) != 1) |
            (len(np.unique(dr)) != 1) |
            (len(np.unique(a0)) != 1) |
            (len(np.unique(nbeam)) != 1) |
            (nbeam[0] != 360)):
        raise ValueError('GroundRadar Data layout dos not match')

    gr_dict = {}
    gr_dict.update({'source': source, 'date': date, 'lon': lon, 'lat': lat,
                    'alt': alt, 'ngate': ngate, 'nbeam': nbeam, 'ntilt': ntilt,
                    'r0': r0, 'dr': dr, 'a0': a0, 'elang': elang})
    if not loaddata:
        return gr_dict

    sdate = []
    refl = []
    for i in range(0, ntilt):
        dset = gr_data['dataset{0}'.format(i+1)]
        dat = dset['what']['startdate']
        tim = dset['what']['starttime']
        date = dt.datetime.strptime(dat + tim, "%Y%d%m%H%M%S")
        sdate.append(date)
        data = dset['data1']
        quantity = data['what']['quantity']
        factor = data['what']['gain']
        offset = data['what']['offset']
        if quantity == 'DBZH':
            dat = data['variables']['data']['data'] * factor + offset
            refl.append(dat)

    sdate = np.array(sdate)
    refl = np.array(refl)

    gr_dict.update({'sdate': sdate, 'refl': refl})

    return gr_dict


def _get_tilts2(dic):
    i = 0
    for k in dic.keys():
        if 'dataset' in k:
            i += 1
    return i/5


def read_gr2(filename, loaddata=True):

    gr_data = wrl.io.read_generic_hdf5(filename)
    dat = gr_data['what']['attrs']['date'].decode()
    tim = gr_data['what']['attrs']['time'].decode()
    date = dt.datetime.strptime(dat + tim, "%Y%d%m%H%M%S")
    source = gr_data['what']['attrs']['source']

    lon = gr_data['where']['attrs']['lon']
    lat = gr_data['where']['attrs']['lat']
    alt = gr_data['where']['attrs']['height']

    if gr_data['what']['attrs']['object'].decode() == 'PVOL':
        ntilt = int(_get_tilts2(gr_data))
        print("ntilt:", ntilt)
    else:
        raise ValueError('GR file is no PPI/Volume File')

    ngate = np.zeros(ntilt, dtype=np.int16)
    nbeam = np.zeros(ntilt)
    elang = np.zeros(ntilt)
    r0 = np.zeros(ntilt)
    dr = np.zeros(ntilt)
    a0 = np.zeros(ntilt)

    for i in np.arange(0, ntilt, dtype=np.uint8):
        a0[i] = gr_data['dataset{0}/how'.format(i+1)]['attrs']['astart']
        elang[i] = gr_data['dataset{0}/where'.format(i+1)]['attrs']['elangle']
        ngate[i] = gr_data['dataset{0}/where'.format(i+1)]['attrs']['nbins']
        r0[i] = gr_data['dataset{0}/where'.format(i+1)]['attrs']['rstart']
        dr[i] = gr_data['dataset{0}/where'.format(i+1)]['attrs']['rscale']
        nbeam[i] = gr_data['dataset{0}/where'.format(i+1)]['attrs']['nrays']

    if ((len(np.unique(r0)) != 1) |
            (len(np.unique(dr)) != 1) |
            (len(np.unique(a0)) != 1) |
            (len(np.unique(nbeam)) != 1) |
            (nbeam[0] != 360)):
        raise ValueError('GroundRadar Data layout dos not match')

    gr_dict = {}
    gr_dict.update({'source': source, 'date': date, 'lon': lon, 'lat': lat,
                    'alt': alt, 'ngate': ngate, 'nbeam': nbeam, 'ntilt': ntilt,
                    'r0': r0, 'dr': dr, 'a0': a0, 'elang': elang})
    if not loaddata:
        return gr_dict

    sdate = []
    refl = []
    for i in np.arange(0, ntilt, dtype=np.uint8):
        dat = gr_data['dataset{0}/what'.format(i+1)]['attrs']['startdate'].decode()
        tim = gr_data['dataset{0}/what'.format(i+1)]['attrs']['starttime'].decode()
        date = dt.datetime.strptime(dat + tim, "%Y%d%m%H%M%S")
        sdate.append(date)
        quantity = gr_data['dataset{0}/data1/what'.format(i+1)]['attrs']['quantity']
        factor = gr_data['dataset{0}/data1/what'.format(i+1)]['attrs']['gain']
        offset = gr_data['dataset{0}/data1/what'.format(i+1)]['attrs']['offset']
        if quantity.decode() == 'DBZH':
            dat = gr_data['dataset{0}/data1/data'.format(i+1)]['data'] * factor + offset
            refl.append(dat)

    sdate = np.array(sdate)
    refl = np.array(refl)

    gr_dict.update({'sdate': sdate, 'refl': refl})

    return gr_dict



if __name__ == '__main__':

    out = read_trmm_gdal(r"E:/src/git/heistermann/wradlib-data/trmm/2A-RW-BRS.TRMM.PR.2A23.20100206-S111422-E111519.069662.7.HDF",
                         r"E:/src/git/heistermann/wradlib-data/trmm/2A-RW-BRS.TRMM.PR.2A25.20100206-S111422-E111519.069662.7.HDF")
