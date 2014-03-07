#-------------------------------------------------------------------------------
# Name:         io
# Purpose:
#
# Authors:      Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:      26.10.2011
# Copyright:    (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

"""
Raw Data I/O
^^^^^^^^^^^^

Please have a look at the tutorial :doc:`tutorial_supported_formats` for an introduction
on how to deal with different file formats.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   readDX
   writePolygon2Text
   read_EDGE_netcdf
   read_BUFR
   read_OPERA_hdf5
   read_GAMIC_hdf5
   read_RADOLAN_composite

"""

# standard libraries

import sys
import re
import datetime as dt
import pytz
import cPickle as pickle
import os
import warnings

# site packages
import h5py
import numpy as np
import netCDF4 as nc # ATTENTION: Needs to be imported AFTER h5py, otherwise ungraceful crash
from osgeo import gdal
import util


# current DWD file naming pattern (2008) for example:
# raa00-dx_10488-200608050000-drs---bin
dwdpattern = re.compile('raa..-(..)[_-]([0-9]{5})-([0-9]*)-(.*?)---bin')


def _getTimestampFromFilename(filename):
    """Helper function doing the actual work of getDXTimestamp"""
    time = dwdpattern.search(filename).group(3)
    if len(time) == 10:
        time = '20' + time
    return dt.datetime.strptime(time, '%Y%m%d%H%M')


def getDXTimestamp(name, tz=pytz.utc):
    """Converts a dx-timestamp (as part of a dx-product filename) to a python datetime.object.

    Parameters
    ----------
    name : string representing a DWD product name

    tz : timezone object (see pytz package or datetime module for explanation)
         in case the timezone of the data is not UTC

    opt : currently unused

    Returns
    -------
    time : timezone-aware datetime.datetime object
    """
    return _getTimestampFromFilename(name).replace(tzinfo=tz)


def unpackDX(raw):
    """function removes DWD-DX-product bit-13 zero packing"""
    # data is encoded in the first 12 bits
    data = 4095
    # the zero compression flag is bit 13
    flag = 4096

    beam = []

##    # naive version
##    # 49193 function calls in 0.772 CPU seconds
##    # 20234 function calls in 0.581 CPU seconds
##    for item in raw:
##        if item & flag:
##            beam.extend([0]* (item & data))
##        else:
##            beam.append(item & data)

    # performance version - hopefully
    # 6204 function calls in 0.149 CPU seconds

    # get all compression cases
    flagged = np.where(raw & flag)[0]

    # if there is no zero in the whole data, we can return raw as it is
    if flagged.size == 0:
        assert raw.size == 128
        return raw

    # everything until the first flag is normal data
    beam.extend(raw[0:flagged[0]])

    # iterate over all flags except the last one
    for this, next in zip(flagged[:-1],flagged[1:]):
        # create as many zeros as there are given within the flagged
        # byte's data part
        beam.extend([0]* (raw[this] & data))
        # append the data until the next flag
        beam.extend(raw[this+1:next])

    # process the last flag
    # add zeroes
    beam.extend([0]* (raw[flagged[-1]] & data))

    # add remaining data
    beam.extend(raw[flagged[-1]+1:])

    # return the data
    return np.array(beam)


def parse_DX_header(header):
    """Internal function to retrieve and interpret the ASCII header of a DWD
    DX product file."""
    # empty container
    out = {}
    # RADOLAN product type def
    out["producttype"] = header[0:2]
    # file time stamp as Python datetime object
    out["datetime"] = dt.datetime.strptime(header[2:8]+header[13:17]+"00",
                                           "%d%H%M%m%y%S")
    # radar location ID (always 10000 for composites)
    out["radarid"] = header[8:13]
    pos_BY = header.find("BY")
    pos_VS = header.find("VS")
    pos_CO = header.find("CO")
    pos_CD = header.find("CD")
    pos_CS = header.find("CS")
    pos_EP = header.find("EP")
    pos_MS = header.find("MS")

    out['bytes'] = int(header[pos_BY+2:pos_BY+7])
    out['version'] = header[pos_VS+2:pos_VS+4]
    out['cluttermap'] = int(header[pos_CO+2:pos_CO+3])
    out['dopplerfilter'] = int(header[pos_CD+2:pos_CD+3])
    out['statfilter'] = int(header[pos_CS+2:pos_CS+3])
    out['elevprofile'] = [float(header[pos_EP+2+3*i:pos_EP+2+3*(i+1)]) for i in range(8)]
    out['message'] = header[pos_MS+5:pos_MS+5+int(header[pos_MS+2:pos_MS+5])]

    return out


def readDX(filename):
    r"""Data reader for German Weather Service DX product raw radar data files.

    This product uses a simple algorithm to compress zero values to reduce data
    file size.

    Notes
    -----
    While the format appears to be well defined, there have been reports on DX-
    files that seem to produce errors. e.g. while one file usually contains a
    360 degree by 128 1km range bins, there are files, that contain 361 beams.

    Also, while usually azimuths are stored monotonously in ascending order,
    this is not guaranteed by the format. This routine does not (yet) check
    for this and directly returns the data in the order found in the file.
    If you are in doubt, check the 'azim' attribute.

    Be aware that this function does no extensive checking on its output.
    If e.g. beams contain different numbers of range bins, the resulting data
    will not be a 2-D array but a 1-D array of objects, which will most probably
    break calling code. It was decided to leave the handling of these
    (hopefully) rare events to the user, who might still be able to retrieve
    some reasonable data, instead of raising an exception, making it impossible
    to get any data from a file containing errors.

    Parameters
    ----------
    filename : binary file of DX raw data

    Returns
    -------
    data : numpy array of image data [dBZ]; shape (360,128)

    attributes : dictionary of attributes - currently implemented keys:

        - 'azim' - azimuths np.array of shape (360,)
        - 'elev' - elevations (1 per azimuth); np.array of shape (360,)
        - 'clutter' - clutter mask; boolean array of same shape as `data`;
            corresponds to bit 15 set in each dataset.
        - 'bytes'- the total product length (including header). Apparently,
            this value may be off by one byte for unknown reasons
        - 'version'- a product version string - use unknown
        - 'cluttermap' - number of the (DWD internal) cluttermap used
        - 'dopplerfilter' - number of the dopplerfilter used (DWD internal)
        - 'statfilter' - number of a statistical filter used (DWD internal)
        - 'elevprofile' - as stated in the format description, this list
            indicates the elevations in the eight 45 degree sectors. These
            sectors need not start at 0 degrees north, so it is advised to
            explicitly evaluate the `elev` attribute, if elevation information
            is needed.
        - 'message' - additional text stored in the header.
    """

    azimuthbitmask = 2**(14-1)
    databitmask = 2**(13-1) - 1
    clutterflag = 2**15
    dataflag = 2**13 -1
    # open the DX file in binary mode for reading
    if type(filename) == file:
        f = filename
    else:
        f = open(filename, 'rb')

    # header string for later processing
    header = ''
    atend = False
    # read header
    while True :
        mychar = f.read(1)
        # 0x03 signals the end of the header but sometimes there might be
        # an additional 0x03 char after that
        if (mychar == chr(3)):
            atend=True
        if mychar != chr(3) and atend:
            break
        header = header + mychar

    attrs = parse_DX_header(header)

    # position file at end of header
    f.seek(len(header))

    # read number of bytes as declared in the header
    # intermediate fix:
    # if product length is uneven but header is even (e.g. because it has two
    # chr(3) at the end, read one byte less
    buflen = attrs['bytes'] - len(header)
    if (buflen % 2) !=0:
        # make sure that this is consistent with our assumption
        # i.e. contact DWD again, if DX files show up with uneven byte lengths
        # *and* only one 0x03 character
        #assert header[-2] == chr(3)
        buflen -= 1

    buf = f.read(buflen)
    # we can interpret the rest directly as a 1-D array of 16 bit unsigned ints
    raw = np.frombuffer(buf, dtype='uint16')

    # reading finished, close file, but only if we opened it.
    if type(filename)!=file:
        f.close()

    # a new ray/beam starts with bit 14 set
    # careful! where always returns its results in a tuple, so in order to get
    # the indices we have to retrieve element 0 of this tuple
    newazimuths = np.where( raw == azimuthbitmask )[0]  ###Thomas kontaktieren!!!!!!!!!!!!!!!!!!!

    # for the following calculations it is necessary to have the end of the data
    # as the last index
    newazimuths = np.append(newazimuths,len(raw))

    # initialize our list of rays/beams
    beams = []
    # initialize our list of elevations
    elevs = []
    # initialize our list of azimuths
    azims = []

    # iterate over all beams
    for i in range(newazimuths.size-1):
        # unpack zeros
        beam = unpackDX(raw[newazimuths[i]+3:newazimuths[i+1]])
        beams.append(beam)
        elevs.append((raw[newazimuths[i]+2] & databitmask)/10.)
        azims.append((raw[newazimuths[i]+1] & databitmask)/10.)

    beams = np.array(beams)

    #attrs =  {}
    attrs['elev']  = np.array(elevs)
    attrs['azim'] = np.array(azims)
    attrs['clutter'] = (beams & clutterflag) != 0

    # converting the DWD rvp6-format into dBZ data and return as numpy array together with attributes
    return (beams & dataflag) * 0.5 - 32.5, attrs


def _write_polygon2txt(f, idx, vertices):
    f.write('%i %i\n'%idx)
    for i, vert in enumerate(vertices):
        f.write('%i '%(i,))
        f.write('%f %f %f %f\n' % tuple(vert))


def writePolygon2Text(fname, polygons):
    """Writes Polygons to a Text file which can be interpreted by ESRI \
    ArcGIS's "Create Features from Text File (Samples)" tool.

    This is (yet) only a convenience function with limited functionality.
    E.g. interior rings are not yet supported.

    Parameters
    ----------
    fname : string
        name of the file to save the vertex data to
    polygons : list of lists
        list of polygon vertices.
        Each vertex itself is a list of 3 coordinate values and an
        additional value. The third coordinate and the fourth value may be nan.

    Returns
    -------
    None

    Notes
    -----
    As Polygons are closed shapes, the first and the last vertex of each
    polygon **must** be the same!

    Examples
    --------
    Writes two triangle Polygons to a text file

    >>> poly1 = [[0.,0.,0.,0.],[0.,1.,0.,1.],[1.,1.,0.,2.],[0.,0.,0.,0.]]
    >>> poly2 = [[0.,0.,0.,0.],[0.,1.,0.,1.],[1.,1.,0.,2.],[0.,0.,0.,0.]]
    >>> polygons = [poly1, poly2]
    >>> writePolygon2Text('polygons.txt', polygons)

    The resulting text file will look like this::

        Polygon
        0 0
        0 0.000000 0.000000 0.000000 0.000000
        1 0.000000 1.000000 0.000000 1.000000
        2 1.000000 1.000000 0.000000 2.000000
        3 0.000000 0.000000 0.000000 0.000000
        1 0
        0 0.000000 0.000000 0.000000 0.000000
        1 0.000000 1.000000 0.000000 1.000000
        2 1.000000 1.000000 0.000000 2.000000
        3 0.000000 0.000000 0.000000 0.000000
        END

    """
    with open(fname, 'w') as f:
        f.write('Polygon\n')
        count = 0
        for vertices in polygons:
            _write_polygon2txt(f, (count, 0), vertices)
            count += 1
        f.write('END\n')


def read_EDGE_netcdf(filename, enforce_equidist=False):
    """Data reader for netCDF files exported by the EDGE radar software

    The corresponding NetCDF files from the EDGE software typically contain only
    one variable (e.g. reflectivity) for one elevation angle (sweep). The elevation
    angle is specified in the attributes keyword "Elevation".

    Please note that the radar might not return data with equidistant azimuth angles.
    In case you need equidistant azimuth angles, please set enforce_equidist to True.

    Parameters
    ----------
    filename : path of the netCDF file
    enforce_equidist : boolean
        Set True if the values of the azimuth angles should be forced to be equidistant
        default value is False

    Returns
    -------
    output : numpy array of image data (dBZ), dictionary of attributes

    """
    try:
        # read the data from file
        dset = nc.Dataset(filename)
        data = dset.variables[dset.TypeName][:]
        # Check azimuth angles and rotate image
        az = dset.variables['Azimuth'][:]
        # These are the indices of the minimum and maximum azimuth angle
        ix_minaz = np.argmin(az)
        ix_maxaz = np.argmax(az)
        if enforce_equidist:
            az = np.linspace(np.round(az[ix_minaz],2), np.round(az[ix_maxaz],2), len(az))
        else:
            az = np.roll(az, -ix_minaz)
        # rotate accordingly
        data = np.roll(data, -ix_minaz, axis=0)
        data = np.where(data==dset.getncattr('MissingData'), np.nan, data)
        # Ranges
        binwidth = (dset.getncattr('MaximumRange-value') * 1000.) / len(dset.dimensions['Gate'])
        r = np.arange(binwidth, (dset.getncattr('MaximumRange-value') * 1000.) + binwidth, binwidth)
        # collect attributes
        attrs =  {}
        for attrname in dset.ncattrs():
            attrs[attrname] = dset.getncattr(attrname)
##        # Limiting the returned range
##        if range_lim and range_lim / binwidth <= data.shape[1]:
##            data = data[:,:range_lim / binwidth]
##            r = r[:range_lim / binwidth]
        # Set additional metadata attributes
        attrs['az'] = az
        attrs['r']  = r
        attrs['sitecoords'] = (attrs['Latitude'], attrs['Longitude'], attrs['Height'])
        attrs['time'] = dt.datetime.utcfromtimestamp(attrs.pop('Time'))
        attrs['max_range'] = data.shape[1] * binwidth
    except:
        raise
    finally:
        dset.close()

    return data, attrs


def read_BUFR(buffile):
    """Main BUFR interface: Decodes BUFR file and returns metadata and values

    The actual function refererence is contained in :doc:`wradlib.bufr.decodebufr`.

    """
    # wradib modules
    bufr = util.import_optional("wradlib.bufr")
    return bufr.decodebufr(buffile)


def parse_DWD_quant_composite_header(header):
    """Parses the ASCII header of a DWD quantitative composite file

    Parameters
    ----------
    header : string (ASCII header)

    Returns
    -------
    output : dictionary of metadata retreived from file header

    """
    # empty container
    out = {}
    # RADOLAN product type def
    out["producttype"] = header[0:2]
    # file time stamp as Python datetime object
    out["datetime"] = dt.datetime.strptime(header[2:8]+header[13:17]+"00",
                                           "%d%H%M%m%y%S")
    # radar location ID (always 10000 for composites)
    out["radarid"] = header[8:13]
    pos_VS = header.find("VS")
    pos_SW = header.find("SW")
    pos_PR = header.find("PR")
    pos_INT = header.find("INT")
    pos_GP = header.find("GP")
    pos_MS = header.find("MS")
    if pos_VS > -1:
        out["maxrange"] = {0:"100 km and 128 km (mixed)",
                           1: "100 km",
                           2:"128 km",
                           3:"150 km" }[int(header[(pos_VS+2):pos_VS+4])]
    else:
        out["maxrange"] = "100 km"
    out["radolanversion"] = header[(pos_SW+2):pos_SW+11]
    out["precision"] = 10**int(header[pos_PR+4:pos_PR+7])
    out["intervalseconds"] = int(header[(pos_INT+3):pos_INT+7])*60
    dimstrings = header[(pos_GP+2):pos_GP+11].strip().split("x")
    out["nrow"] = int(dimstrings[0])
    out["ncol"] = int(dimstrings[1])
    locationstring = header[(pos_MS+2):].strip().split("<")[1].strip().strip(">")
    out["radarlocations"] = locationstring.split(",")
    return out


def read_RADOLAN_composite(fname, missing=-9999):
    """Read quantitative radar composite format of the German Weather Service

    The quantitative composite format of the DWD (German Weather Service) was
    established in the course of the `RADOLAN project <http://www.dwd.de/radolan>`
    and includes several file types, e.g. RX, RO, RK, RZ, RP, RT, RC, RI, RG and
    many, many more (see format description on the project homepage, [DWD2009]).

    At the moment, the national RADOLAN composite is a 900 x 900 grid with 1 km
    resolution and in polar-stereographic projection.

    **Beware**: This function already evaluates and applies the so-called PR factor which is
    specified in the header section of the RADOLAN files. The raw values in an RY file
    are in the unit 0.01 mm/5min, while read_RADOLAN_composite returns values
    in mm/5min (i. e. factor 100 higher). The factor is also returned as part of
    attrs dictionary under keyword "precision".

    Parameters
    ----------
    fname : path to the composite file

    missing : value assigned to no-data cells

    Returns
    -------
    output : tuple of two items (data, attrs)
        - data : numpy array of shape (number of rows, number of columns)
        - attrs : dictionary of metadata information from the file header

    References
    ----------

    .. [DWD2009] Germany Weather Service (DWD), 2009: RADLOAN/RADVO-OP -
        Beschreibung des Kompositformats, Version 2.2.1. Offenbach, Germany,
        URL: http://dwd.de/radolan (in German)

    """
    mask = 4095 # max value integer
    NODATA = missing
    header = '' # header string for later processing
    # open file handle
    f = open(fname, 'rb')
    # read header
    while True :
        mychar = f.read(1)
        if mychar == chr(3) :
            break
        header = header + mychar
    attrs = parse_DWD_quant_composite_header(header)
    attrs["nodataflag"] = NODATA
    if not attrs["radarid"]=="10000":
        warnings.warn("WARNING: You are using function e" +
                      "wradlib.io.read_RADOLAN_composit for a non " +
                      "composite file.\n " +
                      "This might work...but please check the validity " +
                      "of the results")
    if attrs["producttype"] == "RX":
        # read the actual data
        indat = f.read(attrs["nrow"]*attrs["ncol"])
        # convert from 8-bit integers
        # and upgrade to 32-bit ints, so that nodata values may be inserted
        arr = np.frombuffer(indat, np.uint8).astype(np.int)
        arr = np.where(arr==250,NODATA,arr)
        clutter = np.where(arr==249)[0]
    else:
        # read the actual data
        indat = f.read(attrs["nrow"]*attrs["ncol"]*2)
        # convert to 16-bit integers
        arr = np.frombuffer(indat, np.uint16).astype(np.int)
        # evaluate bits 14, 15 and 16
        nodata   = np.where(arr & int("10000000000000",2))
        negative = np.where(arr & int("100000000000000",2))
        clutter  = np.where(arr & int("1000000000000000",2))
        # mask out the last 4 bits
        arr = arr & mask
        # consider negative flag if product is RD (differences from adjustment)
        if attrs["producttype"]=="RD":
            # NOT TESTED, YET
            arr[negative] = -arr[negative]
        # apply precision factor
        arr *= attrs["precision"]
        # set nodata value
        arr[nodata] = NODATA
    # bring it into shape
    arr = arr.reshape( (attrs["nrow"], attrs["ncol"]) )

    # append clutter mask
    attrs['cluttermask'] = clutter

    # close the file
    f.close()

    return arr, attrs

def browse_hdf5_group(grp):
    """Browses one hdf5 file level
    """
    pass

def read_generic_hdf5(fname):
    """Reads hdf5 files according to their structure

    In contrast to other file readers under wradlib.io, this function will *not* return
    a two item tuple with (data, metadata). Instead, this function returns ONE
    dictionary that contains all the file contents - both data and metadata. The keys
    of the output dictionary conform to the Group/Subgroup directory branches of
    the original file.

    Parameters
    ----------
    fname : string (a hdf5 file path)

    Returns
    -------
    output : a dictionary that contains both data and metadata according to the
              original hdf5 file structure

    """
    f = h5py.File(fname, "r")
    fcontent = {}
    def filldict(x, y):
        # create a new container
        tmp = {}
        # add attributes if present
        if len(y.attrs) > 0:
            tmp['attrs'] = dict(y.attrs)
        # add data if it is a dataset
        if isinstance(y, h5py.Dataset):
            tmp['data'] = np.array(y)
        # only add to the dictionary, if we have something meaningful to add
        if tmp != {}:
            fcontent[x] = tmp
    f.visititems(filldict)

    f.close()

    return fcontent

def read_OPERA_hdf5(fname):
    """Reads hdf5 files according to OPERA conventions

    Please refer to the `OPERA data model documentation <http://www.knmi.nl/opera/opera3/OPERA_2008_03_WP2.1b_ODIM_H5_v2.1.pdf>`_
    in order to understand how an hdf5 file is organized that conforms to the OPERA
    ODIM_H5 conventions.

    In contrast to other file readers under wradlib.io, this function will *not* return
    a two item tuple with (data, metadata). Instead, this function returns ONE
    dictionary that contains all the file contents - both data and metadata. The keys
    of the output dictionary conform to the Group/Subgroup directory branches of
    the original file. If the end member of a branch (or path) is "data", then the
    corresponding item of output dictionary is a numpy array with actual data. Any other
    end member (either *how*, *where*, and *what*) will contain the meta information
    applying to the coresponding level of the file hierarchy.

    Parameters
    ----------
    fname : string (a hdf5 file path)

    Returns
    -------
    output : a dictionary that contains both data and metadata according to the
              original hdf5 file structure

    """
    f = h5py.File(fname, "r")
    # try verify OPERA conventions
##    if not f.keys() == ['dataset1', 'how', 'what', 'where']:
##        print "File is not organized according to OPERA conventions (ODIM_H5)..."
##        print "Expected the upper level subgroups to be: dataset1, how, what', where"
##        print "Try to use e.g. ViTables software in order to inspect the file hierarchy."
##        sys.exit(1)

    # now we browse through all Groups and Datasets and store the info in one dictionary
    fcontent = {}
    def filldict(x, y):
        if isinstance(y, h5py.Group):
            if len(y.attrs) > 0:
                fcontent[x] = dict(y.attrs)
        elif isinstance(y, h5py.Dataset):
            fcontent[x] = np.array(y)
    f.visititems(filldict)

    f.close()

    return fcontent


def read_gamic_scan_attributes(scan, scan_type):
    """Read attributes from one particular scan from a GAMIC hdf5 file

    Provided by courtesy of Kai Muehlbauer (University of Bonn).

    Parameters
    ----------
    scan : scan object from hdf5 file
    scan_type : string
        "PVOL" (plan position indicator) or "RHI" (range height indicator)

    Returns
    -------
    sattrs  : dictionary of scan attributes

    """

    global zero_index, el, az

    # placeholder for attributes
    sattrs = {}

    # link to scans 'how' hdf5 group
    sg1 = scan['how']

    # get scan attributes
    for attrname in list(sg1.attrs):
        sattrs[attrname] = sg1.attrs.get(attrname)
    sattrs['bin_range'] = sattrs['range_step'] * sattrs['range_samples']

    # get scan header
    ray_header = scan['ray_header']

    # az, el, zero_index for PPI scans
    if scan_type == 'PVOL':
        azi_start = ray_header['azimuth_start']
        azi_stop = ray_header['azimuth_stop']
        # Azimuth corresponding to 1st ray
        zero_index = np.where(azi_stop < azi_start)
        azi_stop[zero_index[0]] += 360
        zero_index = zero_index[0] + 1
        az = (azi_start+azi_stop)/2
        az = np.roll(az, -zero_index, axis=0)
        az = np.round(az, 1)
        el = sg1.attrs.get('elevation')

    # az, el, zero_index for RHI scans
    if scan_type == 'RHI':
        ele_start = np.round(ray_header['elevation_start'], 1)
        ele_stop = np.round(ray_header['elevation_stop'], 1)
        angle_step = np.round(sattrs['angle_step'], 1)
        angle_step = np.round(sattrs['ele_stop'], 1) / angle_step
        # Elevation corresponding to 1st ray
        if ele_start[0] < 0:
            ele_start = ele_start[1:]
            ele_stop = ele_stop[1:]
        zero_index = np.where(ele_stop > ele_start)
        zero_index = zero_index[0]  # - 1
        el = (ele_start+ele_stop)/2
        el = np.round(el, 1)
        el = el[-angle_step:]

        az = sg1.attrs.get('azimuth')

    # save zero_index (first ray) to scan attributes
    sattrs['zero_index'] = zero_index[0]

    # create range array
    r = np.arange(sattrs['bin_range'], sattrs['bin_range']*sattrs['bin_count']+sattrs['bin_range'], sattrs['bin_range'])

    # save variables to scan attributes
    sattrs['az'] = az
    sattrs['el'] = el
    sattrs['r'] = r
    sattrs['Time'] = sattrs.pop('timestamp')
    sattrs['max_range'] = r[-1]

    return sattrs


def read_gamic_scan(scan, scan_type, wanted_moments):
    """Read data from one particular scan from GAMIC hdf5 file

    Provided by courtesy of Kai Muehlbauer (University of Bonn).

    Parameters
    ----------
    scan : scan object from hdf5 file
    scan_type : string
        "PVOL" (plan position indicator) or "RHI" (range height indicator)
    wanted_moments  : sequence of strings containing upper case names of moment(s) to be returned

    Returns
    -------
    data : dictionary of moment data (numpy arrays)
    sattrs : dictionary of scan attributes

    """

    # placeholder for data and attrs
    data = {}
    sattrs = {}

    # try to read wanted moments
    for mom in list(scan):
        if 'moment' in mom:
            data1 = {}
            sg2 = scan[mom]
            actual_moment = sg2.attrs.get('moment').upper()
            if actual_moment in wanted_moments or wanted_moments == 'all':
                # read attributes only once
                if not sattrs:
                    sattrs = read_gamic_scan_attributes(scan, scan_type)
                mdata = sg2[...]
                dyn_range_max = sg2.attrs.get('dyn_range_max')
                dyn_range_min = sg2.attrs.get('dyn_range_min')
                bin_format = sg2.attrs.get('format')
                if bin_format == 'UV8':
                    div = 256.0
                else:
                    div = 65536.0
                mdata = dyn_range_min + mdata*(dyn_range_max-dyn_range_min)/div

                if scan_type == 'PVOL':
                    # rotate accordingly
                    mdata = np.roll(mdata, -1 * sattrs['zero_index'], axis=0)

                if scan_type == 'RHI':
                    # remove first zero angles
                    sdiff = mdata.shape[0]-sattrs['el'].shape[0]
                    mdata = mdata[sdiff:, :]

                data1['data'] = mdata
                data1['dyn_range_max'] = dyn_range_max
                data1['dyn_range_min'] = dyn_range_min
                data[actual_moment] = data1

    return data, sattrs


def read_GAMIC_hdf5(filename, wanted_elevations=None, wanted_moments=None):
    """Data reader for hdf5 files produced by the commercial GAMIC Enigma V3 MURAN software

    Provided by courtesy of Kai Muehlbauer (University of Bonn). See GAMIC
    homepage for further info (http://www.gamic.com/cgi-bin/info.pl?link=softwarebrowser3).

    Parameters
    ----------
    filename : path of the gamic hdf5 file
    scan_type : string
        "PVOL" (plan position indicator) or "RHI" (range height indicator)
    elevation_angle : sequence of strings of elevation_angle(s) of scan (only needed for PPI)
    moments : sequence of strings of moment name(s)

    Returns
    -------
    data : dictionary of scan and moment data (numpy arrays)
    attrs : dictionary of attributes

    """

    # check elevations
    if wanted_elevations is None:
        wanted_elevations = 'all'

    # check wanted_moments
    if wanted_moments is None:
        wanted_moments = 'all'

    # read the data from file
    f = h5py.File(filename, 'r')

    # placeholder for attributes and data
    attrs = {}
    vattrs = {}
    data = {}

    # check if GAMIC file and
    try:
        swver = f['how'].attrs.get('software')
    except KeyError:
        print("WRADLIB: File is no GAMIC hdf5!")
        raise

    # get scan_type (PVOL or RHI)
    scan_type = f['what'].attrs.get('object')

    # single or volume scan
    if scan_type == 'PVOL':
        # loop over 'main' hdf5 groups (how, scanX, what, where)
        for n in list(f):
            if 'scan' in n:
                g = f[n]
                sg1 = g['how']

                # get scan elevation
                el = sg1.attrs.get('elevation')
                el = str(round(el, 2))

                # try to read scan data and attrs if wanted_elevations are found
                if (el in wanted_elevations) or (wanted_elevations == 'all'):
                    sdata, sattrs = read_gamic_scan(scan=g, scan_type=scan_type,
                                                    wanted_moments=wanted_moments)
                    if sdata:
                        data[n.upper()] = sdata
                    if sattrs:
                        attrs[n.upper()] = sattrs

    # single rhi scan
    elif scan_type == 'RHI':
        # loop over 'main' hdf5 groups (how, scanX, what, where)
        for n in list(f):
            if 'scan' in n:
                g = f[n]
                # try to read scan data and attrs
                sdata, sattrs = read_gamic_scan(scan=g, scan_type=scan_type,
                                                wanted_moments=wanted_moments)
                if sdata:
                    data[n.upper()] = sdata
                if sattrs:
                    attrs[n.upper()] = sattrs

    # collect volume attributes if wanted data is available
    if data:
        vattrs['Latitude'] = f['where'].attrs.get('lat')
        vattrs['Longitude'] = f['where'].attrs.get('lon')
        vattrs['Height'] = f['where'].attrs.get('height')
        # check whether its useful to implement that feature
        #vattrs['sitecoords'] = (vattrs['Latitude'], vattrs['Longitude'], vattrs['Height'])
        attrs['VOL'] = vattrs

    f.close()

    return data, attrs


def to_pickle(fpath, obj):
    """Pickle object <obj> to file <fpath>
    """
    output = open(fpath, 'wb')
    pickle.dump(obj, output)
    output.close()


def from_pickle(fpath):
    """Return pickled object from file <fpath>
    """
    pkl_file = open(fpath, 'rb')
    obj = pickle.load(pkl_file)
    pkl_file.close()
    return obj


def to_hdf5(fpath, data, mode="w", metadata=None, dataset="data", compression="gzip"):
    """Quick storage of one <data> array and a <metadata> dict in an hdf5 file

    This is more efficient than pickle, cPickle or numpy.save. The data is stored in
    a subgroup named ``data`` (i.e. hdf5file["data").

    Parameters
    ----------
    fpath : string (path to the hdf5 file)
    data : numpy array
    mode : string, file open mode, defaults to "w" (create, truncate if exists)
    metadata : dictionary of data's attributes
    dataset : string describing dataset
    compression : h5py compression type {"gzip"|"szip"|"lzf"}, see h5py documentation for details

    """
    f = h5py.File(fpath, mode=mode)
    dset = f.create_dataset(dataset, data=data, compression=compression)
    # store metadata
    if metadata:
        for key in metadata.keys():
            dset.attrs[key] = metadata[key]
    # close hdf5 file
    f.close()


def from_hdf5(fpath, dataset="data"):
    """Loading data from hdf5 files that was stored by <wradlib.io.to_hdf5>

    Parameters
    ----------
    fpath : string (path to the hdf5 file)
    dataset : name of the Dataset in which the data is stored

    """
    f = h5py.File(fpath, mode="r")
    # Check whether Dataset exists
    if not dataset in f.keys():
        print("Cannot read Dataset <%s> from hdf5 file <%s>" % (dataset, f))
        f.close()
        sys.exit()
    data = np.array(f[dataset][:])
    # get metadata
    metadata = {}
    for key in f[dataset].attrs.keys():
        metadata[key] = f[dataset].attrs[key]
    f.close()
    return data, metadata


def read_safnwc(filename):
    """Read MSG SAFNWC hdf5 file into a gdal georeferenced object
    
    Parameters
    ----------
    filename : satellite file name

    Returns
    -------
    ds : gdal dataset with satellite data

    """

    root = gdal.Open(filename)
    ds = gdal.Open('HDF5:'+filename+'://CT')
    name = os.path.basename(filename)[7:11]
    try:
        proj = root.GetMetadata()["PROJECTION"];
    except Exception as error:
        raise NameError("No metadata for satellite file %s" %(filename))
    geotransform = root.GetMetadata()["GEOTRANSFORM_GDAL_TABLE"].split(",");
    geotransform[0] = root.GetMetadata()["XGEO_UP_LEFT"];
    geotransform[3] = root.GetMetadata()["YGEO_UP_LEFT"];
    ds.SetProjection(proj)
    ds.SetGeoTransform([float(x) for x in geotransform])
    return(ds)


if __name__ == '__main__':
    print 'wradlib: Calling module <io> as main...'
