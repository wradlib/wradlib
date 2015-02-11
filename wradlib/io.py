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
   read_generic_hdf5
   read_generic_netcdf
   read_OPERA_hdf5
   read_GAMIC_hdf5
   read_RADOLAN_composite
   read_Rainbow

"""

# standard libraries

import sys
import datetime as dt
import cPickle as pickle
import StringIO
from collections import OrderedDict

import re
import pytz
import os
import warnings


# site packages
import h5py
import numpy as np
import netCDF4 as nc  # ATTENTION: Needs to be imported AFTER h5py, otherwise ungraceful crash
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
    for this, next in zip(flagged[:-1], flagged[1:]):
        # create as many zeros as there are given within the flagged
        # byte's data part
        beam.extend([0] * (raw[this] & data))
        # append the data until the next flag
        beam.extend(raw[this + 1:next])

    # process the last flag
    # add zeroes
    beam.extend([0] * (raw[flagged[-1]] & data))

    # add remaining data
    beam.extend(raw[flagged[-1] + 1:])

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
    out["datetime"] = dt.datetime.strptime(header[2:8] + header[13:17] + "00",
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

    out['bytes'] = int(header[pos_BY + 2:pos_BY + 7])
    out['version'] = header[pos_VS + 2:pos_VS + 4]
    out['cluttermap'] = int(header[pos_CO + 2:pos_CO + 3])
    out['dopplerfilter'] = int(header[pos_CD + 2:pos_CD + 3])
    out['statfilter'] = int(header[pos_CS + 2:pos_CS + 3])
    out['elevprofile'] = [float(header[pos_EP + 2 + 3 * i:pos_EP + 2 + 3 * (i + 1)]) for i in range(8)]
    out['message'] = header[pos_MS + 5:pos_MS + 5 + int(header[pos_MS + 2:pos_MS + 5])]

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

    azimuthbitmask = 2 ** (14 - 1)
    databitmask = 2 ** (13 - 1) - 1
    clutterflag = 2 ** 15
    dataflag = 2 ** 13 - 1
    # open the DX file in binary mode for reading
    if type(filename) == file:
        f = filename
    else:
        f = open(filename, 'rb')

    # header string for later processing
    header = ''
    atend = False
    # read header
    while True:
        mychar = f.read(1)
        # 0x03 signals the end of the header but sometimes there might be
        # an additional 0x03 char after that
        if mychar == chr(3):
            atend = True
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
    if (buflen % 2) != 0:
        # make sure that this is consistent with our assumption
        # i.e. contact DWD again, if DX files show up with uneven byte lengths
        # *and* only one 0x03 character
        #assert header[-2] == chr(3)
        buflen -= 1

    buf = f.read(buflen)
    # we can interpret the rest directly as a 1-D array of 16 bit unsigned ints
    raw = np.frombuffer(buf, dtype='uint16')

    # reading finished, close file, but only if we opened it.
    if type(filename) != file:
        f.close()

    # a new ray/beam starts with bit 14 set
    # careful! where always returns its results in a tuple, so in order to get
    # the indices we have to retrieve element 0 of this tuple
    newazimuths = np.where(raw == azimuthbitmask)[0]  # Thomas kontaktieren!!!!!!!!!!!!!!!!!!!

    # for the following calculations it is necessary to have the end of the data
    # as the last index
    newazimuths = np.append(newazimuths, len(raw))

    # initialize our list of rays/beams
    beams = []
    # initialize our list of elevations
    elevs = []
    # initialize our list of azimuths
    azims = []

    # iterate over all beams
    for i in range(newazimuths.size - 1):
        # unpack zeros
        beam = unpackDX(raw[newazimuths[i] + 3:newazimuths[i + 1]])
        beams.append(beam)
        elevs.append((raw[newazimuths[i] + 2] & databitmask) / 10.)
        azims.append((raw[newazimuths[i] + 1] & databitmask) / 10.)

    beams = np.array(beams)

    #attrs =  {}
    attrs['elev'] = np.array(elevs)
    attrs['azim'] = np.array(azims)
    attrs['clutter'] = (beams & clutterflag) != 0

    # converting the DWD rvp6-format into dBZ data and return as numpy array together with attributes
    return (beams & dataflag) * 0.5 - 32.5, attrs


def _write_polygon2txt(f, idx, vertices):
    f.write('%i %i\n' % idx)
    for i, vert in enumerate(vertices):
        f.write('%i ' % (i,))
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
            az = np.linspace(np.round(az[ix_minaz], 2), np.round(az[ix_maxaz], 2), len(az))
        else:
            az = np.roll(az, -ix_minaz)
        # rotate accordingly
        data = np.roll(data, -ix_minaz, axis=0)
        data = np.where(data == dset.getncattr('MissingData'), np.nan, data)
        # Ranges
        binwidth = (dset.getncattr('MaximumRange-value') * 1000.) / len(dset.dimensions['Gate'])
        r = np.arange(binwidth, (dset.getncattr('MaximumRange-value') * 1000.) + binwidth, binwidth)
        # collect attributes
        attrs = {}
        for attrname in dset.ncattrs():
            attrs[attrname] = dset.getncattr(attrname)
        ##        # Limiting the returned range
        ##        if range_lim and range_lim / binwidth <= data.shape[1]:
        ##            data = data[:,:range_lim / binwidth]
        ##            r = r[:range_lim / binwidth]
        # Set additional metadata attributes
        attrs['az'] = az
        attrs['r'] = r
        attrs['sitecoords'] = (attrs['Longitude'], attrs['Latitude'], attrs['Height'])
        attrs['time'] = dt.datetime.utcfromtimestamp(attrs.pop('Time'))
        attrs['max_range'] = data.shape[1] * binwidth
    except:
        raise
    finally:
        dset.close()

    return data, attrs


def get_radolan_header_token():
    """Return array with known header token of radolan composites

    Returns
    -------
    head:   dict with known header token, value set to None
    """
    head = {'BY': None, 'VS': None, 'SW': None, 'PR': None,
            'INT': None, 'GP': None, 'MS': None, 'LV': None,
            'CS': None, 'MX': None, 'BG': None}
    return head


def get_radolan_header_token_pos(header):
    """Get Token and positions from DWD radolan header

    Parameters
    ----------
    header: string (ASCII header)

    Returns
    -------
    head: dictionary with found header tokens and positions

    """

    head_dict = get_radolan_header_token()

    for token in head_dict.keys():
        d = header.rfind(token)
        if d > -1:
            head_dict[token] = d

    head = {}

    for k, v in head_dict.iteritems():
        if v is not None:
            start = v + len(k)
            filt = filter(lambda x: x > v, head_dict.values())
            if filt:
                stop = min(filt)
            else:
                stop = None
            head[k] = (start, stop)
        else:
            head[k] = v

    return head


def parse_DWD_quant_composite_header(header):
    """Parses the ASCII header of a DWD quantitative composite file

    Parameters
    ----------
    header : string (ASCII header)

    Returns
    -------
    output : dictionary of metadata retrieved from file header

    """
    # empty container
    out = {}

    # RADOLAN product type def
    out["producttype"] = header[0:2]
    # file time stamp as Python datetime object
    out["datetime"] = dt.datetime.strptime(header[2:8] + header[13:17] + "00",
                                           "%d%H%M%m%y%S")
    # radar location ID (always 10000 for composites)
    out["radarid"] = header[8:13]

    # get dict of header token with positions
    head = get_radolan_header_token_pos(header)

    # iterate over token and fill output dict accordingly
    for k, v in head.iteritems():
        if v:
            if k == 'BY':
                out['datasize'] = int(header[v[0]:v[1]]) - len(header) - 1
            if k == 'VS':
                out["maxrange"] = {0: "100 km and 128 km (mixed)",
                                   1: "100 km",
                                   2: "128 km",
                                   3: "150 km"}.get(int(header[v[0]:v[1]]), "100 km")
            if k == 'SW':
                out["radolanversion"] = header[v[0]:v[1]].strip()
            if k == 'PR':
                out["precision"] = float('1' + header[v[0]:v[1]].strip())
            if k == 'INT':
                out["intervalseconds"] = int(header[v[0]:v[1]]) * 60
            if k == 'GP':
                dimstrings = header[v[0]:v[1]].strip().split("x")
                out["nrow"] = int(dimstrings[0])
                out["ncol"] = int(dimstrings[1])
            if k == 'BG':
                dimstrings = header[v[0]:v[1]]
                dimstrings = dimstrings[:len(dimstrings) / 2], dimstrings[len(dimstrings) / 2:]
                out["nrow"] = int(dimstrings[0])
                out["ncol"] = int(dimstrings[1])
            if k == 'LV':
                lv = header[v[0]:v[1]].split()
                out['nlevel'] = np.int(lv[0])
                out['level'] = np.array(lv[1:]).astype('float')
            if k == 'MS':
                locationstring = header[v[0]:].strip().split("<")[1].split(">")[0]
                out["radarlocations"] = locationstring.split(",")
            if k == 'CS':
                out['indicator'] = {0: "near ground level",
                                    1: "maximum",
                                    2: "tops"}.get(int(header[v[0]:v[1]]))
            if k == 'MX':
                out['imagecount'] = int(header[v[0]:v[1]])
    return out


def decode_radolan_runlength_line(line, attrs):
    """Decodes one line of runlength coded binary data of DWD
    composite file and returns decoded array

    Parameters
    ----------
    line: numpy array of byte values

    Returns
    -------
    arr:  numpy array of decoded values
    """
    # byte '0' is line number, we don't need it
    # so we start with offset byte,
    lo = 1
    byte = line[lo]
    # line empty condition, lf directly behind line number
    if byte == 10:
        return np.ones(attrs['ncol'], dtype=np.uint8) * attrs['nodataflag']
    offset = byte - 16

    # check if offset byte is 255 and take next byte(s)
    # also for the offset
    while byte == 255:
        lo += 1
        byte = line[lo]
        offset += byte - 16

    # just take the rest
    dline = line[lo + 1:]

    # this could be optimized
    # iterate over line string, until lf (10) is reached
    for lo, byte in enumerate(dline):
        if byte == 10:
            break
        width = (byte & 0xF0) >> 4
        val = byte & 0x0F
        # the "offset pixel" are "not measured" values
        # so we set them to 'nodata'
        if lo == 0:
            arr = np.ones(offset, dtype=np.uint8) * attrs['nodataflag']
        arr = np.append(arr, np.ones(width, dtype=np.uint8) * val)

    trailing = attrs['ncol'] - len(arr)
    if trailing > 0:
        arr = np.append(arr, np.ones(trailing, dtype=np.uint8) * attrs['nodataflag'])
    elif trailing < 0:
        arr = dline[:trailing]

    return arr


def read_radolan_runlength_line(fid):
    """Reads one line of runlength coded binary data of DWD
    composite file and returns it as numpy array

    Parameters
    ----------
    fid: file/buffer id

    Returns
    -------
    line:  numpy array of coded values
    """
    line = fid.readline()

    # check if eot
    if line == '\x04':
        return None

    # convert input buffer to np.uint8 array
    line = np.frombuffer(line, np.uint8).astype(np.uint8)

    return line


def decode_radolan_runlength_array(binarr, attrs):
    """Decodes the binary runlength coded section from DWD composite
    file and return decoded numpy array with correct shape

    Parameters
    ----------
    binarr:    string Buffer
    attrs:  Attribute dict of file header

    Returns
    -------
    arr: numpy array of decoded values
    """
    buf = StringIO.StringIO(binarr)

    # read and decode first line
    line = read_radolan_runlength_line(buf)
    arr = decode_radolan_runlength_line(line, attrs)

    # read following lines
    line = read_radolan_runlength_line(buf)

    while line is not None:
        dline = decode_radolan_runlength_line(line, attrs)
        arr = np.vstack((arr, dline))
        line = read_radolan_runlength_line(buf)
    # return upside down because first line read is top line
    return np.flipud(arr)


def read_radolan_binary_array(fid, size):
    """Read binary data from file given by filehandle

    Parameters
    ----------
    fid: file handle
    size: number of bytes to read

    Returns
    -------
    binarr: string array of binary data
    """
    binarr = fid.read(size)
    fid.close()
    if len(binarr) != size:
        raise IOError('{0}: File corruption while reading {1}! '
                      '\nCould not read enough data!'.format(__name__, fid.name))
    return binarr


def get_radolan_filehandle(fname):
    """Opens radolan file and returns file handle

    Parameters
    ----------
    fname: filename

    Returns
    -------
    f: filehandle
    """

    gzip = util.import_optional('gzip')

    # open file handle
    try:
        f = gzip.open(fname, 'rb')
        f.read(1)
    except IOError:
        f = open(fname, 'rb')
        f.read(1)

    # rewind file
    f.seek(0, 0)

    return f


def read_radolan_header(fid):
    """Reads radolan ASCII header and returns it as string

    Parameters
    ----------
    fid: file handle

    Returns
    -------
    header: string
    """
    # rewind, just in case...
    fid.seek(0, 0)

    header = ''
    while True:
        mychar = fid.read(1)
        if mychar == chr(3):
            break
        header = header + mychar

    return header


def read_RADOLAN_composite(fname, missing=-9999, loaddata=True):
    """Read quantitative radar composite format of the German Weather Service

    The quantitative composite format of the DWD (German Weather Service) was
    established in the course of the `RADOLAN project <http://www.dwd.de/RADOLAN>`
    and includes several file types, e.g. RX, RO, RK, RZ, RP, RT, RC, RI, RG, PC,
    PG and many, many more.
    (see format description on the RADOLAN project homepage :cite:`DWD2009`).

    At the moment, the national RADOLAN composite is a 900 x 900 grid with 1 km
    resolution and in polar-stereographic projection. There are other grid resolutions
    for different composites (eg. PC, PG)

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

    """

    NODATA = missing
    mask = 0xFFF  # max value integer

    f = get_radolan_filehandle(fname)

    header = read_radolan_header(f)

    attrs = parse_DWD_quant_composite_header(header)

    if not loaddata:
        f.close()
        return None, attrs

    attrs["nodataflag"] = NODATA

    if not attrs["radarid"] == "10000":
        warnings.warn("WARNING: You are using function e" +
                      "wradlib.io.read_RADOLAN_composit for a non " +
                      "composite file.\n " +
                      "This might work...but please check the validity " +
                      "of the results")

    # read the actual data
    indat = read_radolan_binary_array(f, attrs['datasize'])

    if attrs["producttype"] in ["RX", "EX"]:
        #convert to 8bit integer
        arr = np.frombuffer(indat, np.uint8).astype(np.uint8)
        arr = np.where(arr == 250, NODATA, arr)
        attrs['cluttermask'] = np.where(arr == 249)[0]

    elif attrs['producttype'] in ["PG", "PC"]:
        arr = decode_radolan_runlength_array(indat, attrs)
    else:
        # convert to 16-bit integers
        arr = np.frombuffer(indat, np.uint16).astype(np.uint16)
        # evaluate bits 13, 14, 15 and 16
        attrs['secondary'] = np.where(arr & 0x1000)[0]
        nodata = np.where(arr & 0x2000)[0]
        negative = np.where(arr & 0x4000)[0]
        attrs['cluttermask'] = np.where(arr & 0x8000)[0]
        # mask out the last 4 bits
        arr = arr & mask
        # consider negative flag if product is RD (differences from adjustment)
        if attrs["producttype"] == "RD":
            # NOT TESTED, YET
            arr[negative] = -arr[negative]
        # apply precision factor
        # this promotes arr to float if precision is float
        arr = arr * attrs["precision"]
        # set nodata value
        arr[nodata] = NODATA

    # anyway, bring it into right shape
    arr = arr.reshape((attrs["nrow"], attrs["ncol"]))

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

    Please refer to the `OPERA data model documentation
    <https://www.eol.ucar.edu/system/files/OPERA_2008_03_WP2.1b_ODIM_H5_v2.1.pdf>`_
    in order to understand how an hdf5 file is organized that conforms to the OPERA
    ODIM_H5 conventions.

    In contrast to other file readers under wradlib.io, this function will *not* return
    a two item tuple with (data, metadata). Instead, this function returns ONE
    dictionary that contains all the file contents - both data and metadata. The keys
    of the output dictionary conform to the Group/Subgroup directory branches of
    the original file. If the end member of a branch (or path) is "data", then the
    corresponding item of output dictionary is a numpy array with actual data. Any other
    end member (either *how*, *where*, and *what*) will contain the meta information
    applying to the corresponding level of the file hierarchy.

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
        az = (azi_start + azi_stop) / 2
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
        el = (ele_start + ele_stop) / 2
        el = np.round(el, 1)
        el = el[-angle_step:]

        az = sg1.attrs.get('azimuth')

    # save zero_index (first ray) to scan attributes
    sattrs['zero_index'] = zero_index[0]

    # create range array
    r = np.arange(sattrs['bin_range'], sattrs['bin_range'] * sattrs['bin_count'] + sattrs['bin_range'],
                  sattrs['bin_range'])

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
                mdata = dyn_range_min + mdata * (dyn_range_max - dyn_range_min) / div

                if scan_type == 'PVOL':
                    # rotate accordingly
                    mdata = np.roll(mdata, -1 * sattrs['zero_index'], axis=0)

                if scan_type == 'RHI':
                    # remove first zero angles
                    sdiff = mdata.shape[0] - sattrs['el'].shape[0]
                    mdata = mdata[sdiff:, :]

                data1['data'] = mdata
                data1['dyn_range_max'] = dyn_range_max
                data1['dyn_range_min'] = dyn_range_min
                data[actual_moment] = data1

    return data, sattrs


def read_GAMIC_hdf5(filename, wanted_elevations=None, wanted_moments=None):
    """Data reader for hdf5 files produced by the commercial GAMIC Enigma V3 MURAN software

    Provided by courtesy of Kai Muehlbauer (University of Bonn). See GAMIC
    homepage for further info (http://www.gamic.com).

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
        #vattrs['sitecoords'] = (vattrs['Longitude'], vattrs['Latitude'], vattrs['Height'])
        attrs['VOL'] = vattrs

    f.close()

    return data, attrs


def find_key(key, dictionary):
    """Searches for given key in given (nested) dictionary.

    Returns all found parent dictionaries in a list.

    Parameters
    ----------
    key : string
        the key to be searched for in the nested dict
    dictionary : dict
        the dictionary to be searched

    Returns
    -------
    output : a dictionary or list of dictionaries

    """
    for k, v in dictionary.iteritems():
        if k == key:
            yield dictionary
        elif isinstance(v, dict):
            for result in find_key(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find_key(key, d):
                    yield result


def decompress(data):
    """Decompression of data

    Parameters
    ----------
    data : string (from xml)
        data string containing compressed data.
    """
    zlib = util.import_optional('zlib')
    return zlib.decompress(data)


def get_RB_data_layout(datadepth):
    """Calculates DataWidth and DataType from given DataDepth of RAINBOW radar data

    Parameters
    ----------
    datadepth : int
        DataDepth as read from the Rainbow xml metadata.

    Returns
    -------
    datawidth : int
        Width in Byte of data

    datatype : string
        conversion string .

    """

    if sys.byteorder != 'big':
        byteorder = '>'
    else:
        byteorder = '<'

    datawidth = datadepth / 8

    if datawidth in [1, 2, 4]:
        datatype = byteorder + 'u' + str(datawidth)
    else:
        raise ValueError("Wrong DataDepth: %d. Conversion only for depth 8, 16, 32" % datadepth)

    return datawidth, datatype


def get_RB_data_attribute(xmldict, attr):
    """Get Attribute `attr` from dict `xmldict`

    Parameters
    ----------
    xmldict : dict
        Blob Description Dictionary

    attr : string
        Attribute key

    Returns
    -------
    sattr : int
        Attribute Values

    """

    try:
        sattr = int(xmldict['@' + attr])
    except KeyError:
        if attr == 'bins':
            sattr = None
        else:
            raise KeyError('Attribute @' + attr + ' is missing from Blob Description'
                                                  'There may be some problems with your file')
    return sattr


def get_RB_blob_attribute(blobdict, attr):
    """Get Attribute `attr` from dict `blobdict`

    Parameters
    ----------
    blobdict : dict
        Blob Description Dictionary

    attr : string
        Attribute key

    Returns
    -------
        Attribute Value

    """
    try:
        value = blobdict['BLOB']['@' + attr]
    except KeyError:
        raise KeyError('Attribute @' + attr + ' is missing from Blob.' +
                       'There may be some problems with your file')

    return value


def get_RB_blob_data(datastring, blobid):
    """ Read BLOB data from datastring and return it

    Parameters
    ----------
    datastring : dict
        Blob Description Dictionary

    blobid : int
        Number of requested blob

    Returns
    -------
    data : string
        Content of blob

    """
    xmltodict = util.import_optional('xmltodict')

    start = 0
    searchString = r'<BLOB blobid="{}"'.format(blobid)
    start = datastring.find(searchString, start)
    if start == -1:
        raise EOFError('Blob ID {} not found!'.format(blobid))
    end = datastring.find('>', start)
    xmlstring = datastring[start:end + 1]

    # cheat the xml parser by making xml well-known
    xmldict = xmltodict.parse(xmlstring + '</BLOB>')
    cmpr = get_RB_blob_attribute(xmldict, 'compression')
    size = int(get_RB_blob_attribute(xmldict, 'size'))
    data = datastring[end + 2:end + 2 + size]  # read blob data to string

    # decompress if necessary
    # the first 4 bytes are neglected for an unknown reason
    if cmpr == "qt":
        data = decompress(data[4:])

    return data


def map_RB_data(data, datadepth):
    """ Map BLOB data to correct DataWidth and Type and convert it to numpy array

    Parameters
    ----------
    data : string
        Blob Data

    datadepth : int
        bit depth of Blob data

    Returns
    -------
    data : numpy array
        Content of blob

    """
    datawidth, datatype = get_RB_data_layout(datadepth)

    # import from data buffer well aligned to data array
    data = np.ndarray(shape=(len(data) / datawidth,), dtype=datatype, buffer=data)

    return data


def get_RB_blob_from_string(datastring, blobdict):
    """
    Read BLOB data from datastring and return it as numpy array with correct
    dataWidth and shape

    Parameters
    ----------
    datastring : dict
        Blob Description Dictionary

    blobdict : dict
        Blob Dict

    Returns
    -------
    data : numpy array
        Content of blob as numpy array

    """

    blobid = get_RB_data_attribute(blobdict, 'blobid')
    data = get_RB_blob_data(datastring, blobid)

    # map data to correct datatype and width
    datadepth = get_RB_data_attribute(blobdict, 'depth')
    data = map_RB_data(data, datadepth)

    # reshape data
    bins = get_RB_data_attribute(blobdict, 'bins')
    if bins:
        rays = get_RB_data_attribute(blobdict, 'rays')
        data.shape = (rays, bins)

    return data


def get_RB_blob_from_file(filename, blobdict):
    """
    Read BLOB data from file and return it with correct
    dataWidth and shape

    Parameters
    ----------
    filename : string
        Filename of Data File

    blobdict : dict
        Blob Dict

    Returns
    -------
    data : numpy array
        Content of blob as numpy array

    """
    try:
        fid = open(filename, "rb")
    except IOError:
        print "WRADLIB: Error opening Rainbow file ", filename
        raise IOError

    datastring = fid.read()
    fid.close()

    data = get_RB_blob_from_string(datastring, blobdict)

    return data


def get_RB_file_as_string(filename):
    """ Read Rainbow File Contents in dataString

    Parameters
    ----------
    filename : string
        Filename of Data File

    Returns
    -------
    dataString : string
        File Contents as dataString

    """
    try:
        fid = open(filename, "rb")
    except IOError:
        print "WRADLIB: Error opening Rainbow file ", filename
        raise IOError

    dataString = fid.read()
    fid.close()

    return dataString


def get_RB_blobs_from_file(filename, rbdict):
    """Read all BLOBS found in given nested dict, loads them from file
    given by filename and add them to the dict at the appropriate position.

    Parameters
    ----------
    :param filename: string
        Filename of Data File
    :param rbdict: dict
        Rainbow file Contents

    Returns
    -------
    :rtype : dict
        Rainbow File Contents

    """

    blobs = list(find_key('@blobid', rbdict))

    datastring = get_RB_file_as_string(filename)
    for blob in blobs:
        data = get_RB_blob_from_string(datastring, blob)
        blob['data'] = data

    return rbdict


def get_RB_header(filename):
    """Read Rainbow Header from filename, converts it to a dict and returns it

    Parameters
    ----------
    filename : string
        Filename of Data File

    Returns
    -------
    object : dictionary
        Rainbow File Contents

    """
    try:
        fid = open(filename, "rb")
    except IOError:
        print "WRADLIB: Error opening Rainbow file ", filename
        raise IOError

    # load the header lines, i.e. the XML part
    endXMLmarker = "<!-- END XML -->"
    header = ""
    line = ""
    while not line.startswith(endXMLmarker):
        header += line[:-1]
        line = fid.readline()
        if len(line) == 0:
            break

    fid.close()

    xmltodict = util.import_optional('xmltodict')

    return xmltodict.parse(header)


def read_Rainbow(filename, loaddata=True):
    """"Reads Rainbow files files according to their structure

    In contrast to other file readers under wradlib.io, this function will *not* return
    a two item tuple with (data, metadata). Instead, this function returns ONE
    dictionary that contains all the file contents - both data and metadata. The keys
    of the output dictionary conform to the XML outline in the original data file.

    The radar data will be extracted from the data blobs, converted and added to the
    dict with key 'data' at the place where the @blobid was pointing from.

    Parameters
    ----------
    filename : string (a rainbow file path)

    Returns
    -------
    rbdict : a dictionary that contains both data and metadata according to the
              original rainbow file structure
    """

    rbdict = get_RB_header(filename)

    if loaddata:
        rbdict = get_RB_blobs_from_file(filename, rbdict)

    return rbdict


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
    ds = gdal.Open('HDF5:' + filename + '://CT')
    name = os.path.basename(filename)[7:11]
    try:
        proj = root.GetMetadata()["PROJECTION"]
    except Exception as error:
        raise NameError("No metadata for satellite file %s" % (filename))
    geotransform = root.GetMetadata()["GEOTRANSFORM_GDAL_TABLE"].split(",")
    geotransform[0] = root.GetMetadata()["XGEO_UP_LEFT"]
    geotransform[3] = root.GetMetadata()["YGEO_UP_LEFT"]
    ds.SetProjection(proj)
    ds.SetGeoTransform([float(x) for x in geotransform])
    return (ds)


def read_generic_netcdf(fname):
    """Reads netcdf files and returns a dictionary with corresponding structure.

    In contrast to other file readers under wradlib.io, this function will *not* return
    a two item tuple with (data, metadata). Instead, this function returns ONE
    dictionary that contains all the file contents - both data and metadata. The keys
    of the output dictionary conform to the Group/Subgroup directory branches of
    the original file.

    Please see the examples below on how to browse through a return object. The
    most important keys are the "dimensions" which define the shape of the data
    arrays, and the "variables" which contain the actual data and typically also
    the data that define the dimensions (e.g. sweeps, azimuths, ranges). These keys
    should be present in any netcdf file.

    Notes
    -----
    The returned dictionary could be quite big, depending on the content of the file.

    Parameters
    ----------
    fname : string (a netcdf file path)

    Returns
    -------
    out : an ordered dictionary that contains both data and metadata according to the
              original netcdf file structure

    Examples
    --------
    See :download:`generic_netcdf_example.py script <../../../examples/generic_netcdf_example.py>`.

    .. literalinclude:: ../../../examples/generic_netcdf_example.py


    """
    try:
        ncid = nc.Dataset(fname, 'r')
    except:
        print("wradlib: Could not read " % fname)
        print("Check whether file exists, and whether it is a netCDF file.")
        print("Raising exception...")
        raise

    if ncid.groups:
        # To be implemented if necessary, all netcdf files
        # I got my hands on have just one group/Dataset
        print("Groups", ncid.groups)


    out = OrderedDict()

    # get file format (should be NETCDF3 or NETCDF4)
    try:
        out["file_format"] = ncid.file_format
    except:
        pass

    # global attributes
    for k, v in ncid.__dict__.iteritems():
        out[k] = v

    # dimensions
    dimids = np.array([])
    if ncid.dimensions:
        dim = OrderedDict()
        for k, v in ncid.dimensions.iteritems():
            tmp = OrderedDict()
            try:
                tmp['data_model'] =  v._data_model
            except:
                pass
            try:
                tmp['size'] = v.__len__()
            except:
                pass
            tmp['dimid'] = v._dimid
            dimids = np.append(dimids,v._dimid)
            tmp['grpid'] = v._grpid
            tmp['isunlimited'] = v.isunlimited()
            dim[k] = tmp
        # Usually, the dimensions should be ordered by dimid automatically in case netcdf used OrderedDict
        # However, we should double check
        if np.array_equal(dimids, np.sort(dimids)):
            # is already sorted
            out['dimensions'] = dim
        else:
            # need to sort
            dim2 = OrderedDict()
            keys = dim.keys()
            for dimid in np.sort(dimids):
                dim2[keys[dimid]] = dim[keys[dimid]]
            out["dimensions"] = dim2


    # variables
    if ncid.variables:
        var = OrderedDict()
        for k, v in ncid.variables.iteritems():
            if isinstance(v.__dict__, dict):
                tmp = OrderedDict()
                for k1, v1 in v.__dict__.iteritems():
                    tmp[k1] = v1
                if v[:].dtype == 'S1':
                    tmp['data'] = v[:].compressed().tostring()
                else:
                    tmp['data'] = v[:]
                var[k] = tmp
            else:
                var[k] = v
        out['variables'] = var

    ncid.close()

    return out


if __name__ == '__main__':
    print 'wradlib: Calling module <io> as main...'
