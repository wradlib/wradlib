#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Raw Data I/O
^^^^^^^^^^^^

Please have a look at the tutorial
:ref:`notebooks/fileio/wradlib_radar_formats.ipynb`
for an introduction on how to deal with different file formats.

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
   read_safnwc
   write_raster_dataset
   to_AAIGrid
   to_GeoTIFF
   to_hdf5
   from_hdf5
   read_raster_data
   open_shape

"""

# standard libraries
from __future__ import absolute_import
import sys
import datetime as dt

try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    from StringIO import StringIO
    import io
except ImportError:
    from io import StringIO  # noqa
    import io

# from builtins import bytes, chr
from collections import OrderedDict
import re
import os
import warnings

# site packages
import h5py
import numpy as np
# ATTENTION: Needs to be imported AFTER h5py, otherwise ungraceful crash
import netCDF4 as nc
from osgeo import gdal, ogr, osr
from . import util as util
from . import georef as georef

# current DWD file naming pattern (2008) for example:
# raa00-dx_10488-200608050000-drs---bin
dwdpattern = re.compile('raa..-(..)[_-]([0-9]{5})-([0-9]*)-(.*?)---bin')


def _getTimestampFromFilename(filename):
    """Helper function doing the actual work of getDXTimestamp"""
    time = dwdpattern.search(filename).group(3)
    if len(time) == 10:
        time = '20' + time
    return dt.datetime.strptime(time, '%Y%m%d%H%M')


def getDXTimestamp(name):
    """Converts a dx-timestamp (as part of a dx-product filename) to a
    python datetime.object.

    Parameters
    ----------
    name : string
        representing a DWD product name
    tz : timezone object
        (see pytz package or datetime module for explanation)
        in case the timezone of the data is not UTC
    opt : currently unused

    Returns
    -------
    time : timezone-aware datetime.datetime object
    """
    return _getTimestampFromFilename(name).replace(tzinfo=util.UTC())


def unpackDX(raw):
    """function removes DWD-DX-product bit-13 zero packing"""
    # data is encoded in the first 12 bits
    data = 4095
    # the zero compression flag is bit 13
    flag = 4096

    beam = []

    # # naive version
    # # 49193 function calls in 0.772 CPU seconds
    # # 20234 function calls in 0.581 CPU seconds
    # for item in raw:
    #     if item & flag:
    #         beam.extend([0]* (item & data))
    #     else:
    #         beam.append(item & data)

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
    for this, nxt in zip(flagged[:-1], flagged[1:]):
        # create as many zeros as there are given within the flagged
        # byte's data part
        beam.extend([0] * (raw[this] & data))
        # append the data until the next flag
        beam.extend(raw[this + 1:nxt])

    # process the last flag
    # add zeroes
    beam.extend([0] * (raw[flagged[-1]] & data))

    # add remaining data
    beam.extend(raw[flagged[-1] + 1:])

    # return the data
    return np.array(beam)


def parse_DX_header(header):
    """Internal function to retrieve and interpret the ASCII header of a DWD
    DX product file.

    Parameters
    ----------
    header : string
        string representation of DX header
    """
    # empty container
    out = {}
    # RADOLAN product type def
    out["producttype"] = header[0:2]
    # time stamp from file header as Python datetime object
    out["datetime"] = dt.datetime.strptime(header[2:8] + header[13:17] + "00",
                                           "%d%H%M%m%y%S")
    # Make it aware of its time zone (UTC)
    out["datetime"] = out["datetime"].replace(tzinfo=util.UTC())
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
    out['elevprofile'] = [float(header[pos_EP + 2 + 3 * i:pos_EP + 2 + 3 * (i + 1)]) for i in range(8)]  # noqa
    out['message'] = header[pos_MS + 5:pos_MS + 5 + int(header[pos_MS + 2:pos_MS + 5])]  # noqa

    return out


def readDX(filename):
    """Data reader for German Weather Service DX product raw radar data files.

    This product uses a simple algorithm to compress zero values to reduce data
    file size.

    Note
    ----
    While the format appears to be well defined, there have been reports on DX-
    files that seem to produce errors. e.g. while one file usually contains a
    360 degree by 128 1km range bins, there are files, that contain 361 beams.

    Also, while usually azimuths are stored monotonously in ascending order,
    this is not guaranteed by the format. This routine does not (yet) check
    for this and directly returns the data in the order found in the file.
    If you are in doubt, check the 'azim' attribute.

    Be aware that this function does no extensive checking on its output.
    If e.g. beams contain different numbers of range bins, the resulting data
    will not be a 2-D array but a 1-D array of objects, which will most
    probably break calling code. It was decided to leave the handling of these
    (hopefully) rare events to the user, who might still be able to retrieve
    some reasonable data, instead of raising an exception, making it impossible
    to get any data from a file containing errors.

    Parameters
    ----------
    filename : string
        binary file of DX raw data

    Returns
    -------
    data : :func:`numpy:numpy.array`
        of image data [dBZ]; shape (360,128)

    attributes : dict
        dictionary of attributes - currently implemented keys:

        - 'azim' - azimuths np.array of shape (360,)
        - 'elev' - elevations (1 per azimuth); np.array of shape (360,)
        - 'clutter' - clutter mask; boolean array of same shape as `data`;
          corresponds to bit 15 set in each dataset.
        - 'bytes'- the total product length (including header).
          Apparently, this value may be off by one byte for unknown reasons
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

    Examples
    --------

    See :ref:`notebooks/fileio/wradlib_reading_dx.ipynb`.

    """

    azimuthbitmask = 2 ** (14 - 1)
    databitmask = 2 ** (13 - 1) - 1
    clutterflag = 2 ** 15
    dataflag = 2 ** 13 - 1

    f = get_radolan_filehandle(filename)

    # header string for later processing
    header = ''
    atend = False

    # read header
    while True:
        mychar = f.read(1)
        # 0x03 signals the end of the header but sometimes there might be
        # an additional 0x03 char after that

        if mychar == b'\x03':
            atend = True
        if mychar != b'\x03' and atend:
            break
        header += str(mychar.decode())

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
        # assert header[-2] == chr(3)
        buflen -= 1

    buf = f.read(buflen)
    # we can interpret the rest directly as a 1-D array of 16 bit unsigned ints
    raw = np.frombuffer(buf, dtype='uint16')

    # reading finished, close file, but only if we opened it.
    if isinstance(filename, io.IOBase):
        f.close()

    # a new ray/beam starts with bit 14 set
    # careful! where always returns its results in a tuple, so in order to get
    # the indices we have to retrieve element 0 of this tuple
    newazimuths = np.where(raw == azimuthbitmask)[0]  # Thomas kontaktieren!

    # for the following calculations it is necessary to have the end of the
    # data as the last index
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

    # attrs =  {}
    attrs['elev'] = np.array(elevs)
    attrs['azim'] = np.array(azims)
    attrs['clutter'] = (beams & clutterflag) != 0

    # converting the DWD rvp6-format into dBZ data and return as numpy array
    # together with attributes
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

    Note
    ----
    As Polygons are closed shapes, the first and the last vertex of each
    polygon **must** be the same!

    Examples
    --------
    Writes two triangle Polygons to a text file::

        poly1 = [[0.,0.,0.,0.],[0.,1.,0.,1.],[1.,1.,0.,2.],[0.,0.,0.,0.]]
        poly2 = [[0.,0.,0.,0.],[0.,1.,0.,1.],[1.,1.,0.,2.],[0.,0.,0.,0.]]
        polygons = [poly1, poly2]
        writePolygon2Text('polygons.txt', polygons)

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

    The corresponding NetCDF files from the EDGE software typically contain
    only one variable (e.g. reflectivity) for one elevation angle (sweep).
    The elevation angle is specified in the attributes keyword "Elevation".

    Please note that the radar might not return data with equidistant azimuth
    angles. In case you need equidistant azimuth angles, please set
    enforce_equidist to True.

    Parameters
    ----------
    filename : string
        path of the netCDF file
    enforce_equidist : boolean
        Set True if the values of the azimuth angles should be forced to be
        equidistant; default value is False

    Returns
    -------
    output : :func:`numpy:numpy.array`
        of image data (dBZ), dictionary of attributes
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
            az = np.linspace(np.round(az[ix_minaz], 2),
                             np.round(az[ix_maxaz], 2), len(az))
        else:
            az = np.roll(az, -ix_minaz)
        # rotate accordingly
        data = np.roll(data, -ix_minaz, axis=0)
        data = np.where(data == dset.getncattr('MissingData'), np.nan, data)
        # Ranges
        binwidth = ((dset.getncattr('MaximumRange-value') * 1000.) /
                    len(dset.dimensions['Gate']))
        r = np.arange(binwidth,
                      (dset.getncattr('MaximumRange-value') * 1000.) +
                      binwidth, binwidth)
        # collect attributes
        attrs = {}
        for attrname in dset.ncattrs():
            attrs[attrname] = dset.getncattr(attrname)
        # # Limiting the returned range
        # if range_lim and range_lim / binwidth <= data.shape[1]:
        #     data = data[:,:range_lim / binwidth]
        #     r = r[:range_lim / binwidth]
        # Set additional metadata attributes
        attrs['az'] = az
        attrs['r'] = r
        attrs['sitecoords'] = (attrs['Longitude'], attrs['Latitude'],
                               attrs['Height'])
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
    head : dict
        with known header token, value set to None
    """
    head = {'BY': None, 'VS': None, 'SW': None, 'PR': None,
            'INT': None, 'GP': None, 'MS': None, 'LV': None,
            'CS': None, 'MX': None, 'BG': None, 'ST': None,
            'VV': None, 'MF': None}
    return head


def get_radolan_header_token_pos(header):
    """Get Token and positions from DWD radolan header

    Parameters
    ----------
    header : string
        (ASCII header)

    Returns
    -------
    head : dictionary
        with found header tokens and positions
    """

    head_dict = get_radolan_header_token()

    for token in head_dict.keys():
        d = header.rfind(token)
        if d > -1:
            head_dict[token] = d
    head = {}

    result_dict = {}
    result_dict.update((k, v) for k, v in head_dict.items() if v is not None)
    for k, v in head_dict.items():
        if v is not None:
            start = v + len(k)
            filt = [x for x in result_dict.values() if x > v]
            if filt:
                stop = min(filt)
            else:
                stop = len(header)
            head[k] = (start, stop)
        else:
            head[k] = v

    return head


def parse_DWD_quant_composite_header(header):
    """Parses the ASCII header of a DWD quantitative composite file

    Parameters
    ----------
    header : string
        (ASCII header)

    Returns
    -------
    output : dictionary
        of metadata retrieved from file header
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
    # for k, v in head.iteritems():
    for k, v in head.items():
        if v:
            if k == 'BY':
                out['datasize'] = int(header[v[0]:v[1]]) - len(header) - 1
            if k == 'VS':
                out["maxrange"] = {0: "100 km and 128 km (mixed)",
                                   1: "100 km",
                                   2: "128 km",
                                   3: "150 km"}.get(int(header[v[0]:v[1]]),
                                                    "100 km")
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
                dimstrings = (dimstrings[:int(len(dimstrings) / 2)],
                              dimstrings[int(len(dimstrings) / 2):])
                out["nrow"] = int(dimstrings[0])
                out["ncol"] = int(dimstrings[1])
            if k == 'LV':
                lv = header[v[0]:v[1]].split()
                out['nlevel'] = np.int(lv[0])
                out['level'] = np.array(lv[1:]).astype('float')
            if k == 'MS':
                locationstring = (header[v[0]:].strip().split("<")[1].
                                  split(">")[0])
                out["radarlocations"] = locationstring.split(",")
            if k == 'ST':
                locationstring = (header[v[0]:].strip().split("<")[1].
                                  split(">")[0])
                out["radardays"] = locationstring.split(",")
            if k == 'CS':
                out['indicator'] = {0: "near ground level",
                                    1: "maximum",
                                    2: "tops"}.get(int(header[v[0]:v[1]]))
            if k == 'MX':
                out['imagecount'] = int(header[v[0]:v[1]])
            if k == 'VV':
                out['predictiontime'] = int(header[v[0]:v[1]])
            if k == 'MF':
                out['moduleflag'] = int(header[v[0]:v[1]])
    return out


def decode_radolan_runlength_line(line, attrs):
    """Decodes one line of runlength coded binary data of DWD
    composite file and returns decoded array

    Parameters
    ----------
    line : :func:`numpy:numpy.array`
        of byte values
    attrs : dict
        dictionary of attributes derived from file header

    Returns
    -------
    arr : :func:`numpy:numpy.array`
        of decoded values
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
        arr = np.append(arr, np.ones(trailing,
                                     dtype=np.uint8) * attrs['nodataflag'])
    elif trailing < 0:
        arr = dline[:trailing]

    return arr


def read_radolan_runlength_line(fid):
    """Reads one line of runlength coded binary data of DWD
    composite file and returns it as numpy array

    Parameters
    ----------
    fid : object
        file/buffer id

    Returns
    -------
    line : :func:`numpy:numpy.array`
        of coded values
    """
    line = fid.readline()

    # check if eot
    if line == b'\x04':
        return None

    # convert input buffer to np.uint8 array
    line = np.frombuffer(line, np.uint8).astype(np.uint8)

    return line


def decode_radolan_runlength_array(binarr, attrs):
    """Decodes the binary runlength coded section from DWD composite
    file and return decoded numpy array with correct shape

    Parameters
    ----------
    binarr : string
        Buffer
    attrs : dict
        Attribute dict of file header

    Returns
    -------
    arr : :func:`numpy:numpy.array`
        of decoded values
    """
    buf = io.BytesIO(binarr)

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
    fid : object
        file handle
    size : int
        number of bytes to read

    Returns
    -------
    binarr : string
        array of binary data
    """
    binarr = fid.read(size)
    fid.close()
    if len(binarr) != size:
        raise IOError('{0}: File corruption while reading {1}! \nCould not '
                      'read enough data!'.format(__name__, fid.name))
    return binarr


def get_radolan_filehandle(fname):
    """Opens radolan file and returns file handle

    Parameters
    ----------
    fname : string
        filename

    Returns
    -------
    f : object
        filehandle
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
    fid : object
        file handle

    Returns
    -------
    header : string
    """
    # rewind, just in case...
    fid.seek(0, 0)

    header = ''
    while True:
        mychar = fid.read(1)
        if mychar == b'\x03':
            break
        header += str(mychar.decode())
    return header


def read_RADOLAN_composite(f, missing=-9999, loaddata=True):
    """Read quantitative radar composite format of the German Weather Service

    The quantitative composite format of the DWD (German Weather Service) was
    established in the course of the
    RADOLAN project and includes several file
    types, e.g. RX, RO, RK, RZ, RP, RT, RC, RI, RG, PC, PG and many, many more.
    (see format description on the RADOLAN project homepage :cite:`DWD2009`).

    At the moment, the national RADOLAN composite is a 900 x 900 grid with 1 km
    resolution and in polar-stereographic projection. There are other grid
    resolutions for different composites (eg. PC, PG)

    Warning
    -------
    This function already evaluates and applies the so-called
    PR factor which is specified in the header section of the RADOLAN files.
    The raw values in an RY file are in the unit 0.01 mm/5min, while
    read_RADOLAN_composite returns values in mm/5min (i. e. factor 100 higher).
    The factor is also returned as part of attrs dictionary under
    keyword "precision".

    Parameters
    ----------
    f : string or file handle
        path to the composite file or file handle
    missing : int
        value assigned to no-data cells
    loaddata : bool
        True | False, If False function returns (None, attrs)

    Returns
    -------
    output : tuple
        tuple of two items (data, attrs):

            - data : :func:`numpy:numpy.array` of shape (number of rows,
              number of columns)
            - attrs : dictionary of metadata information from the file header

    Examples
    --------

    See :ref:`notebooks/radolan/radolan_format.ipynb`.

    """

    NODATA = missing
    mask = 0xFFF  # max value integer

    # If a file name is supplied, get a file handle
    try:
        header = read_radolan_header(f)
    except AttributeError:
        f = get_radolan_filehandle(f)
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

    if attrs['producttype'] in ['RX', 'EX', 'WX']:
        # convert to 8bit integer
        arr = np.frombuffer(indat, np.uint8).astype(np.uint8)
        arr = np.where(arr == 250, NODATA, arr)
        attrs['cluttermask'] = np.where(arr == 249)[0]
    elif attrs['producttype'] in ['PG', 'PC']:
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
        arr &= mask
        # consider negative flag if product is RD (differences from adjustment)
        if attrs['producttype'] == 'RD':
            # NOT TESTED, YET
            arr[negative] = -arr[negative]
        # apply precision factor
        # this promotes arr to float if precision is float
        arr = arr * attrs['precision']
        # set nodata value
        arr[nodata] = NODATA

    # anyway, bring it into right shape
    arr = arr.reshape((attrs['nrow'], attrs['ncol']))

    return arr, attrs


def browse_hdf5_group(grp):
    """Browses one hdf5 file level
    """
    pass


def read_generic_hdf5(fname):
    """Reads hdf5 files according to their structure

    In contrast to other file readers under :meth:`wradlib.io`, this function
    will *not* return a two item tuple with (data, metadata). Instead, this
    function returns ONE dictionary that contains all the file contents - both
    data and metadata. The keys of the output dictionary conform to the
    Group/Subgroup directory branches of the original file.

    Parameters
    ----------
    fname : string
        a hdf5 file path

    Returns
    -------
    output : dict
        a dictionary that contains both data and metadata according to the
        original hdf5 file structure

    Examples
    --------

    See :ref:`notebooks/fileio/wradlib_radar_formats.ipynb#Generic-HDF5`.

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

    Please refer to the OPERA data model documentation :cite:`OPERA-data-model`
    in order to understand how an hdf5 file is organized that conforms to the
    OPERA ODIM_H5 conventions.

    In contrast to other file readers under :meth:`wradlib.io`, this function
    will *not* return a two item tuple with (data, metadata). Instead, this
    function returns ONE dictionary that contains all the file contents - both
    data and metadata. The keys of the output dictionary conform to the
    Group/Subgroup directory branches of the original file.
    If the end member of a branch (or path) is "data", then the corresponding
    item of output dictionary is a numpy array with actual data.
    Any other end member (either *how*, *where*,
    and *what*) will contain the meta information applying to the corresponding
    level of the file hierarchy.

    Parameters
    ----------
    fname : string
        a hdf5 file path

    Returns
    -------
    output : dict
        a dictionary that contains both data and metadata according to the
        original hdf5 file structure

    """
    f = h5py.File(fname, "r")

    # now we browse through all Groups and Datasets and store the info in one
    # dictionary
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
    scan : object
        scan object from hdf5 file
    scan_type : string
        "PVOL" (plan position indicator) or "RHI" (range height indicator)

    Returns
    -------
    sattrs : dict
        dictionary of scan attributes
    """

    # global zero_index, el, az

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
        angle_step = int(np.round(sattrs['ele_stop'], 1) / angle_step)
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
    r = np.arange(sattrs['bin_range'],
                  sattrs['bin_range'] * sattrs['bin_count'] +
                  sattrs['bin_range'], sattrs['bin_range'])

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
    scan : object
        scan object from hdf5 file
    scan_type : string
        "PVOL" (plan position indicator) or "RHI" (range height indicator)
    wanted_moments : strings
        sequence of strings containing upper case names of moment(s) to
        be returned

    Returns
    -------
    data : dict
        dictionary of moment data (numpy arrays)
    sattrs : dict
        dictionary of scan attributes
    """

    # placeholder for data and attrs
    data = {}
    sattrs = {}

    # try to read wanted moments
    for mom in list(scan):
        if 'moment' in mom:
            data1 = {}
            sg2 = scan[mom]
            actual_moment = sg2.attrs.get('moment').decode().upper()
            if (actual_moment in wanted_moments) or (wanted_moments == 'all'):
                # read attributes only once
                if not sattrs:
                    sattrs = read_gamic_scan_attributes(scan, scan_type)
                mdata = sg2[...]
                dyn_range_max = sg2.attrs.get('dyn_range_max')
                dyn_range_min = sg2.attrs.get('dyn_range_min')
                bin_format = sg2.attrs.get('format').decode()
                if bin_format == 'UV8':
                    div = 256.0
                else:
                    div = 65536.0
                mdata = (dyn_range_min + mdata *
                         (dyn_range_max - dyn_range_min) / div)

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
    """Data reader for hdf5 files produced by the commercial
    GAMIC Enigma V3 MURAN software

    Provided by courtesy of Kai Muehlbauer (University of Bonn). See GAMIC
    homepage for further info (http://www.gamic.com).

    Parameters
    ----------
    filename : string
        path of the gamic hdf5 file
    wanted_elevations : strings
        sequence of strings of elevation_angle(s) of scan (only needed for PPI)
    wanted_moments : strings
        sequence of strings of moment name(s)

    Returns
    -------
    data : dict
        dictionary of scan and moment data (numpy arrays)
    attrs : dict
        dictionary of attributes

    Examples
    --------

    See :ref:`notebooks/fileio/wradlib_radar_formats.ipynb#GAMIC-HDF5`.

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
        f['how'].attrs.get('software')
    except KeyError:
        print("WRADLIB: File is no GAMIC hdf5!")
        raise

    # get scan_type (PVOL or RHI)
    scan_type = f['what'].attrs.get('object').decode()

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

                # try to read scan data and attrs
                # if wanted_elevations are found
                if (el in wanted_elevations) or (wanted_elevations == 'all'):
                    sdata, sattrs = read_gamic_scan(scan=g,
                                                    scan_type=scan_type,
                                                    wanted_moments=wanted_moments)  # noqa
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
        # vattrs['sitecoords'] = (vattrs['Longitude'], vattrs['Latitude'],
        #                         vattrs['Height'])
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
    output : dict
        a dictionary or list of dictionaries

    """
    for k, v in dictionary.items():
        if k == key:
            yield dictionary
        elif isinstance(v, dict):
            for result in find_key(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                if isinstance(d, dict):
                    for result in find_key(key, d):
                        yield result


def decompress(data):
    """Decompression of data

    Parameters
    ----------
    data : string
        (from xml) data string containing compressed data.
    """
    zlib = util.import_optional('zlib')
    return zlib.decompress(data)


def get_RB_data_layout(datadepth):
    """Calculates DataWidth and DataType from given DataDepth of
    RAINBOW radar data

    Parameters
    ----------
    datadepth : int
        DataDepth as read from the Rainbow xml metadata.

    Returns
    -------
    datawidth : int
        Width in Byte of data.

    datatype : string
        conversion string .
    """

    if sys.byteorder != 'big':
        byteorder = '>'
    else:
        byteorder = '<'

    datawidth = int(datadepth / 8)

    if datawidth in [1, 2, 4]:
        datatype = byteorder + 'u' + str(datawidth)
    else:
        raise ValueError("Wrong DataDepth: %d. "
                         "Conversion only for depth 8, 16, 32" % datadepth)

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
            raise KeyError('Attribute @{0} is missing from '
                           'Blob Description. There may be some '
                           'problems with your file'.format(attr))
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
    ret : Attribute Value

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
    datastring : string
        Blob Description String

    blobid : int
        Number of requested blob

    Returns
    -------
    data : string
        Content of blob

    """
    xmltodict = util.import_optional('xmltodict')

    start = 0
    searchString = '<BLOB blobid="{0}"'.format(blobid)
    start = datastring.find(searchString.encode(), start)
    if start == -1:
        raise EOFError('Blob ID {0} not found!'.format(blobid))
    end = datastring.find(b'>', start)
    xmlstring = datastring[start:end + 1]

    # cheat the xml parser by making xml well-known
    xmldict = xmltodict.parse(xmlstring.decode() + '</BLOB>')
    cmpr = get_RB_blob_attribute(xmldict, 'compression')
    size = int(get_RB_blob_attribute(xmldict, 'size'))
    data = datastring[end + 2:end + 2 + size]  # read blob data to string

    # decompress if necessary
    # the first 4 bytes are neglected for an unknown reason
    if cmpr == "qt":
        data = decompress(data[4:])

    return data


def map_RB_data(data, datadepth):
    """ Map BLOB data to correct DataWidth and Type and convert it
    to numpy array

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
    flagdepth = None
    if datadepth < 8:
        flagdepth = datadepth
        datadepth = 8

    datawidth, datatype = get_RB_data_layout(datadepth)

    # import from data buffer well aligned to data array
    data = np.ndarray(shape=(int(len(data) / datawidth),),
                      dtype=datatype, buffer=data)

    if flagdepth:
        data = np.unpackbits(data)

    return data


def get_RB_data_shape(blobdict):
    """
    Retrieve correct BLOB data shape from blobdict

    Parameters
    ----------
    blobdict : dict
        Blob Description Dict

    Returns
    -------
    tuple : shape
        shape of data
    """
    # this is a bit hacky, but we do not know beforehand,
    # so we extract this on the run
    try:
        dim0 = get_RB_data_attribute(blobdict, 'rays')
        dim1 = get_RB_data_attribute(blobdict, 'bins')
        # if rays and bins are found, return both
        return dim0, dim1
    except KeyError as e1:
        try:
            # if only rays is found, return rays
            return dim0
        except UnboundLocalError:
            try:
                # if both rays and bins not found assuming pixmap
                dim0 = get_RB_data_attribute(blobdict, 'rows')
                dim1 = get_RB_data_attribute(blobdict, 'columns')
                dim2 = get_RB_data_attribute(blobdict, 'depth')
                if dim2 < 8:
                    # if flagged data return rows x columns x depth
                    return dim0, dim1, dim2
                else:
                    # otherwise just rows x columns
                    return dim0, dim1
            except KeyError as e2:
                # if no some keys are missing, print errors and raise
                print(e1)
                print(e2)
                raise


def get_RB_blob_from_string(datastring, blobdict):
    """
    Read BLOB data from datastring and return it as numpy array with correct
    dataWidth and shape

    Parameters
    ----------
    datastring : string
        Blob Description String
    blobdict : dict
        Blob Description Dict

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
    data.shape = get_RB_data_shape(blobdict)

    return data


def get_RB_blob_from_file(f, blobdict):
    """
    Read BLOB data from file and return it with correct
    dataWidth and shape

    Parameters
    ----------
    f : string or file handle
        File handle of or path to Rainbow file
    blobdict : dict
        Blob Dict

    Returns
    -------
    data : numpy array
        Content of blob as numpy array
    """

    # Try to read the data from a file handle
    try:
        f.seek(0, 0)
        fid = f
        datastring = fid.read()
    except AttributeError:
        # If we did not get a file handle, assume that we got a filename,
        # get a file handle and read the data
        try:
            fid = open(f, "rb")
            datastring = fid.read()
            fid.close()
        except IOError:
            print("WRADLIB: Error opening Rainbow file ", f)
            raise IOError

    data = get_RB_blob_from_string(datastring, blobdict)

    return data


def get_RB_file_as_string(fid):
    """ Read Rainbow File Contents in dataString

    Parameters
    ----------
    fid : file handle
        File handle of Data File

    Returns
    -------
    dataString : string
        File Contents as dataString
    """

    try:
        dataString = fid.read()
    except:
        raise IOError('Could not read from file handle')

    return dataString


def get_RB_blobs_from_file(fid, rbdict):
    """Read all BLOBS found in given nested dict, loads them from file
    given by filename and add them to the dict at the appropriate position.

    Parameters
    ----------
    fid : file handle
        File handle of Data File
    rbdict : dict
        Rainbow file Contents

    Returns
    -------
    ret : dict
        Rainbow File Contents
    """

    blobs = list(find_key('@blobid', rbdict))

    datastring = get_RB_file_as_string(fid)
    for blob in blobs:
        data = get_RB_blob_from_string(datastring, blob)
        blob['data'] = data

    return rbdict


def get_RB_header(fid):
    """Read Rainbow Header from filename, converts it to a dict and returns it

    Parameters
    ----------
    fid : file handle
        File handle of Data File

    Returns
    -------
    object : dictionary
        Rainbow File Contents

    """

    # load the header lines, i.e. the XML part
    endXMLmarker = b"<!-- END XML -->"
    header = b""
    line = b""

    try:
        while not line.startswith(endXMLmarker):
            header += line[:-1]
            line = fid.readline()
            if len(line) == 0:
                break
    except:
        raise IOError('Could not read from file handle')

    xmltodict = util.import_optional('xmltodict')

    return xmltodict.parse(header)


def read_Rainbow(f, loaddata=True):
    """Reads Rainbow files files according to their structure

    In contrast to other file readers under :meth:`wradlib.io`, this function
    will *not* return a two item tuple with (data, metadata). Instead, this
    function returns ONE dictionary that contains all the file contents - both
    data and metadata. The keys of the output dictionary conform to the XML
    outline in the original data file.

    The radar data will be extracted from the data blobs, converted and added
    to the dict with key 'data' at the place where the @blobid was pointing
    from.

    Parameters
    ----------
    f : string or file handle
        a rainbow file path or file handle of rainbow file
    loaddata : bool
        True | False, If False function returns only metadata

    Returns
    -------
    rbdict : dict
        a dictionary that contains both data and metadata according to the
        original rainbow file structure

    Examples
    --------

    See :ref:`notebooks/fileio/wradlib_load_rainbow_example.ipynb`.

    .. versionchanged 0.10.0
       Added reading from file handles.

    """

    # Check if a file handle has been passed
    try:
        f.seek(0, 0)
        fid = f
    except AttributeError:
        # If we did not get a file handle, assume that we got a filename and
        #  get a file handle for the corresponding file
        try:
            fid = open(f, "rb")
        except IOError:
            print("WRADLIB: Error opening Rainbow file ", f)
            raise IOError

    rbdict = get_RB_header(fid)

    if loaddata:
        rbdict = get_RB_blobs_from_file(fid, rbdict)

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


def to_hdf5(fpath, data, mode="w", metadata=None,
            dataset="data", compression="gzip"):
    """Quick storage of one <data> array and a <metadata> dict in an hdf5 file

    This is more efficient than pickle, cPickle or numpy.save. The data is
    stored in a subgroup named ``data`` (i.e. hdf5file["data").
    See :meth:`~wradlib.io.from_hdf5` for retrieving stored data.

    Parameters
    ----------
    fpath : string
        path to the hdf5 file
    data : :func:`numpy:numpy.array`
    mode : string
        file open mode, defaults to "w" (create, truncate if exists)
    metadata : dict
        dictionary of data's attributes
    dataset : string
        describing dataset
    compression : string
        h5py compression type {"gzip"|"szip"|"lzf"}, see h5py documentation
        for details
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
    """Loading data from hdf5 files that was stored by \
    :meth:`~wradlib.io.to_hdf5`

    Parameters
    ----------
    fpath : string
        path to the hdf5 file
    dataset : string
        name of the Dataset in which the data is stored
    """
    f = h5py.File(fpath, mode="r")
    # Check whether Dataset exists
    if dataset not in f.keys():
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
    filename : string
        satellite file name

    Returns
    -------
    ds : gdal.DataSet
        with satellite data
    """

    root = gdal.Open(filename)
    ds1 = gdal.Open('HDF5:' + filename + '://CT')
    ds = gdal.GetDriverByName('MEM').CreateCopy('out', ds1, 0)

    # name = os.path.basename(filename)[7:11]
    try:
        proj = osr.SpatialReference()
        proj.ImportFromProj4(ds.GetMetadata()["PROJECTION"])
    except Exception:
        raise NameError("No metadata for satellite file %s" % filename)
    geotransform = root.GetMetadata()["GEOTRANSFORM_GDAL_TABLE"].split(",")
    geotransform[0] = root.GetMetadata()["XGEO_UP_LEFT"]
    geotransform[3] = root.GetMetadata()["YGEO_UP_LEFT"]
    ds.SetProjection(proj.ExportToWkt())
    ds.SetGeoTransform([float(x) for x in geotransform])
    return ds


def read_netcdf_group(ncid):
    """Reads netcdf (nested) groups into python dictionary with corresponding
    structure.

    Note
    ----
    The returned dictionary could be quite big, depending on the content of
    the file.

    Parameters
    ----------
    ncid : object
        nc/group id from netcdf file

    Returns
    -------
    out : ordered dict
        an ordered dictionary that contains both data and metadata
        according to the original netcdf file structure
    """
    out = OrderedDict()

    # attributes
    for k, v in ncid.__dict__.items():
        out[k] = v

    # groups
    if ncid.groups:
        for k, v in ncid.groups.items():
            out[k] = read_netcdf_group(v)

    # dimensions
    dimids = np.array([])
    if ncid.dimensions:
        dim = OrderedDict()
        for k, v in ncid.dimensions.items():
            tmp = OrderedDict()
            try:
                tmp['data_model'] = v._data_model
            except AttributeError:
                pass
            try:
                tmp['size'] = v.__len__()
            except AttributeError:
                pass
            tmp['dimid'] = v._dimid
            dimids = np.append(dimids, v._dimid)
            tmp['grpid'] = v._grpid
            tmp['isunlimited'] = v.isunlimited()
            dim[k] = tmp
        # Usually, the dimensions should be ordered by dimid automatically
        # in case netcdf used OrderedDict. However, we should double check
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
        for k, v in ncid.variables.items():
            tmp = OrderedDict()
            for k1 in v.ncattrs():
                tmp[k1] = v.getncattr(k1)
            if v[:].dtype.kind == 'S':
                try:
                    tmp['data'] = nc.chartostring(v[:])
                except:
                    tmp['data'] = v[:]
            else:
                tmp['data'] = v[:]
            var[k] = tmp
        out['variables'] = var

    return out


def read_generic_netcdf(fname):
    """Reads netcdf files and returns a dictionary with corresponding
    structure.

    In contrast to other file readers under :meth:`wradlib.io`, this function
    will *not* return a two item tuple with (data, metadata). Instead, this
    function returns ONE dictionary that contains all the file contents - both
    data and metadata. The keys of the output dictionary conform to the
    Group/Subgroup directory branches of the original file.

    Please see the examples below on how to browse through a return object. The
    most important keys are the "dimensions" which define the shape of the data
    arrays, and the "variables" which contain the actual data and typically
    also the data that define the dimensions (e.g. sweeps, azimuths, ranges).
    These keys should be present in any netcdf file.

    Note
    ----
    The returned dictionary could be quite big, depending on the content of
    the file.

    Parameters
    ----------
    fname : string
        a netcdf file path

    Returns
    -------
    out : ordered dict
        an ordered dictionary that contains both data and metadata according
        to the original netcdf file structure

    Examples
    --------

    See :ref:`notebooks/fileio/wradlib_generic_netcdf_example.ipynb`.

    """
    try:
        ncid = nc.Dataset(fname, 'r')
    except RuntimeError:
        print("wradlib1: Could not read %s." % fname)
        print("Check whether file exists, and whether it is a netCDF file.")
        print("Raising exception...")
        raise

    out = read_netcdf_group(ncid)

    ncid.close()

    return out


def _check_arguments(fpath, data):
    """Helper function to check input arguments for GIS export function
    """
    # Check arguments
    if not type(data) == np.ndarray:
        raise Exception("Argument 'data' in has to be of type numpy.ndarray. "
                        "Found argument of %s instead" % str(type(data)))

    if not data.ndim == 2:
        raise Exception("Argument 'data' has to be 2-dimensional. "
                        "Found %d dimensions instead" % data.ndim)

    if not os.path.exists(os.path.dirname(fpath)):
        raise Exception("Directory does not exist: %s" %
                        os.path.dirname(fpath))


def write_raster_dataset(fpath, dataset, format, options=None, remove=False):
    """ Write raster dataset to file format

    .. versionadded 0.10.0

    Parameters
    ----------
    fpath : string
        A file path - should have file extension corresponding to format.
    dataset : gdal.Dataset
        gdal raster dataset
    format : string
        gdal raster format string
    options : list
        List of option strings for the corresponding format.
    remove : bool
        if True, existing OGR.DataSource will be
        removed before creation

    Note
    ----
    For format and options refer to
    `formats_list <http://www.gdal.org/formats_list.html>`_.

    Examples
    --------
    See :ref:`notebooks/fileio/wradlib_gis_export_example.ipynb`.
    """
    # check for option list
    if options is None:
        options = []

    driver = gdal.GetDriverByName(format)
    metadata = driver.GetMetadata()

    # check driver capability
    if 'DCAP_CREATECOPY' in metadata and metadata['DCAP_CREATECOPY'] != 'YES':
        assert "Driver %s doesn't support CreateCopy() method.".format(format)

    if remove:
        if os.path.exists(fpath):
            driver.Delete(fpath)

    target = driver.CreateCopy(fpath, dataset, 0, options)
    del target


@util.deprecated(write_raster_dataset)
def to_AAIGrid(fpath, data, xllcorner, yllcorner, cellsize,
               nodata=-9999, proj=None, fmt="%.2f", to_esri=True):
    """Write a cartesian grid to an Arc/Info ASCII grid file.

    .. versionadded:: 0.6.0

    The function writes a text file to ``fpath`` that contains the header info
    and the grid data passed with the argument ``data``. Find details on ESRI
    grids (or Arc/Info ASCII grids) on wikipedia :cite:`ESRI-grid`.
    This should work for most GIS software systems
    (tested for QGIS and ESRI ArcGIS).

    In case a GDAL SpatialReference object (argument ``proj``) is passed,
    the function will also try to write an accompanying projection (``.prj``)
    file that has the same file name, but a different extension.

    Please refer to :mod:`wradlib.georef`
    to see how to create SpatialReference objects from e.g.
    EPSG codes :meth:`~wradlib.georef.epsg_to_osr`,
    PROJ.4 strings :meth:`~wradlib.georef.proj4_to_osr`,
    or WKT strings :meth:`~wradlib.georef.wkt_to_osr`. Other projections
    are addressed by :meth:`~wradlib.georef.create_osr`.

    Parameters
    ----------
    fpath : string
        a file path - must have a ".txt" or ".asc" extension.
    data : :func:`numpy:numpy.array`
        two dimensional numpy array of type integer or float
    xllcorner : float
        x coordinate of the lower left corner of the grid
    yllcorner : float
        y coordinate of the lower left corner of the grid
    cellsize : float
        size of the grid cells - needs to be consistent with proj
    nodata : float
        no data flag
    proj : osr.SpatialReference
        a SpatialReference of class 'osr.SpatialReference'
    fmt : string
        format string
    to_esri : bool
        set True if the prj file should be made ESRI compatible

    Note
    ----
    Has been tested with ESRI ArcGIS 9.3 and QGIS 2.8.

    Examples
    --------
    See :ref:`notebooks/fileio/wradlib_gis_export_example.ipynb`.

    """
    # Check input data
    _check_arguments(fpath, data)

    ext = os.path.splitext(fpath)[-1]
    if ext not in [".txt", ".asc"]:
        raise Exception("File name extension should be either "
                        "'.txt' or '.asc'. Found extension instead: %s" % ext)

    # Define header
    header = ("\n"
              "ncols         %d\n"
              "nrows         %d\n"
              "xllcorner     %.4f\n"
              "yllcorner     %.4f\n"
              "cellsize      %.4f\n"
              "NODATA_value  %.1f\n" % (data.shape[0], data.shape[1],
                                        xllcorner, yllcorner, cellsize,
                                        nodata))

    # Replace NaNs by NoData
    # ...but we do not want to manipulate the original array!
    data = data.copy()
    data[np.isnan(data)] = nodata

    # Write grid file
    # with open(fpath, "w") as f:
    #    f.write(header)
    np.savetxt(fpath, np.flipud(data), fmt=fmt, header=header, comments='')

    if proj is None:
        # No prj file will be written
        return 0
    elif not type(proj) == osr.SpatialReference:
        raise Exception("Expected 'proj' argument of type "
                        "'osr.SpatialReference', but got %s. See library "
                        "reference for wradlib.georef on how to create "
                        "SpatialReference objects from different sources "
                        "(proj4, WKT, EPSG, ...)." % type(proj))

    if to_esri:
        # Make a copy before manipulation
        proj = proj.Clone()
        proj.MorphToESRI()

    # Write projection file
    prjpath = os.path.splitext(fpath)[0] + ".prj"
    with open(prjpath, "w") as f:
        f.write(proj.ExportToWkt())

    return 0


@util.deprecated(write_raster_dataset)
def to_GeoTIFF(fpath, data, geotransform, nodata=-9999, proj=None):
    """Write a cartesian grid to a GeoTIFF file.

    .. versionadded:: 0.6.0

    The function writes a GeoTIFF file to ``fpath`` that contains the grid data
    passed with the argument ``data``. For details on the GeoTIFF format
    see e.g. wikipedia :cite:`GeoTIFF`.

    Warning
    -------
    The GeoTIFF files produced by this function might not work with ESRI
    ArcGIS, depending on the projection. Problems are particularly expected
    with the RADOLAN projection, due to inconsistencies in the definition of
    polar stereographic projections between GDAL and ESRI ArcGIS.

    The projection information (argument ``proj``) needs to be passed as a GDAL
    SpatialReference object. Please refer to :mod:`wradlib.georef`
    to see how to create SpatialReference objects from e.g.
    EPSG codes :meth:`~wradlib.georef.epsg_to_osr`,
    PROJ.4 strings :meth:`~wradlib.georef.proj4_to_osr`,
    or WKT strings :meth:`~wradlib.georef.wkt_to_osr`. Other projections
    are addressed by :meth:`~wradlib.georef.create_osr`.

    Writing a GeoTIFF file requires a ``geotransform`` list to define how to
    compute map coordinates from grid indices. The list needs to contain the
    following items: top left x, w-e pixel resolution, rotation, top left y,
    rotation, n-s pixel resolution. The unit of the pixel resolution has to be
    consistent with the projection information. **BE CAREFUL**: You need to
    consider whether your grid coordinates define the corner (typically lower
    left) or the center of your pixels.
    And since the ``geotransform`` is used to define the grid from the top-left
    corner, the n-s pixel resolution is usually a negative value.

    Here is an example of the ``geotransform`` that worked e.g. with RADOLAN
    grids. Notice that the RADOLAN coordinates returned by wradlib refer to the
    lower left pixel corners, so you have to add another pixel unit to the top
    left y coordinate in order to define the top left corner of the
    bounding box::

        import wradlib
        xy = wradlib.georef.get_radolan_grid(900,900)
        # top left x, w-e pixel size, rotation, top left y, rotation,
        # n-s pixel size
        geotransform = [xy[0,0,0], 1., 0, xy[-1,-1,1]+1., 0, -1.]

    Parameters
    ----------
    fpath : string
        a file path - must have a ".txt" or ".asc" extension.
    data : :func:`numpy:numpy.array`
        two dimensional numpy array of type integer or float
    geotransform : sequence
        sequence of length 6 (# top left x, w-e pixel size, rotation,
        top left y, rotation, n-s pixel size)
    nodata : float
        no data flag
    proj : osr.SpatialReference
        a SpatialReference of class 'osr.SpatialReference'

    Note
    ----
    Has been tested with ESRI ArcGIS 9.3 and QGIS 2.8.

    Examples
    --------

    See :ref:`notebooks/fileio/wradlib_gis_export_example.ipynb`.

    """
    # Check input data
    _check_arguments(fpath, data)
    ext = os.path.splitext(fpath)[-1]
    if ext not in [".tif", ".tiff"]:
        raise Exception("File name extension should be either '.tif' or "
                        "'.tiff'. Found extension instead: %s" % ext)

    # Set up our export object
    driver = gdal.GetDriverByName("GTiff")

    # Mapping ur data type to GDAL data types
    if data.dtype == "float64":
        gdal_dtype = gdal.GDT_Float64
    elif data.dtype == "float32":
        gdal_dtype = gdal.GDT_Float32
    elif data.dtype == "int32":
        gdal_dtype = gdal.GDT_Int32
    elif data.dtype == "int16":
        gdal_dtype = gdal.GDT_Int16
    else:
        raise Exception("The data type of your input array data should be one "
                        "of the following: float64, float32, int32, int16. "
                        "You can use numpy's 'astype' method to convert "
                        "your array to the desired data type.")

    # Creat our export object
    ds = driver.Create(fpath, data.shape[0], data.shape[1], 1, gdal_dtype)

    # set the reference info
    if proj is None:
        pass
    elif not isinstance(proj, osr.SpatialReference):
        raise Exception("Expected 'proj' argument of type "
                        "'osr.SpatialReference', but got %s. See library "
                        "reference for wradlib.georef on how to create "
                        "SpatialReference objects from different sources "
                        "(proj4, WKT, EPSG, ...)." % type(proj))
    else:
        ds.SetProjection(proj.ExportToWkt())

    # top left x, w-e pixel resolution, rotation, top left y, rotation,
    # n-s pixel resolution
    ds.SetGeoTransform(geotransform)

    # Replace NaNs by NoData
    # ...but we do not want to manipulate the original array!
    data = data.copy()
    data[np.isnan(data)] = nodata
    # and replace them by NoData flag
    ds.GetRasterBand(1).SetNoDataValue(nodata)

    # Write data
    ds.GetRasterBand(1).WriteArray(np.flipud(data))

    # This is how we close the export file
    ds = None


@util.deprecated(georef.extract_raster_dataset)
def read_raster_data(filename, driver=None, **kwargs):
    """Read raster data

    Reads raster data files supported by GDAL. If driver is not given,
    GDAL tries to autodetect the file format. Works well in most cases.

    .. seealso:: http://www.gdal.org/formats_list.html

    Resamples data on the fly if special keyword arguments are given

    .. versionadded:: 0.6.0

    Parameters
    ----------
    filename : string
        filename of raster file
    driver : string
        GDAL Raster Format Code
        see: http://www.gdal.org/formats_list.html
        if no driver is given gdal is autodetecting which may fail

    Keyword Arguments
    -----------------
    spacing : float or tuple of two floats
        pixel spacing of resampled dataset, same unit as pixel coordinates
    size : tuple of two ints
        X/YRasterSize of resampled dataset
    resample : GDALResampleAlg
        defaults to GRA_Bilinear
        GRA_NearestNeighbour = 0, GRA_Bilinear = 1, GRA_Cubic = 2,
        GRA_CubicSpline = 3, GRA_Lanczos = 4, GRA_Average = 5, GRA_Mode = 6,
        GRA_Max = 8, GRA_Min = 9, GRA_Med = 10, GRA_Q1 = 11, GRA_Q3 = 12

    Returns
    -------
    coords : :func:`numpy:numpy.array`
        numpy ndarray of raster coordinates
    values : :func:`numpy:numpy.array`
        numpy 2darray of raster values

    Examples
    --------

    See :ref:`notebooks/beamblockage/wradlib_beamblock.ipynb` and
    :ref:`notebooks/visualisation/wradlib_overlay.ipynb`

    """

    dataset = open_raster(filename, driver=driver)

    if 'spacing' in kwargs or 'size' in kwargs:
        dataset1 = georef.resample_raster_dataset(dataset, **kwargs)
    else:
        dataset1 = dataset

    # we have to flipud data, because raster data is origin "upper left"
    values = np.flipud(georef.read_gdal_values(dataset1))
    coords = np.flipud(georef.read_gdal_coordinates(dataset1,
                                                    mode='centers',
                                                    z=False))

    # close dataset
    dataset1 = None

    return coords, values


def open_shape(filename, driver=None):
    """
    Open shapefile, return ogr dataset and layer

    .. warning:: dataset and layer have to live in the same context,
                 if dataset is deleted all layer references will get lost

    .. versionadded:: 0.6.0

    Parameters
    ----------
    filename : string
        shapefile name
    driver : string
        gdal driver string

    Returns
    -------
    dataset : ogr.Dataset
        dataset
    layer : ogr.Layer
        layer
    """

    if driver is None:
        driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(filename)
    if dataset is None:
        print('Could not open file')
        raise IOError
    layer = dataset.GetLayer()
    return dataset, layer


def open_raster(filename, driver=None):
    """
    Open raster file, return ogr dataset

    .. versionadded:: 0.6.0

    Parameters
    ----------
    filename : string
        raster file name
    driver : string
        gdal driver string

    Returns
    -------
    dataset : ogr.Dataset
        dataset
    """

    dataset = gdal.Open(filename)

    if driver:
        gdal.GetDriverByName(driver)

    return dataset


if __name__ == '__main__':
    print('wradlib: Calling module <io> as main...')
