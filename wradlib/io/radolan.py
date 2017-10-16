#!/usr/bin/env python
# Copyright (c) 2011-2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Read RADOLAN and DX
^^^^^^^^^^^^^^^^^^^
Reading DX and RADOLAN data from German Weather Service

.. autosummary::
    :nosignatures:
    :toctree: generated/

    readDX
    read_RADOLAN_composite
    get_radolan_filehandle
    read_radolan_header
    parse_DWD_quant_composite_header
    read_radolan_binary_array
    decode_radolan_runlength_array
"""

# standard libraries
from __future__ import absolute_import
import datetime as dt

try:
    from StringIO import StringIO
    import io
except ImportError:
    from io import StringIO  # noqa
    import io

import re
import warnings

# site packages
import numpy as np
from .. import util as util

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
            'VV': None, 'MF': None, 'QN': None}
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
            if k == 'QN':
                out['quantification'] = int(header[v[0]:v[1]])
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
        if not mychar:
            raise EOFError('Unexpected EOF detected while reading '
                           'RADOLAN header')
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
