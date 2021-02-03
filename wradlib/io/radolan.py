#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Read RADOLAN and DX
^^^^^^^^^^^^^^^^^^^
Reading DX and RADOLAN data from German Weather Service

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "read_dx",
    "read_radolan_composite",
    "get_radolan_filehandle",
    "read_radolan_header",
    "parse_dwd_composite_header",
    "read_radolan_binary_array",
    "decode_radolan_runlength_array",
    "radolan_to_xarray",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
import io
import re
import warnings

import numpy as np
import xarray as xr

from wradlib import util
from wradlib.georef import rect

# current DWD file naming pattern (2008) for example:
# raa00-dx_10488-200608050000-drs---bin
dwdpattern = re.compile("raa..-(..)[_-]([0-9]{5})-([0-9]*)-(.*?)---bin")


def _get_timestamp_from_filename(filename):
    """Helper function doing the actual work of get_dx_timestamp"""
    time = dwdpattern.search(filename).group(3)
    if len(time) == 10:
        time = "20" + time
    return dt.datetime.strptime(time, "%Y%m%d%H%M")


def get_dx_timestamp(name):
    """Converts a dx-timestamp (as part of a dx-product filename) to a
    python datetime.object.

    Parameters
    ----------
    name : string
        representing a DWD product name

    Returns
    -------
    time : timezone-aware datetime.datetime object
    """
    return _get_timestamp_from_filename(name).replace(tzinfo=util.UTC())


def unpack_dx(raw):
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
    beam.extend(raw[0 : flagged[0]])

    # iterate over all flags except the last one
    for this, nxt in zip(flagged[:-1], flagged[1:]):
        # create as many zeros as there are given within the flagged
        # byte's data part
        beam.extend([0] * (raw[this] & data))
        # append the data until the next flag
        beam.extend(raw[this + 1 : nxt])

    # process the last flag
    # add zeroes
    beam.extend([0] * (raw[flagged[-1]] & data))

    # add remaining data
    beam.extend(raw[flagged[-1] + 1 :])

    # return the data
    return np.array(beam)


def get_dx_header_token():
    """Return array with known header token of dx data

    Returns
    -------
    head : dict
        with known header token, value set to None
    """
    head = {
        "BY": None,
        "VS": None,
        "CO": None,
        "CD": None,
        "CS": None,
        "EP": None,
        "MS": None,
    }
    return head


def parse_dx_header(header):
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
    out["datetime"] = dt.datetime.strptime(
        header[2:8] + header[13:17] + "00", "%d%H%M%m%y%S"
    )
    # Make it aware of its time zone (UTC)
    out["datetime"] = out["datetime"].replace(tzinfo=util.UTC())
    # radar location ID (always 10000 for composites)
    out["radarid"] = header[8:13]

    head = get_radolan_header_token_pos(header, mode="dx")

    # iterate over token and fill output dict accordingly
    for k, v in head.items():
        if v:
            if k == "BY":
                out["bytes"] = int(header[v[0] : v[1]])
            if k == "VS":
                out["version"] = header[v[0] : v[1]]
            if k == "CO":
                out["cluttermap"] = int(header[v[0] : v[1]])
            if k == "CD":
                out["dopplerfilter"] = int(header[v[0] : v[1]])
            if k == "CS":
                out["statfilter"] = int(header[v[0] : v[1]])
            if k == "EP":
                out["elevprofile"] = [
                    float(header[v[0] + 3 * i : v[0] + 3 * (i + 1)]) for i in range(8)
                ]
            if k == "MS":
                try:
                    cnt = int(header[v[0] : v[0] + 3])
                    out["message"] = header[v[0] + 3 : v[0] + 3 + cnt]
                except ValueError:
                    pass

    return out


def read_dx(filename):
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
    filename : string or file-like
        filename of binary file of DX raw data or file-like object

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
    See :ref:`/notebooks/fileio/wradlib_reading_dx.ipynb`.
    """

    azimuthbitmask = 2 ** (14 - 1)
    databitmask = 2 ** (13 - 1) - 1
    clutterflag = 2 ** 15
    dataflag = 2 ** 13 - 1

    with get_radolan_filehandle(filename) as f:

        # header string for later processing
        header = ""
        atend = False

        # read header
        while True:
            mychar = f.read(1)
            # 0x03 signals the end of the header but sometimes there might be
            # an additional 0x03 char after that

            if mychar == b"\x03":
                atend = True
            if mychar != b"\x03" and atend:
                break
            header += str(mychar.decode())

        attrs = parse_dx_header(header)

        # position file at end of header
        f.seek(len(header))

        # read number of bytes as declared in the header
        # intermediate fix:
        # if product length is uneven but header is even (e.g. because it has two
        # chr(3) at the end, read one byte less
        buflen = attrs["bytes"] - len(header)
        if (buflen % 2) != 0:
            # make sure that this is consistent with our assumption
            # i.e. contact DWD again, if DX files show up with uneven byte lengths
            # *and* only one 0x03 character
            # assert header[-2] == chr(3)
            buflen -= 1

        buf = f.read(buflen)
        # we can interpret the rest directly as a 1-D array of 16 bit unsigned ints
        raw = np.frombuffer(buf, dtype="uint16")

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
        beam = unpack_dx(raw[newazimuths[i] + 3 : newazimuths[i + 1]])
        beams.append(beam)
        elevs.append((raw[newazimuths[i] + 2] & databitmask) / 10.0)
        azims.append((raw[newazimuths[i] + 1] & databitmask) / 10.0)

    beams = np.array(beams)

    # attrs =  {}
    attrs["elev"] = np.array(elevs)
    attrs["azim"] = np.array(azims)
    attrs["clutter"] = (beams & clutterflag) != 0

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
    head = {
        "BY": None,
        "VS": None,
        "SW": None,
        "PR": None,
        "INT": None,
        "GP": None,
        "MS": None,
        "LV": None,
        "CS": None,
        "MX": None,
        "BG": None,
        "ST": None,
        "VV": None,
        "MF": None,
        "QN": None,
        "VR": None,
        "U": None,
    }
    return head


def get_radolan_header_token_pos(header, mode="composite"):
    """Get Token and positions from DWD radolan header

    Parameters
    ----------
    header : string
        (ASCII header)

    Keyword Arguments
    -----------------
    mode : str
        'composite' or 'dx', defaults to 'composite'

    Returns
    -------
    head : dictionary
        with found header tokens and positions
    """

    if mode == "composite":
        head_dict = get_radolan_header_token()
    elif mode == "dx":
        head_dict = get_dx_header_token()
    else:
        raise ValueError(
            f"unknown mode {mode}, use either 'composite' or 'dx' depending on data source"
        )

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


def parse_dwd_composite_header(header):
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
    out["datetime"] = dt.datetime.strptime(
        header[2:8] + header[13:17] + "00", "%d%H%M%m%y%S"
    )
    # radar location ID (always 10000 for composites)
    out["radarid"] = header[8:13]

    # get dict of header token with positions
    head = get_radolan_header_token_pos(header)
    # iterate over token and fill output dict accordingly
    for k, v in head.items():
        if v:
            if k == "BY":
                out["datasize"] = int(header[v[0] : v[1]]) - len(header) - 1
            if k == "VS":
                out["maxrange"] = {
                    0: "100 km and 128 km (mixed)",
                    1: "100 km",
                    2: "128 km",
                    3: "150 km",
                }.get(int(header[v[0] : v[1]]), "100 km")
            if k == "SW":
                out["radolanversion"] = header[v[0] : v[1]].strip()
            if k == "PR":
                out["precision"] = float("1" + header[v[0] : v[1]].strip())
            if k == "INT":
                out["intervalseconds"] = int(header[v[0] : v[1]]) * 60
            if k == "U":
                out["intervalunit"] = int(header[v[0] : v[1]])
                if out["intervalunit"] == 1:
                    out["intervalseconds"] *= 1440
            if k == "GP":
                dimstrings = header[v[0] : v[1]].strip().split("x")
                out["nrow"] = int(dimstrings[0])
                out["ncol"] = int(dimstrings[1])
            if k == "BG":
                dimstrings = header[v[0] : v[1]]
                dimstrings = (
                    dimstrings[: int(len(dimstrings) / 2)],
                    dimstrings[int(len(dimstrings) / 2) :],
                )
                out["nrow"] = int(dimstrings[0])
                out["ncol"] = int(dimstrings[1])
            if k == "LV":
                lv = header[v[0] : v[1]].split()
                out["nlevel"] = np.int_(lv[0])
                out["level"] = np.array(lv[1:]).astype("float")
            if k == "MS":
                locationstring = header[v[0] :].strip().split("<")[1].split(">")[0]
                out["radarlocations"] = locationstring.split(",")
            if k == "ST":
                locationstring = header[v[0] :].strip().split("<")[1].split(">")[0]
                out["radardays"] = locationstring.split(",")
            if k == "CS":
                out["indicator"] = {
                    0: "near ground level",
                    1: "maximum",
                    2: "tops",
                }.get(int(header[v[0] : v[1]]))
            if k == "MX":
                out["imagecount"] = int(header[v[0] : v[1]])
            if k == "VV":
                out["predictiontime"] = int(header[v[0] : v[1]])
            if k == "MF":
                out["moduleflag"] = int(header[v[0] : v[1]])
            if k == "QN":
                out["quantification"] = int(header[v[0] : v[1]])
            if k == "VR":
                out["reanalysisversion"] = header[v[0] : v[1]].strip()
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
        return np.ones(attrs["ncol"], dtype=np.uint8) * attrs["nodataflag"]
    offset = byte - 16

    # check if offset byte is 255 and take next byte(s)
    # also for the offset
    while byte == 255:
        lo += 1
        byte = line[lo]
        offset += byte - 16

    # just take the rest
    dline = line[lo + 1 :]

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
            arr = np.ones(offset, dtype=np.uint8) * attrs["nodataflag"]
        arr = np.append(arr, np.ones(width, dtype=np.uint8) * val)

    trailing = attrs["ncol"] - len(arr)
    if trailing > 0:
        arr = np.append(arr, np.ones(trailing, dtype=np.uint8) * attrs["nodataflag"])
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
    if line == b"\x04":
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


def read_radolan_binary_array(fid, size, raise_on_error=True):
    """Read binary data from file given by filehandle

    Parameters
    ----------
    fid : object
        file handle
    size : int
        number of bytes to read
    raise_on_error : bool
        raise IOError if data is truncated

    Returns
    -------
    binarr : string
        array of binary data
    """
    binarr = fid.read(size)
    if raise_on_error and len(binarr) != size:
        try:
            desc = fid.name
        except AttributeError:
            desc = repr(fid)
        raise IOError(
            "{0}: File corruption while reading {1}! \nCould not "
            "read enough data!".format(__name__, desc)
        )
    return binarr


def get_radolan_filehandle(fname):
    """Opens radolan file and returns file handle

    Parameters
    ----------
    fname : string or file-like
        filename or file-like object

    Returns
    -------
    f : object
        file handle
    """
    ret = lambda obj: obj
    if isinstance(fname, str):
        gzip = util.import_optional("gzip")
        # open file handle
        try:
            with gzip.open(fname, "rb") as f:
                f.read(1)
                f.seek(0, 0)
                ret = gzip.open
        except IOError:
            with open(fname, "rb") as f:
                f.read(1)
                f.seek(0, 0)
                ret = open
        return ret(fname, "rb")
    else:
        return ret(fname)


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

    header = ""
    while True:
        mychar = fid.read(1)
        if not mychar:
            raise EOFError("Unexpected EOF detected while reading " "RADOLAN header")
        if mychar == b"\x03":
            break
        header += str(mychar.decode())
    return header


def read_radolan_composite(f, missing=-9999, loaddata=True, fillmissing=False):
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
    read_radolan_composite returns values in mm/5min (i. e. factor 100 higher).
    The factor is also returned as part of attrs dictionary under
    keyword "precision".

    Note
    ----
    Using `loaddata='xarray'` the data is wrapped in an xarray Dataset.
    x,y, dimension as well as x,y and time coordinates
    (polar stereographic projection) are supplied.

    Parameters
    ----------
    f : string or file-like
        path to the composite file or file-like object
    missing : int
        value assigned to no-data cells
    loaddata : bool | str
        True | False | 'xarray', If False function returns (None, attrs)
        If 'xarray' returns (xarray Dataset, attrs)
    fillmissing : bool
        If True fills truncated values with "missing". Defaults to False.
        Does not work for run-length encoded files ("PC" and "PG).

    Returns
    -------
    output : tuple
        tuple of two items (data, attrs):
            - data : :func:`numpy:numpy.array` of shape (number of rows,
              number of columns) | xarray.Dataset
            - attrs : dictionary of metadata information from the file header

    Examples
    --------
    See :ref:`/notebooks/radolan/radolan_format.ipynb`.
    """

    NODATA = missing
    mask = 0xFFF  # max value integer

    # get file handle
    with get_radolan_filehandle(f) as fid:

        header = read_radolan_header(fid)
        attrs = parse_dwd_composite_header(header)

        if not loaddata:
            # if close:
            #     f.close()
            return None, attrs

        attrs["nodataflag"] = NODATA

        if not attrs["radarid"] == "10000":
            warnings.warn(
                "WARNING: You are using function e"
                + "wradlib.io.read_RADOLAN_composit for a non "
                + "composite file.\n "
                + "This might work...but please check the validity "
                + "of the results"
            )

        # handle truncated data
        binarr_kwargs = {}
        if fillmissing and attrs["producttype"] not in ["PG", "PC"]:
            binarr_kwargs.update(dict(raise_on_error=False))

        # read the actual data
        indat = read_radolan_binary_array(fid, attrs["datasize"], **binarr_kwargs)

        # helper function to fill truncated data with 'nodata' values
        def _from_buffer(data, size, dtype):
            if len(data) < size:
                isize = np.dtype(dtype).itemsize
                fill_value = 250 if isize == 1 else 8192
                if len(data) % isize:
                    data = data[:-1]
                fill = np.full((size - len(data)) // isize, fill_value, dtype=dtype)
                data += fill.tobytes()
            return np.frombuffer(data, dtype).astype(dtype)

    if attrs["producttype"] in ["RX", "EX", "WX"]:
        # convert to 8bit integer
        arr = _from_buffer(indat, attrs["datasize"], np.uint8)
        attrs["nodatamask"] = np.where(arr == 250)[0]
        attrs["cluttermask"] = np.where(arr == 249)[0]
    elif attrs["producttype"] in ["PG", "PC"]:
        arr = decode_radolan_runlength_array(indat, attrs)
        attrs["nodatamask"] = np.where(arr == 255)[0]
    else:
        # convert to 16-bit integers
        arr = _from_buffer(indat, attrs["datasize"], np.uint16)
        # evaluate bits 13, 14, 15 and 16
        attrs["secondary"] = np.where(arr & 0x1000)[0]
        attrs["nodatamask"] = np.where(arr & 0x2000)[0]
        negative = np.where(arr & 0x4000)[0]
        attrs["cluttermask"] = np.where(arr & 0x8000)[0]
        # mask out the last 4 bits
        arr &= mask
        # consider negative flag if product is RD (differences from adjustment)
        if attrs["producttype"] == "RD":
            # NOT TESTED, YET
            arr[negative] = -arr[negative]

    # anyway, bring it into right shape
    arr = arr.reshape((attrs["nrow"], attrs["ncol"]))

    if loaddata == "xarray":
        arr = radolan_to_xarray(arr, attrs)
    else:
        # apply precision factor
        # this promotes arr to float if precision is float
        if "precision" in attrs:
            arr = arr * attrs["precision"]
        # set nodata value
        if "nodatamask" in attrs:
            arr.flat[attrs["nodatamask"]] = NODATA

    return arr, attrs


def _get_radolan_product_attributes(attrs):
    """Create RADOLAN product attributes dictionary

    Parameters
    ----------
    attrs : dict
        dictionary of metadata information from the file header

    Returns
    -------
    pattrs : dict
        RADOLAN product attributes
    """
    product = attrs["producttype"]
    pattrs = {}

    if product not in ["PG", "PC"]:
        interval = attrs["intervalseconds"]
        precision = attrs["precision"]

    if product in ["RX", "EX", "WX", "WN"]:
        pattrs.update(radolan["dBZ"])
    elif product in ["RY", "RZ", "EY", "EZ", "YW"]:
        pattrs.update(radolan["RR"])
        scale_factor = np.float32(precision * 3600 / interval)
        pattrs.update({"scale_factor": scale_factor})
    elif product in [
        "RH",
        "RB",
        "RW",
        "RL",
        "RU",
        "EH",
        "EB",
        "EW",
        "SQ",
        "SH",
        "SF",
        "W1",
        "W2",
        "W3",
        "W4",
    ]:
        pattrs.update(radolan["RA"])
        pattrs.update({"scale_factor": np.float32(precision)})
    elif product in ["PG", "PC"]:
        pattrs.update(radolan["PG"])
    else:
        raise ValueError("WRADLIB: unkown RADOLAN product!")

    return pattrs


def radolan_to_xarray(data, attrs):
    """Converts RADOLAN data to xarray Dataset

    Parameters
    ----------
    data : :func:`numpy:numpy.array`
        array of shape (number of rows, number of columns)
    attrs : dict
        dictionary of metadata information from the file header

    Returns
    -------
    dset : xarray.Dataset
        RADOLAN data and coordinates
    """
    product = attrs["producttype"]
    pattrs = _get_radolan_product_attributes(attrs)
    radolan_grid_xy = rect.get_radolan_grid(attrs["nrow"], attrs["ncol"])
    x0 = radolan_grid_xy[0, :, 0]
    y0 = radolan_grid_xy[:, 0, 1]
    if pattrs:
        if "nodatamask" in attrs:
            data.flat[attrs["nodatamask"]] = pattrs["_FillValue"]
        if "cluttermask" in attrs:
            data.flat[attrs["cluttermask"]] = pattrs["_FillValue"]
    darr = xr.DataArray(
        data,
        attrs=pattrs,
        dims=["y", "x"],
        coords={"time": attrs["datetime"], "x": x0, "y": y0},
    )
    dset = xr.Dataset({product: darr})
    dset = dset.pipe(xr.decode_cf)

    return dset


radolan = {
    "dBZ": {
        "scale_factor": np.float32(0.5),
        "add_offset": np.float32(-32.5),
        "valid_min": np.int32(0),
        "valid_max": np.int32(255),
        "_FillValue": np.int32(255),
        "standard_name": "equivalent_reflectivity_factor",
        "long_name": "equivalent_reflectivity_factor",
        "unit": "dBZ",
    },
    "RR": {
        "add_offset": np.float_(0),
        "valid_min": np.int32(0),
        "valid_max": np.int32(4095),
        "_FillValue": np.int32(65535),
        "standard_name": "rainfall_rate",
        "long_name": "rainfall_rate",
        "unit": "mm h-1",
    },
    "RA": {
        "add_offset": np.float_(0),
        "valid_min": np.int32(0),
        "valid_max": np.int32(4095),
        "_FillValue": np.int32(65535),
        "standard_name": "rainfall_amount",
        "long_name": "rainfall_amount",
        "unit": "mm",
    },
    "PG": {
        "valid_min": np.int32(0),
        "valid_max": np.int32(255),
        "_FillValue": np.int32(255),
    },
}
