#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Rainbow Data I/O
^^^^^^^^^^^^^^^^
.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["read_rainbow"]
__doc__ = __doc__.format("\n   ".join(__all__))

import sys

import numpy as np

from wradlib import util


def find_key(key, dictionary):
    """Searches for given key in given (nested) dictionary.

    Returns all found parent dictionaries in a list.

    Parameters
    ----------
    key : str
        the key to be searched for in the nested dict
    dictionary : dict
        the dictionary to be searched

    Yields
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
    data : str
        (from xml) data string containing compressed data.
    """
    zlib = util.import_optional("zlib")
    return zlib.decompress(data)


def get_rb_data_layout(datadepth):
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
    datatype : str
        conversion string .
    """

    if sys.byteorder != "big":
        byteorder = ">"
    else:
        byteorder = "<"

    datawidth = int(datadepth / 8)

    if datawidth in [1, 2, 4]:
        datatype = byteorder + "u" + str(datawidth)
    else:
        raise ValueError(
            "Wrong DataDepth: %d. " "Conversion only for depth 8, 16, 32" % datadepth
        )

    return datawidth, datatype


def get_rb_data_attribute(xmldict, attr):
    """Get Attribute `attr` from dict `xmldict`

    Parameters
    ----------
    xmldict : dict
        Blob Description Dictionary
    attr : str
        Attribute key

    Returns
    -------
    sattr : int
        Attribute Values
    """

    try:
        sattr = int(xmldict["@" + attr])
    except KeyError:
        raise KeyError(
            "Attribute @{0} is missing from "
            "Blob Description. There may be some "
            "problems with your file".format(attr)
        )
    return sattr


def get_rb_blob_attribute(blobdict, attr):
    """Get Attribute `attr` from dict `blobdict`

    Parameters
    ----------
    blobdict : dict
        Blob Description Dictionary
    attr : str
        Attribute key

    Returns
    -------
    ret : Attribute Value
    """
    try:
        value = blobdict["BLOB"]["@" + attr]
    except KeyError:
        raise KeyError(
            "Attribute @"
            + attr
            + " is missing from Blob."
            + "There may be some problems with your file"
        )

    return value


def get_rb_blob_data(datastring, blobid):
    """Read BLOB data from datastring and return it

    Parameters
    ----------
    datastring : str
        Blob Description String
    blobid : int
        Number of requested blob

    Returns
    -------
    data : str
        Content of blob
    """
    xmltodict = util.import_optional("xmltodict")

    start = 0
    search_string = '<BLOB blobid="{0}"'.format(blobid)
    start = datastring.find(search_string.encode(), start)
    if start == -1:
        raise EOFError("Blob ID {0} not found!".format(blobid))
    end = datastring.find(b">", start)
    xmlstring = datastring[start : end + 1]

    # cheat the xml parser by making xml well-known
    xmldict = xmltodict.parse(xmlstring.decode() + "</BLOB>")
    cmpr = get_rb_blob_attribute(xmldict, "compression")
    size = int(get_rb_blob_attribute(xmldict, "size"))
    data = datastring[end + 2 : end + 2 + size]  # read blob data to string

    # decompress if necessary
    # the first 4 bytes are neglected for an unknown reason
    if cmpr == "qt":
        data = decompress(data[4:])

    return data


def map_rb_data(data, datadepth):
    """Map BLOB data to correct DataWidth and Type and convert it
    to numpy array

    Parameters
    ----------
    data : str
        Blob Data
    datadepth : int
        bit depth of Blob data

    Returns
    -------
    data : :py:class:`numpy:numpy.ndarray`
        Content of blob
    """
    flagdepth = None
    if datadepth < 8:
        flagdepth = datadepth
        datadepth = 8

    datawidth, datatype = get_rb_data_layout(datadepth)

    # import from data buffer well aligned to data array
    data = np.ndarray(shape=(int(len(data) / datawidth),), dtype=datatype, buffer=data)

    if flagdepth:
        data = np.unpackbits(data)

    return data


def get_rb_data_shape(blobdict):
    """Retrieve correct BLOB data shape from blobdict

    Parameters
    ----------
    blobdict : dict
        Blob Description Dict

    Returns
    -------
    shape : tuple
        shape of data
    """
    # this is a bit hacky, but we do not know beforehand,
    # so we extract this on the run
    try:
        dim0 = get_rb_data_attribute(blobdict, "rays")
        dim1 = get_rb_data_attribute(blobdict, "bins")
        # if rays and bins are found, return both
        return dim0, dim1
    except KeyError as e1:
        try:
            # if only rays is found, return rays
            return dim0
        except UnboundLocalError:
            try:
                # if both rays and bins not found assuming pixmap
                dim0 = get_rb_data_attribute(blobdict, "rows")
                dim1 = get_rb_data_attribute(blobdict, "columns")
                dim2 = get_rb_data_attribute(blobdict, "depth")
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


def get_rb_blob_from_string(datastring, blobdict):
    """Read BLOB data from datastring and return it as numpy array with correct
    dataWidth and shape

    Parameters
    ----------
    datastring : str
        Blob Description String
    blobdict : dict
        Blob Description Dict

    Returns
    -------
    data : :py:class:`numpy:numpy.ndarray`
        Content of blob as numpy array
    """

    blobid = get_rb_data_attribute(blobdict, "blobid")
    data = get_rb_blob_data(datastring, blobid)

    # map data to correct datatype and width
    datadepth = get_rb_data_attribute(blobdict, "depth")
    data = map_rb_data(data, datadepth)

    # reshape data
    data.shape = get_rb_data_shape(blobdict)

    return data


def get_rb_blob_from_file(name, blobdict):
    """Read BLOB data from file and return it with correct
    dataWidth and shape

    Parameters
    ----------
    name : str or file-like
        Path to Rainbow file or file-like object
    blobdict : dict
        Blob Dict

    Returns
    -------
    data : :py:class:`numpy:numpy.ndarray`
        Content of blob as numpy array
    """
    with util._open_file(name) as f:
        datastring = f.read()

    data = get_rb_blob_from_string(datastring, blobdict)

    return data


def get_rb_file_as_string(fid):
    """Read Rainbow File Contents in data_string

    Parameters
    ----------
    fid : object
        File handle of Data File

    Returns
    -------
    data_string : str
        File Contents as data_string
    """

    try:
        data_string = fid.read()
    except Exception:
        raise IOError("Could not read from file handle")

    return data_string


def get_rb_blobs_from_file(fid, rbdict):
    """Read all BLOBS found in given nested dict, loads them from file
    given by filename and add them to the dict at the appropriate position.

    Parameters
    ----------
    fid : object
        File handle of Data File
    rbdict : dict
        Rainbow file Contents

    Returns
    -------
    ret : dict
        Rainbow File Contents
    """

    blobs = list(find_key("@blobid", rbdict))

    datastring = get_rb_file_as_string(fid)
    for blob in blobs:
        data = get_rb_blob_from_string(datastring, blob)
        blob["data"] = data

    return rbdict


def get_rb_header(fid):
    """Read Rainbow Header from filename, converts it to a dict and returns it

    Parameters
    ----------
    fid : object
        File handle of Data File

    Returns
    -------
    object : dict
        Rainbow File Contents
    """

    # load the header lines, i.e. the XML part
    end_xml_marker = b"<!-- END XML -->"
    header = b""
    line = b""

    while not line.startswith(end_xml_marker):
        header += line[:-1]
        line = fid.readline()
        if len(line) == 0:
            raise IOError("WRADLIB: Rainbow Fileheader Corrupt")

    xmltodict = util.import_optional("xmltodict")

    return xmltodict.parse(header)


def read_rainbow(filename, loaddata=True):
    """Reads Rainbow files files according to their structure

    In contrast to other file readers under :mod:`wradlib.io`, this function
    will *not* return a two item tuple with (data, metadata). Instead, this
    function returns ONE dictionary that contains all the file contents - both
    data and metadata. The keys of the output dictionary conform to the XML
    outline in the original data file.
    The radar data will be extracted from the data blobs, converted and added
    to the dict with key 'data' at the place where the @blobid was pointing
    from.

    Parameters
    ----------
    filename : str or file-like
        a rainbow file path or file-like object of rainbow file
    loaddata : bool
        Defaults to True. If False function returns only metadata.

    Returns
    -------
    rbdict : dict
        a dictionary that contains both data and metadata according to the
        original rainbow file structure

    Examples
    --------
    See :ref:`/notebooks/fileio/wradlib_load_rainbow_example.ipynb`.
    """
    with util._open_file(filename) as f:
        rbdict = get_rb_header(f)
        if loaddata:
            rbdict = get_rb_blobs_from_file(f, rbdict)

    return rbdict
