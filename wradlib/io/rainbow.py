#!/usr/bin/env python
# Copyright (c) 2011-2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Read Rainbow
^^^^^^^^^^^^
.. autosummary::
   :nosignatures:
   :toctree: generated/

   read_Rainbow
"""

# standard libraries
from __future__ import absolute_import
import sys

import numpy as np
from .. import util as util


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
    """Retrieve correct BLOB data shape from blobdict

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
    """Read BLOB data from datastring and return it as numpy array with correct
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
    """Read BLOB data from file and return it with correct
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
    except Exception:
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
    except Exception:
        raise IOError('Could not read from file handle')

    xmltodict = util.import_optional('xmltodict')

    return xmltodict.parse(header)


def read_Rainbow(f, loaddata=True):
    """Reads Rainbow files files according to their structure

    .. versionchanged 0.10.0
       Added reading from file handles.

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
