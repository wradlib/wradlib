#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Rainbow Data I/O
^^^^^^^^^^^^^^^^

Reads data from Leonardo's Rainbow5 data formats. Former available code was ported to
`xradar <https://xradar.rtfd.io>`__-package and is imported from there.

:func:`~wradlib.io.rainbow.read_rainbow` reads all data and metadata into a dictionary.
Reading sweep data can be skipped by setting `loaddata=False`.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["read_rainbow"]
__doc__ = __doc__.format("\n   ".join(__all__))


from xradar.io.backends import rainbow as xrainbow

from wradlib import util


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

    data = xrainbow.get_rb_blob_from_string(datastring, blobdict)

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
    except Exception as err:
        raise OSError("Could not read from file handle.") from err

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

    blobs = list(xrainbow.find_key("@blobid", rbdict))

    datastring = get_rb_file_as_string(fid)
    for blob in blobs:
        data = xrainbow.get_rb_blob_from_string(datastring, blob)
        blob["data"] = data

    return rbdict


def read_rainbow(filename, *, loaddata=True):
    """Reads Rainbow files according to their structure

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
    See :ref:`/notebooks/fileio/legacy/read_rainbow.ipynb`.
    """
    with util._open_file(filename) as f:
        rbdict = xrainbow.get_rb_header(f)
        if loaddata:
            rbdict = get_rb_blobs_from_file(f, rbdict)

    return rbdict
