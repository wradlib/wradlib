#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Rainbow Data I/O
^^^^^^^^^^^^^^^^

Reads data from Leonardo's Rainbow5 data formats

:func:`~wradlib.io.rainbow.read_rainbow` reads all data and metadata into a dictionary.
Reading sweep data can be skipped by setting `loaddata=False`.

:func:`~wradlib.io.rainbow.open_rainbow_dataset` and :func:`~wradlib.io.rainbow.open_rainbow_mfdataset`
read Rainbow5 data into xarray Datasets with a CfRadial2-like structure.
For this `mmap.mmap` is utilized.


.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["read_rainbow", "open_rainbow_dataset", "open_rainbow_mfdataset"]
__doc__ = __doc__.format("\n   ".join(__all__))

import datetime as dt
import sys

import numpy as np

from wradlib import util
from wradlib.io.xarray import (
    open_radar_dataset,
    open_radar_mfdataset,
    raise_on_missing_xarray_backend,
)


def _get_dict_value(d, k1, k2):
    v = d.get(k1, None)
    if v is None:
        v = d[k2]
    return v


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
            f"Wrong DataDepth: {datadepth}. Conversion only for depth 8, 16, 32."
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
            f"Attribute @{attr} is missing from "
            "Blob Description. There may be some "
            "problems with your file"
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
    search_string = f'<BLOB blobid="{blobid}"'
    start = datastring.find(search_string.encode(), start)
    if start == -1:
        raise EOFError(f"Blob ID {blobid} not found!")
    end = datastring.find(b">", start)
    xmlstring = datastring[start : end + 1]

    # cheat the xml parser by making xml well-known
    xmldict = xmltodict.parse(xmlstring.decode() + "</BLOB>")
    cmpr = get_rb_blob_attribute(xmldict, "compression")
    size = int(get_rb_blob_attribute(xmldict, "size"))
    data = datastring[end + 2 : end + 2 + size]  # read blob data to string

    # decompress if necessary
    if cmpr == "qt":
        # the first 4 bytes contain the uncompressed size in big endian
        usize = int.from_bytes(data[:4], "big")
        data = decompress(data[4:])
        if len(data) != usize:
            raise ValueError(
                f"Data size mismatch. {usize} bytes expected, "
                f"{len(data)} bytes read."
            )

    return data


def map_rb_data(data, datadepth, datashape=0):
    """Map BLOB data to correct DataWidth and Type and convert it
    to numpy array

    Parameters
    ----------
    data : str
        Blob Data
    datadepth : int
        bit depth of Blob data
    datashape : tuple
        expected data shape, only used for the flags to truncate

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
        data = np.unpackbits(data)[0 : np.prod(datashape)]

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
    datashape = get_rb_data_shape(blobdict)
    data = map_rb_data(data, datadepth, datashape)

    # reshape data
    data.shape = datashape

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


class RainbowFileBase:
    """Base class for Rainbow Files."""

    def __init__(self, **kwargs):
        super().__init__()


class RainbowFile(RainbowFileBase):
    """RainbowFile class"""

    def __init__(self, filename, **kwargs):
        self._debug = kwargs.get("debug", False)
        self._rawdata = kwargs.get("rawdata", False)
        self._loaddata = kwargs.get("loaddata", True)

        self._fp = None
        self._filename = filename
        if isinstance(filename, str):
            self._fp = open(filename, "rb")
            import mmap

            self._fh = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            raise TypeError(
                "Rainbow5 reader currently doesn't support file-like objects"
            )
        self._data = None
        super().__init__(**kwargs)
        # read rainbow header
        self._header = get_rb_header(self._fh)["volume"]
        self._coordinates = None
        slices = self._header["scan"]["slice"]
        if not isinstance(slices, list):
            slices = [slices]
        else:
            self._update_volume_slices()
        self._blobs = [list(find_key("@blobid", slc)) for slc in slices]
        if self._loaddata:
            for i, slc in enumerate(self._blobs):
                for blob in slc:
                    blobid = get_rb_data_attribute(blob, "blobid")
                    self.get_blob(blobid, i)

    def close(self):
        if self._fp is not None:
            self._fp.close()

    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def filename(self):
        return self._filename

    @property
    def version(self):
        return self._header["@version"]

    @property
    def type(self):
        return self._header["@type"]

    @property
    def datetime(self):
        return dt.datetime.strptime(self._header["@datetime"], "%Y-%m-%dT%H:%M:%S")

    @property
    def first_dimension(self):
        if self.type in ["vol", "azi"]:
            return "azimuth"
        elif self.type in ["ele"]:
            return "elevation"
        elif self.type in ["poi"]:
            raise NotImplementedError(
                "Rainbow5 data of type `poi` (pointmode) is currently not supported."
            )
        else:
            raise TypeError(f"Unknown Rainbow File Type: {self.type}")

    @property
    def header(self):
        return self._header

    @property
    def blobs(self):
        return self._blobs

    @property
    def slices(self):
        slices = self._header["scan"]["slice"]
        if not isinstance(slices, list):
            slices = [slices]
        return slices

    @property
    def pargroup(self):
        return self._header["scan"]["pargroup"]

    @property
    def sensorinfo(self):
        try:
            return self.header["sensorinfo"]
        except KeyError:
            return self.header.get("radarinfo", None)

    @property
    def history(self):
        return self.header.get("history", None)

    @property
    def site_coords(self):
        si = self.sensorinfo
        return (
            float(_get_dict_value(si, "lon", "@lon")),
            float(_get_dict_value(si, "lat", "@lat")),
            float(_get_dict_value(si, "alt", "@alt")),
        )

    def _get_rbdict_value(self, rbdict, name, dtype=None, default=None):
        value = rbdict.get(name, None)
        if value is None:
            value = self.pargroup.get(name, default)
        if dtype is not None:
            value = dtype(value)
        return value

    def _update_volume_slices(self):
        if isinstance(self._header["scan"]["slice"], list):
            slice0 = self._header["scan"]["slice"][0]
            for i, slice in enumerate(self._header["scan"]["slice"][1:]):
                newdict = dict(list(slice0.items()) + list(slice.items()))
                self._header["scan"]["slice"][i + 1] = newdict

    def get_blob(self, blobid, slc):
        self._fh.seek(0)
        blob = next(filter(lambda x: int(x["@blobid"]) == blobid, self._blobs[slc]))
        if blob.get("data", False) is False:
            data = get_rb_blob_from_string(self._fh, blob)
            # azimuth
            if blob.get("@refid", "") in ["startangle", "stopangle"]:
                # anglestep = self._get_rbdict_value(self.slices[slc], "anglestep", None, float)
                # anglestep = self.slices[slc].get("anglestep", None)
                # if anglestep is None:
                #     anglestep = self.pargroup["anglestep"]
                # anglestep = float(anglestep)
                # todo: correctly decode elevation angles
                #   elevation can decode negative values
                data = data * 360.0 / 2 ** float(blob["@depth"])
            blob["data"] = data


def open_rainbow_dataset(filename_or_obj, group=None, **kwargs):
    """Open and decode an RAINBOW5 radar sweep or volume from a file or file-like object.

    This function uses :func:`~wradlib.io.open_radar_dataset`` under the hood.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file and opened with an appropriate engine.
    group : str, optional
        Path to a sweep group in the given file to open.

    Keyword Arguments
    -----------------
    **kwargs : dict, optional
        Additional arguments passed on to :py:func:`xarray.open_dataset`.

    Returns
    -------
    dataset : :py:class:`xarray:xarray.Dataset` or :class:`wradlib.io.xarray.RadarVolume`
        The newly created radar dataset or radar volume.

    See Also
    --------
    :func:`~wradlib.io.rainbow.open_rainbow_dataset`
    """
    raise_on_missing_xarray_backend()
    kwargs["group"] = group
    return open_radar_dataset(filename_or_obj, engine="rainbow", **kwargs)


def open_rainbow_mfdataset(filename_or_obj, group=None, **kwargs):
    """Open and decode an RAINBOW5 radar sweep or volume from a file or file-like object.

    This function uses :func:`~wradlib.io.xarray.open_radar_mfdataset` under the hood.
    Needs `dask` package to be installed.

    Parameters
    ----------
    filename_or_obj : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a local or remote
        radar file and opened with an appropriate engine.
    group : str, optional
        Path to a sweep group in the given file to open.

    Keyword Arguments
    -----------------
    reindex_angle : bool or float
        Defaults to None (reindex angle with tol=0.4deg). If given a floating point
        number, it is used as tolerance. If False, no reindexing is performed.
        Only invoked if `decode_coord=True`.
    **kwargs : dict, optional
        Additional arguments passed on to :py:func:`xarray:xarray.open_dataset`.

    Returns
    -------
    dataset : :py:class:`xarray:xarray.Dataset` or :class:`wradlib.io.xarray.RadarVolume`
        The newly created radar dataset or radar volume.

    See Also
    --------
    :func:`~wradlib.io.rainbow.open_rainbow_dataset`
    """
    raise_on_missing_xarray_backend()
    kwargs["group"] = group
    return open_radar_mfdataset(filename_or_obj, engine="rainbow", **kwargs)
