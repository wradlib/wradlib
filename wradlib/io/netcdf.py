#!/usr/bin/env python
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Read NetCDF
^^^^^^^^^^^
.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["read_edge_netcdf", "read_generic_netcdf"]
__doc__ = __doc__.format("\n   ".join(__all__))

import collections
import contextlib
import datetime as dt

import netCDF4 as nc
import numpy as np


@contextlib.contextmanager
def _open_netcdf(filename):
    if isinstance(filename, str):
        yield nc.Dataset(filename)
    else:
        yield nc.Dataset("name", mode="r", memory=filename.read())


def read_edge_netcdf(filename, enforce_equidist=False):
    """Data reader for netCDF files exported by the EDGE radar software

    The corresponding NetCDF files from the EDGE software typically contain
    only one variable (e.g. reflectivity) for one elevation angle (sweep).
    The elevation angle is specified in the attributes keyword "Elevation".
    Please note that the radar might not return data with equidistant azimuth
    angles. In case you need equidistant azimuth angles, please set
    enforce_equidist to True.

    Parameters
    ----------
    filename : string or file-like
        path of the netCDF file or file-like object
    enforce_equidist : boolean
        Set True if the values of the azimuth angles should be forced to be
        equidistant; default value is False

    Returns
    -------
    output : :func:`numpy:numpy.array`
        of image data (dBZ), dictionary of attributes
    """
    with _open_netcdf(filename) as dset:
        data = dset.variables[dset.TypeName][:]
        # Check azimuth angles and rotate image
        az = dset.variables["Azimuth"][:]
        # These are the indices of the minimum and maximum azimuth angle
        ix_minaz = np.argmin(az)
        ix_maxaz = np.argmax(az)
        if enforce_equidist:
            az = np.linspace(
                np.round(az[ix_minaz], 2), np.round(az[ix_maxaz], 2), len(az)
            )
        else:
            az = np.roll(az, -ix_minaz)
        # rotate accordingly
        data = np.roll(data, -ix_minaz, axis=0)
        data = np.where(data == dset.getncattr("MissingData"), np.nan, data)
        # Ranges
        binwidth = (dset.getncattr("MaximumRange-value") * 1000.0) / len(
            dset.dimensions["Gate"]
        )
        r = np.arange(
            binwidth,
            (dset.getncattr("MaximumRange-value") * 1000.0) + binwidth,
            binwidth,
        )
        # collect attributes
        attrs = {}
        for attrname in dset.ncattrs():
            attrs[attrname] = dset.getncattr(attrname)
        # # Limiting the returned range
        # if range_lim and range_lim / binwidth <= data.shape[1]:
        #     data = data[:,:range_lim / binwidth]
        #     r = r[:range_lim / binwidth]
        # Set additional metadata attributes
        attrs["az"] = az
        attrs["r"] = r
        attrs["sitecoords"] = (attrs["Longitude"], attrs["Latitude"], attrs["Height"])
        attrs["time"] = dt.datetime.utcfromtimestamp(attrs.pop("Time"))
        attrs["max_range"] = data.shape[1] * binwidth

    return data, attrs


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
    out = collections.OrderedDict()

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
        dim = collections.OrderedDict()
        for k, v in ncid.dimensions.items():
            tmp = collections.OrderedDict()
            try:
                tmp["data_model"] = v._data_model
            except AttributeError:
                pass
            try:
                tmp["size"] = v.__len__()
            except AttributeError:
                pass
            tmp["dimid"] = v._dimid
            dimids = np.append(dimids, v._dimid)
            tmp["grpid"] = v._grpid
            tmp["isunlimited"] = v.isunlimited()
            dim[k] = tmp
        # Usually, the dimensions should be ordered by dimid automatically
        # in case netcdf used OrderedDict. However, we should double check
        if np.array_equal(dimids, np.sort(dimids)):
            # is already sorted
            out["dimensions"] = dim
        else:
            # need to sort
            dim2 = collections.OrderedDict()
            keys = dim.keys()
            for dimid in np.sort(dimids):
                dim2[keys[dimid]] = dim[keys[dimid]]
            out["dimensions"] = dim2

    # variables
    if ncid.variables:
        var = collections.OrderedDict()
        for k, v in ncid.variables.items():
            tmp = collections.OrderedDict()
            for k1 in v.ncattrs():
                tmp[k1] = v.getncattr(k1)
            if v[:].dtype.kind == "S":
                try:
                    tmp["data"] = nc.chartostring(v[:])
                except Exception:
                    tmp["data"] = v[:]
            else:
                tmp["data"] = v[:]
            var[k] = tmp
        out["variables"] = var

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
    fname : string or file-like
        a netcdf file path or file-like object

    Returns
    -------
    out : ordered dict
        an ordered dictionary that contains both data and metadata according
        to the original netcdf file structure

    Examples
    --------
    See :ref:`/notebooks/fileio/wradlib_generic_netcdf_example.ipynb`.
    """
    with _open_netcdf(fname) as ncid:
        out = read_netcdf_group(ncid)

    return out
