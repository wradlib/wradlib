#!/usr/bin/env python
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.


"""
Xarray based Data I/O (deprecated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reads data from netcdf-based CfRadial1, CfRadial2 and hdf5-based ODIM_H5 and
other hdf5-flavours (GAMIC).

This reader implementation uses

* `xarray <https://xarray.pydata.org/>`_,
* `netcdf4 <https://unidata.github.io/netcdf4-python/>`_,
* `h5py <https://www.h5py.org/>`_ and
* `h5netcdf <https://github.com/h5netcdf/h5netcdf>`_.

The following two approaches are not recommended and will be removed from the codebase
in wradlib v 2.0.0.

In the first approach the data is claimed using netcdf4-Dataset in a diskless
non-persistent mode as follows::

    vol = io.xarray.OdimH5(ncfile)
    # or
    vol = io.xarray.CfRadial(ncfile)

The vol data structure holds one or many ['sweep_X'] xarray datasets, containing the
sweep data. The root group xarray dataset which corresponds to the
CfRadial2 root-group is available via the `.root`-object.

The writer implementation uses xarray for CfRadial2 output and relies on h5py
for the ODIM_H5 output.

The second approach reads ODIM files (metadata) into a *simple* accessible
structure::

    vol = wradlib.io.open_odim(paths, loader='netcdf4', **kwargs)

All datafiles are accessed via the given loader ('netcdf4', 'h5py',
'h5netcdf'). Only absolutely necessary data is actually read in this process,
eg. acquisition time and elevation, to fill the structure accordingly. All
subsequent metadata retrievals are cached to further improve performance.
Actual data access is realised via xarray using engine 'netcdf4' or 'h5netcdf',
depending on the loader.

Since for data handling xarray is utilized all xarray features can be
exploited, like lazy-loading, pandas-like indexing on N-dimensional data and
vectorized mathematical operations across multiple dimensions.

Examples
--------
    See :ref:`/notebooks/fileio/wradlib_odim_multi_file_dataset.ipynb`.

Warning
-------
    This implementation is considered experimental. Changes in the API should
    be expected.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "XRadVol",
    "CfRadial",
    "OdimH5",
    "open_odim",
    "XRadSweep",
    "XRadMoment",
    "XRadTimeSeries",
    "XRadVolume",
    "create_xarray_dataarray",
]
__doc__ = __doc__.format("\n   ".join(__all__))

import collections
import datetime as dt
import glob
import os
import warnings

import dateutil
import deprecation
import numpy as np
import xarray as xr
from packaging.version import Version
from xarray.backends.api import combine_by_coords

from wradlib import version
from wradlib.georef import xarray as geo_xarray
from wradlib.io.xarray import (
    GAMIC_NAMES,
    XRadBase,
    _fix_angle,
    _maybe_decode,
    _reindex_angle,
    _write_odim,
    _write_odim_dataspace,
    az_attrs_template,
    cf_full_vars,
    el_attrs_template,
    global_attrs,
    global_variables,
    h5netcdf,
    h5py,
    moment_attrs,
    moments_mapping,
    netCDF4,
    range_attrs,
    root_vars,
    sweep_vars1,
    sweep_vars2,
    sweep_vars3,
    time_attrs,
    to_netcdf,
)
from wradlib.util import has_import

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(val, **kwargs):
        print(
            "wradlib: Please wait for completion of time consuming task! \n"
            "wradlib: Please install 'tqdm' for showing a progress bar "
            "instead."
        )
        return val


@deprecation.deprecated(
    deprecated_in="1.5",
    removed_in="2.0",
    current_version=version.version,
    details="Use `wradlib.georef.create_xarray_dataarray` " "instead.",
)
def create_xarray_dataarray(*args, **kwargs):
    return geo_xarray.create_xarray_dataarray(*args, **kwargs)


@xr.register_dataset_accessor("gamic")
class GamicAccessor:
    """Dataset Accessor for handling GAMIC HDF5 data files"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._radial_range = None
        self._azimuth_range = None
        self._elevation_range = None
        self._time_range = None
        self._sitecoords = None
        self._polcoords = None
        self._projection = None
        self._time = None

    @property
    def radial_range(self):
        """Return the radial range of this dataset."""
        if self._radial_range is None:
            ngates = self._obj.attrs["bin_count"]
            # range_start = self._obj.attrs['range_start']
            range_samples = self._obj.attrs["range_samples"]
            range_step = self._obj.attrs["range_step"]
            bin_range = range_step * range_samples
            range_data = np.arange(
                bin_range / 2.0, bin_range * ngates, bin_range, dtype="float32"
            )
            range_attrs["meters_to_center_of_first_gate"] = bin_range / 2.0
            da = xr.DataArray(range_data, dims=["dim_1"], attrs=range_attrs)
            self._radial_range = da
        return self._radial_range

    @property
    def azimuth_range(self):
        """Return the azimuth range of this dataset."""
        if self._azimuth_range is None:
            azstart = self._obj["azimuth_start"]
            azstop = self._obj["azimuth_stop"]
            zero_index = np.where(azstop < azstart)
            azstop[zero_index[0]] += 360
            azimuth = (azstart + azstop) / 2.0
            azimuth = azimuth.assign_attrs(az_attrs_template.copy())
            self._azimuth_range = azimuth
        return self._azimuth_range

    @property
    def elevation_range(self):
        """Return the elevation range of this dataset."""
        if self._elevation_range is None:
            elstart = self._obj["elevation_start"]
            elstop = self._obj["elevation_stop"]
            elevation = (elstart + elstop) / 2.0
            elevation = elevation.assign_attrs(el_attrs_template.copy())
            self._elevation_range = elevation
        return self._elevation_range

    @property
    def time_range(self):
        """Return the time range of this dataset."""
        if self._time_range is None:
            times = self._obj["timestamp"] / 1e6
            attrs = {
                "units": "seconds since 1970-01-01T00:00:00Z",
                "standard_name": "time",
            }
            da = xr.DataArray(times, attrs=attrs)
            self._time_range = da
        return self._time_range


@xr.register_dataset_accessor("odim")
class OdimAccessor:
    """Dataset Accessor for handling ODIM_H5 data files"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._radial_range = None
        self._azimuth_range = None
        self._elevation_range = None
        self._time_range = None
        self._time_range2 = None
        self._prt = None
        self._n_samples = None

    @property
    def radial_range(self):
        """Return the radial range of this dataset."""
        if self._radial_range is None:
            ngates = self._obj.attrs["nbins"]
            range_start = self._obj.attrs["rstart"] * 1000.0
            bin_range = self._obj.attrs["rscale"]
            cent_first = range_start + bin_range / 2.0
            range_data = np.arange(
                cent_first, range_start + bin_range * ngates, bin_range, dtype="float32"
            )
            range_attrs["meters_to_center_of_first_gate"] = cent_first
            range_attrs["meters_between_gates"] = bin_range

            da = xr.DataArray(range_data, dims=["dim_1"], attrs=range_attrs)
            self._radial_range = da
        return self._radial_range

    @property
    def azimuth_range2(self):
        """Return the azimuth range of this dataset."""
        if self._azimuth_range is None:
            nrays = self._obj.attrs["nrays"]
            res = 360.0 / nrays
            azimuth_data = np.arange(res / 2.0, 360.0, res, dtype="float32")

            da = xr.DataArray(
                azimuth_data, dims=["dim_0"], attrs=az_attrs_template.copy()
            )
            self._azimuth_range = da
        return self._azimuth_range

    @property
    def azimuth_range(self):
        """Return the azimuth range of this dataset."""
        if self._azimuth_range is None:
            startaz = self._obj.attrs["startazA"]
            stopaz = self._obj.attrs["stopazA"]
            zero_index = np.where(stopaz < startaz)
            stopaz[zero_index[0]] += 360
            azimuth_data = (startaz + stopaz) / 2.0
            da = xr.DataArray(azimuth_data, attrs=az_attrs_template.copy())
            self._azimuth_range = da
        return self._azimuth_range

    @property
    def elevation_range2(self):
        """Return the elevation range of this dataset."""
        if self._elevation_range is None:
            nrays = self._obj.attrs["nrays"]
            elangle = self._obj.attrs["elangle"]
            elevation_data = np.ones(nrays, dtype="float32") * elangle
            da = xr.DataArray(
                elevation_data, dims=["dim_0"], attrs=el_attrs_template.copy()
            )
            self._elevation_range = da
        return self._elevation_range

    @property
    def elevation_range(self):
        """Return the elevation range of this dataset."""
        if self._elevation_range is None:
            startel = self._obj.attrs["startelA"]
            stopel = self._obj.attrs["stopelA"]
            elevation_data = (startel + stopel) / 2.0
            da = xr.DataArray(
                elevation_data, dims=["dim_0"], attrs=el_attrs_template.copy()
            )
            self._elevation_range = da
        return self._elevation_range

    @property
    def time_range(self):
        """Return the time range of this dataset."""
        if self._time_range is None:
            startT = self._obj.attrs["startazT"]
            stopT = self._obj.attrs["stopazT"]
            times = (startT + stopT) / 2.0
            self._time_range = times

        return self._time_range

    @property
    def time_range2(self):
        """Return the time range of this dataset."""
        if self._time_range2 is None:
            startdate = self._obj.attrs["startdate"]
            starttime = self._obj.attrs["starttime"]
            enddate = self._obj.attrs["enddate"]
            endtime = self._obj.attrs["endtime"]

            start = dt.datetime.strptime(startdate + starttime, "%Y%m%d%H%M%S")
            end = dt.datetime.strptime(enddate + endtime, "%Y%m%d%H%M%S")
            start = start.replace(tzinfo=dt.timezone.utc)
            end = end.replace(tzinfo=dt.timezone.utc)

            self._time_range2 = (start.timestamp(), end.timestamp())
        return self._time_range2

    @property
    def prt(self):
        if self._prt is None:
            try:
                prt = 1.0 / self._obj.attrs["prf"]
                da = xr.DataArray(prt, dims=["dim_0"])
                self._prt = da
            except KeyError:
                pass
        return self._prt

    @property
    def n_samples(self):
        if self._n_samples is None:
            try:
                da = xr.DataArray(self._obj.attrs["pulse"], dims=["dim_0"])
                self._n_samples = da
            except KeyError:
                pass
        return self._n_samples


def to_cfradial2(volume, filename, timestep=None, engine=None):
    """Save RadarVolume/XRadVol/XRadVolume to CfRadial2.0 compliant file.

    Parameters
    ----------
    volume : :class:`wradlib.io.xarray.RadarVolume`, :class:`wradlib.io.xarray.XRadVol` or :class:`wradlib.io.xarray.XRadVolume`
    filename : str
        output filename
    timestep : int
        timestep of wanted volume
    """
    if engine is None:
        if has_import(netCDF4):
            engine == "netcdf4"
        elif has_import(h5netcdf):
            engine == "h5netcdf"
        else:
            raise ImportError(
                "wradlib: ``netCDF4`` or ``h5netcdf`` needed to perform this operation."
            )
    volume.root.load()
    root = volume.root.copy(deep=True)
    root.attrs["Conventions"] = "Cf/Radial"
    root.attrs["version"] = "2.0"
    root.to_netcdf(filename, mode="w", group="/", engine=engine)
    for idx, key in enumerate(root.sweep_group_name.values):
        if isinstance(volume, (OdimH5, CfRadial)):
            swp = volume[key]
        elif isinstance(volume, XRadVolume):
            swp = volume[idx][timestep].data
        else:
            ds = volume[idx]
            if "time" not in ds.dims:
                ds = ds.expand_dims("time")
            swp = ds.isel(time=timestep)
        swp.load()
        dim0 = list(set(swp.dims) & {"azimuth", "elevation", "time"})[0]
        try:
            swp = swp.swap_dims({dim0: "time"})
        except ValueError:
            swp = swp.drop_vars("time").rename({"rtime": "time"})
            swp = swp.swap_dims({dim0: "time"})
        swp = swp.drop_vars(["x", "y", "z", "gr", "rays", "bins"], errors="ignore")
        swp = swp.sortby("time")
        swp.to_netcdf(filename, mode="a", group=key, engine=engine)


def to_odim(volume, filename, timestep=0):
    """Save RadarVolume/XRadVol/XRadVolume to ODIM_H5/V2_2 compliant file.

    Parameters
    ----------
    volume : :class:`wradlib.io.xarray.RadarVolume`, :class:`wradlib.io.xarray.XRadVol` or :class:`wradlib.io.xarray.XRadVolume`
    filename : str
        output filename
    timestep : int
        timestep of wanted volume
    """
    root = volume.root

    h5 = h5py.File(filename, "w")

    # root group, only Conventions for ODIM_H5
    _write_odim({"Conventions": "ODIM_H5/V2_2"}, h5)

    # how group
    how = {}
    how.update({"_modification_program": "wradlib"})

    h5_how = h5.create_group("how")
    _write_odim(how, h5_how)

    sweepnames = root.sweep_group_name.values

    # what group, object, version, date, time, source, mandatory
    # p. 10 f
    what = {}
    if len(sweepnames) > 1:
        what["object"] = "PVOL"
    else:
        what["object"] = "SCAN"
    what["version"] = "H5rad 2.2"
    what["date"] = str(root.time_coverage_start.values)[:10].replace("-", "")
    what["time"] = str(root.time_coverage_end.values)[11:19].replace(":", "")
    what["source"] = root.attrs["instrument_name"]

    h5_what = h5.create_group("what")
    _write_odim(what, h5_what)

    # where group, lon, lat, height, mandatory
    where = {
        "lon": root.longitude.values,
        "lat": root.latitude.values,
        "height": root.altitude.values,
    }
    h5_where = h5.create_group("where")
    _write_odim(where, h5_where)

    # datasets
    ds_list = [f"dataset{i + 1}" for i in range(len(sweepnames))]
    ds_idx = np.argsort(ds_list)
    for idx in ds_idx:
        if isinstance(volume, (OdimH5, CfRadial)):
            ds = volume[f"sweep_{idx + 1}"]
        elif isinstance(volume, XRadVolume):
            ds = volume[idx][timestep].data
            ds = ds.drop_vars("time", errors="ignore").rename({"rtime": "time"})
        else:
            ds = volume[idx]
            if "time" not in ds.dims:
                ds = ds.expand_dims("time")
            ds = ds.isel(time=timestep, drop=True)
            ds = ds.drop_vars("time", errors="ignore").rename({"rtime": "time"})
        h5_dataset = h5.create_group(ds_list[idx])

        # what group p. 21 ff.
        h5_ds_what = h5_dataset.create_group("what")
        ds_what = {}
        # skip NaT values
        valid_times = ~np.isnat(ds.time.values)
        t = sorted(ds.time.values[valid_times])
        start = dt.datetime.utcfromtimestamp(np.rint(t[0].astype("O") / 1e9))
        end = dt.datetime.utcfromtimestamp(np.rint(t[-1].astype("O") / 1e9))
        ds_what["product"] = "SCAN"
        ds_what["startdate"] = start.strftime("%Y%m%d")
        ds_what["starttime"] = start.strftime("%H%M%S")
        ds_what["enddate"] = end.strftime("%Y%m%d")
        ds_what["endtime"] = end.strftime("%H%M%S")
        _write_odim(ds_what, h5_ds_what)

        # where group, p. 11 ff. mandatory
        h5_ds_where = h5_dataset.create_group("where")
        rscale = ds.range.values[1] / 1.0 - ds.range.values[0]
        rstart = (ds.range.values[0] - rscale / 2.0) / 1000.0
        # todo: make this work for RHI's
        a1gate = np.argsort(ds.sortby("azimuth").time.values)[0]
        try:
            fixed_angle = ds.fixed_angle
        except AttributeError:
            fixed_angle = ds.elevation.round(decimals=1).median().values
        ds_where = {
            "elangle": fixed_angle,
            "nbins": ds.range.shape[0],
            "rstart": rstart,
            "rscale": rscale,
            "nrays": ds.azimuth.shape[0],
            "a1gate": a1gate,
        }
        _write_odim(ds_where, h5_ds_where)

        # how group, p. 14 ff.
        h5_ds_how = h5_dataset.create_group("how")
        tout = [tx.astype("O") / 1e9 for tx in ds.sortby("azimuth").time.values]
        tout_sorted = sorted(tout)

        # handle non-uniform times (eg. only second-resolution)
        if np.count_nonzero(np.diff(tout_sorted)) < (len(tout_sorted) - 1):
            tout = np.roll(
                np.linspace(tout_sorted[0], tout_sorted[-1], len(tout)), a1gate
            )
            tout_sorted = sorted(tout)

        difft = np.diff(tout_sorted) / 2.0
        difft = np.insert(difft, 0, difft[0])
        azout = ds.sortby("azimuth").azimuth
        diffa = np.diff(azout) / 2.0
        diffa = np.insert(diffa, 0, diffa[0])
        elout = ds.sortby("azimuth").elevation
        diffe = np.diff(elout) / 2.0
        diffe = np.insert(diffe, 0, diffe[0])
        try:
            sweep_number = ds.sweep_number + 1
        except AttributeError:
            sweep_number = timestep
        ds_how = {
            "scan_index": sweep_number,
            "scan_count": len(sweepnames),
            "startazT": tout - difft,
            "stopazT": tout + difft,
            "startazA": azout - diffa,
            "stopazA": azout + diffa,
            "startelA": elout - diffe,
            "stopelA": elout + diffe,
        }
        _write_odim(ds_how, h5_ds_how)

        # write moments
        _write_odim_dataspace(ds, h5_dataset)

    h5.close()


def _preprocess_moment(ds, mom, non_uniform_shape):

    attrs = mom._decode(ds.data.attrs)
    quantity = mom.quantity

    # extract and translate attributes to cf
    what = mom.what
    attrs["scale_factor"] = what["gain"]
    attrs["add_offset"] = what["offset"]
    attrs["_FillValue"] = what["nodata"]
    attrs["_Undetect"] = what["undetect"]

    if mom.parent.decode_coords:
        attrs["coordinates"] = "elevation azimuth range"

    # handle non-standard moment names
    try:
        mapping = moments_mapping[quantity]
    except KeyError:
        pass
    else:
        attrs.update({key: mapping[key] for key in moment_attrs})

    ds["data"] = ds["data"].assign_attrs(attrs)

    # fix dimensions
    dims = sorted(list(ds.dims.keys()), key=lambda x: int(x[len("phony_dim_") :]))

    ds = ds.rename(
        {"data": quantity, dims[0]: mom.parent._dim0[0], dims[1]: mom.parent._dim1}
    )

    # apply coordinates to dataset if source moments have different shapes
    # and correct for it
    if mom.parent.decode_coords & non_uniform_shape:
        coords = mom.parent._get_coords()
        ds = ds.assign_coords(coords.coords)
        if mom.parent._dim0[0] == "azimuth":
            ds = ds.sortby(mom.parent._dim0[0])
            ds = ds.pipe(_reindex_angle, mom.parent)

    return ds


def _open_mfmoments(
    moments,
    chunks=None,
    compat="no_conflicts",
    preprocess=None,
    engine=None,
    lock=None,
    data_vars="all",
    coords="minimal",
    parallel=False,
    **kwargs,
):
    """Open multiple OdimH5 moments as a single dataset.

    This is derived from xarray.open_mfdataset [1]

    Parameters
    ----------
    moments : sequence
        List of XRadSweep objects.
    chunks : int or dict, optional
        Chunk size
    preprocess : callable, optional
        If provided, call this function on each dataset prior to concatenation.
    engine : {'netcdf4', 'h5netcdf'}, optional
        Engine to use when reading files. Defaults to 'netcdf4'.
    lock : False or duck threading.Lock, optional
        Resource lock to use when reading data from disk. Only relevant when
        using dask or another form of parallelism. By default, appropriate
        locks are chosen to safely read and write files with the currently
        active dask scheduler.
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        These data variables will be concatenated together:
          * 'minimal': Only data variables in which the dimension already
            appears are included.
          * 'different': Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * 'all': All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the 'minimal' data variables.
    parallel : bool, optional
        If True, the open and preprocess steps of this function will be
        performed in parallel using ``dask.delayed``. Default is False.
    **kwargs : dict, optional
        Additional arguments passed on to :py:func:`xarray:xarray.open_dataset`.

    Returns
    -------
    xarray.Dataset

    References
    ----------

    .. [1] https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html

    """  # noqa

    open_kwargs = dict(chunks=chunks, **kwargs)

    # if moments are specified in XRadTimeseries only load those
    if moments.parent._moments is not None:
        moments = [p for p in moments if p.quantity in moments.parent._moments]

    engine = moments[0].engine
    if engine == "netcdf4":
        opener = netCDF4.Dataset
        opener_kwargs = {}
        store = xr.backends.NetCDF4DataStore
    else:
        if Version(h5netcdf.__version__) < Version("0.8.0"):
            warnings.warn(
                f"WRADLIB: 'h5netcdf>=0.8.0' needed to perform this "
                f"operation. 'h5netcdf={h5netcdf.__version__} "
                f"available.",
                UserWarning,
            )
            return None
        if Version(xr.__version__) < Version("0.15.0"):
            warnings.warn(
                f"WRADLIB: 'xarray>=0.15.0' needed to perform this "
                f"operation. 'xarray={xr.__version__} "
                f"available.",
                UserWarning,
            )
            return None
        opener = h5netcdf.File
        opener_kwargs = {"phony_dims": "access"}
        store = xr.backends.H5NetCDFStore

    # do not use parallel if all moments in one file
    if len(set([p.filename for p in moments])) == 1:
        single_file = True
        if os.path.isfile(moments[0].filename):
            ds0 = opener(moments[0].filename, "r", **opener_kwargs)
        else:
            ds0 = moments[0].ncfile
    else:
        single_file = False

    if parallel:
        import dask

        # wrap the open_dataset, getattr, and preprocess with delayed
        open_ = dask.delayed(xr.open_dataset)
        getattr_ = dask.delayed(getattr)
        if preprocess is not None:
            preprocess = dask.delayed(preprocess)
    else:
        open_ = xr.open_dataset
        getattr_ = getattr

    if single_file:
        datasets = [open_(store(ds0, group=p.ncpath)) for p in moments]
    else:
        fileid = []
        for p in moments:
            if os.path.isfile(p.filename):
                p = opener(p.filename, "r", **opener_kwargs)
            else:
                p = p.ncfile
            fileid.append(p)
        datasets = [
            open_(store(f, group=p.ncpath), **open_kwargs)
            for f, p in zip(fileid, moments)
        ]
    if Version(xr.__version__) <= Version("0.16.2"):
        closers = [getattr_(ds, "_file_obj") for ds in datasets]
    else:
        closers = [getattr_(ds, "_close") for ds in datasets]

    # check for differences in shape of moments
    non_uniform_shape = len(set([tuple(ds.sizes.values()) for ds in datasets])) > 1
    if preprocess is not None:
        datasets = [
            preprocess(ds, mom, non_uniform_shape) for ds, mom in zip(datasets, moments)
        ]

    if parallel:
        # calling compute here will return the datasets/file_objs lists,
        # the underlying datasets will still be stored as dask arrays
        datasets, closers = dask.compute(datasets, closers)

    # Combine all datasets, closing them in case of a ValueError
    try:
        combined = combine_by_coords(
            datasets, compat="no_conflicts", data_vars="all", coords="minimal"
        )
    except ValueError:
        for ds in datasets:
            ds.close()
        raise

    def multi_file_closer():
        for closer in closers:
            closer()

    combined.set_close(multi_file_closer)

    combined.attrs = datasets[0].attrs
    return combined


class OdimH5GroupAttributeMixin:
    """Mixin Class for Odim Group Attribute Retrieval"""

    __slots__ = ["_attrs", "_ncfile", "_ncpath", "_parent", "_how", "_what", "_where"]

    def __init__(self, ncfile=None, ncpath=None, parent=None):
        super().__init__()
        self._ncfile = ncfile
        self._ncpath = ncpath
        self._parent = parent
        self._attrs = None
        self._how = None
        self._what = None
        self._where = None

    @property
    def is_netcdf(self):
        return has_import(netCDF4) and isinstance(self.ncfile, netCDF4.Dataset)

    @property
    def is_h5netcdf(self):
        return has_import(h5netcdf) and isinstance(self.ncfile, h5netcdf.File)

    @property
    def ncpath(self):
        """Returns path string inside HDF5 File."""
        return self._ncpath

    @property
    def ncid(self):
        """Returns handle for current path."""
        # root-group can't be subset with netcdf4 and h5netcdf
        if self._ncpath == "/":
            if self.is_netcdf or self.is_h5netcdf:
                return self._ncfile
        return self._ncfile[self.ncpath]

    @property
    def ncfile(self):
        """Returns file handle."""
        return self._ncfile

    @property
    def how(self):
        """Return attributes of `how`-group."""
        if self._how is None:
            self._how = self._get_attributes("how")
        return self._how

    @property
    def what(self):
        """Return attributes of `what`-group."""
        if self._what is None:
            self._what = self._get_attributes("what")
        return self._what

    @property
    def where(self):
        """Return attributes of `where`-group."""
        if self._where is None:
            self._where = self._get_attributes("where")
        return self._where

    @property
    def attrs(self):
        """Return group attributes."""
        if self._attrs is None:
            if self.is_netcdf:
                self._attrs = {k: self.ncid.getncattr(k) for k in self.ncid.ncattrs()}
            else:
                self._attrs = self._decode({**self.ncid.attrs})
        return self._attrs

    @property
    def filename(self):
        """Return filename group belongs to."""
        if self.is_netcdf:
            return self.ncfile.filepath()
        else:
            return self.ncfile.filename

    @property
    def groups(self):
        """Return list of available groups."""
        if self.is_netcdf:
            return list(self.ncid.groups)
        else:
            return list(self.ncid.keys())

    @property
    def engine(self):
        """Return engine used for accessing data"""
        if self.is_netcdf:
            return "netcdf4"
        else:
            return "h5netcdf"

    @property
    def parent(self):
        """Return parent object."""
        return self._parent

    def _get_attributes(self, grp, ncid=None):
        """Return dict with attributes extracted from `grp`"""
        if ncid is None:
            ncid = self.ncid
        try:
            if self.is_netcdf:
                attrs = {k: ncid[grp].getncattr(k) for k in ncid[grp].ncattrs()}
                return attrs
            else:
                attrs = {**ncid[grp].attrs}
                attrs = self._decode(attrs)
                return attrs
        except (IndexError, KeyError):
            return None

    def _get_attribute(self, grp, attr=None, ncid=None):
        """Return single attribute extracted from `grp`"""
        if ncid is None:
            ncid = self.ncid
        try:
            if self.is_netcdf:
                return ncid[grp].getncattr(attr)
            else:
                v = ncid[grp].attrs[attr]
                try:
                    v = v.item()
                except (ValueError, AttributeError):
                    pass
                try:
                    v = v.decode()
                except (UnicodeDecodeError, AttributeError):
                    pass
                return v
        except (IndexError, KeyError):
            return None

    def _decode(self, attrs):
        """Decode strings if possible."""
        for k, v in attrs.items():
            try:
                v = v.item()
            except (ValueError, AttributeError):
                pass
            try:
                v = v.decode()
            except (UnicodeDecodeError, AttributeError):
                pass
            attrs[k] = v
        return attrs


class OdimH5SweepMetaDataMixin:
    """Mixin Class for Odim MetaData."""

    def __init__(self):
        super().__init__()
        self._a1gate = None
        self._angle_resolution = None
        self._azimuth = None
        self._elevation = None
        self._fixed_angle = None
        self._nrays = None
        self._nbins = None
        self._time = None
        self._endtime = None
        self._rtime = None
        self._rng = None

    @property
    def a1gate(self):
        """Return and cache a1gate, azimuth of first measured gate"""
        if self._a1gate is None:
            self._a1gate = self._get_a1gate()
        return self._a1gate

    @property
    def angle_resolution(self):
        """Return and cache angular resolution in degree."""
        if self._angle_resolution is None:
            self._angle_resolution = self._get_angle_resolution()
        return self._angle_resolution

    @property
    def azimuth(self):
        """Return and cache azimuth xr.DataArray."""
        if self._azimuth is None:
            self._azimuth = self._get_azimuth()
        return self._azimuth

    @property
    def elevation(self):
        """Return and cache elevation xr.DataArray."""
        if self._elevation is None:
            self._elevation = self._get_elevation()
        return self._elevation

    @property
    def fixed_angle(self):
        """Return and cache elevation angle in degree."""
        if self._fixed_angle is None:
            self._fixed_angle = self._get_fixed_angle()
        return self._fixed_angle

    @property
    def nrays(self):
        """Return and cache number of rays."""
        if self._nrays is None:
            self._nrays = self._get_nrays()
        return self._nrays

    @property
    def nbins(self):
        """Return and cache number of bins."""
        if self._nbins is None:
            self._nbins = self._get_nbins()
        return self._nbins

    @property
    def rng(self):
        """Return and cache range xr.DataArray."""
        if self._rng is None:
            self._rng = self._get_range()
        return self._rng

    @property
    def ray_times(self):
        """Return and cache ray_times xr.DataArray."""
        if self._rtime is None:
            da = self._get_ray_times()
            # decode, if necessary
            if self.decode_times:
                da = self._decode_cf(da)
            self._rtime = da
        return self._rtime

    @property
    def time(self):
        """Return and cache time xr.DataArray."""
        if self._time is None:
            da = self._get_time()
            # decode, if necessary
            if self.decode_times:
                da = self._decode_cf(da)
            self._time = da
        return self._time

    @property
    def starttime(self):
        """Return sweep starttime xr.DataArray."""
        return self._time

    @property
    def endtime(self):
        """Return sweep endtime xr.DataArray."""
        if self._endtime is None:
            da = self._get_time(point="end")
            # decode, if necessary
            if self.decode_times:
                da = self._decode_cf(da)
            self._endtime = da
        return self._endtime


class XRadMoment(OdimH5GroupAttributeMixin):
    """Class for holding one radar moment

    Parameters
    ----------
    ncfile : {netCDF4.Dataset, h5py.File or h5netcdf.File object}
        File handle of file containing radar sweep
    ncpath : str
        path to moment group (datasetX)
    parent : XRadSweep
        parent sweep object
    """

    def __init__(self, ncfile, ncpath, parent):
        super().__init__(ncfile, ncpath, parent)
        self._quantity = None

    def __repr__(self):
        summary = [f"<wradlib.{type(self).__name__}>"]

        dims = "Dimension(s):"
        dims_summary = [f"{self.parent._dim0[0]}: {self.parent.nrays}"]
        dims_summary.append(f"{self.parent._dim1}: {self.parent.nbins}")
        dims_summary = ", ".join(dims_summary)
        summary.append(f"{dims} ({dims_summary})")

        angle = "Elevation(s):"
        angle_summary = f"{self.parent.fixed_angle:.1f}"
        summary.append(f"{angle} ({angle_summary})")

        moms = "Moment:"
        moms_summary = f"{self.quantity}"
        summary.append(f"{moms} ({moms_summary})")

        return "\n".join(summary)

    @property
    def data(self):
        """Return moment xr.DataArray."""
        return self.parent.data[self.quantity]

    @property
    def time(self):
        """Return sweep time."""
        return self.parent.time

    @property
    def quantity(self):
        """Return `quantity` aka moment name"""
        if self._quantity is None:
            if isinstance(self.parent, XRadSweepOdim):
                self._quantity = self.what["quantity"]
            else:
                self._quantity = GAMIC_NAMES[self.attrs["moment"].lower()]
        return self._quantity


class XRadSweep(OdimH5GroupAttributeMixin, OdimH5SweepMetaDataMixin, XRadBase):
    """Class for holding one radar sweep

    Parameters
    ----------

    ncfile : {netCDF4.Dataset, h5py.File or h5netcdf.File object}
        File handle of file containing radar sweep
    ncpath : str
        path to sweep group
    """

    def __init__(self, ncfile, ncpath, parent=None, **kwargs):
        super().__init__(ncfile, ncpath, parent)
        self._dask_kwargs = {
            "chunks": kwargs.get("chunks", None),
            "parallel": kwargs.get("parallel", False),
        }
        self._cf_kwargs = {
            "mask_and_scale": kwargs.get("mask_and_scale", True),
            "decode_coords": kwargs.get("decode_coords", True),
            "decode_times": kwargs.get("decode_times", True),
        }
        self._misc_kwargs = {
            "keep_elevation": kwargs.get("keep_elevation", False),
            "keep_azimuth": kwargs.get("keep_azimuth", False),
        }
        self._data = None
        self._need_time_recalc = False
        self._seq.extend(self._get_moments())
        self._dim0 = ("azimuth", "elevation")
        self._dim1 = "range"
        self.fixed_angle

    def __repr__(self):
        summary = [f"<wradlib.{type(self).__name__}>"]

        dims = "Dimension(s):"
        dims_summary = [f"{self._dim0[0]}: {self.nrays}"]
        dims_summary.append(f"{self._dim1}: {self.nbins}")
        dims_summary = ", ".join(dims_summary)
        summary.append(f"{dims} ({dims_summary})")

        angle = f"{self._dim0[1].capitalize()}(s):"
        angle_summary = f"{self.fixed_angle:0.1f}"
        summary.append(f"{angle} ({angle_summary})")

        moms = "Moment(s):"
        moms_summary = self.moments
        moms_summary = ", ".join(moms_summary)
        summary.append(f"{moms} ({moms_summary})")

        return "\n".join(summary)

    def __del__(self):
        if self._data is not None:
            self._data.close()
            self._data = None
        self._ncfile = None

    def _decode_cf(self, obj):
        if isinstance(obj, xr.DataArray):
            out = xr.decode_cf(xr.Dataset({"arr": obj}), **self._cf_kwargs).arr
        else:
            out = xr.decode_cf(obj, **self._cf_kwargs)
        return out

    def _get_coords(self):
        ds = xr.Dataset(
            coords={
                "time": self.time,
                "rtime": self.ray_times,
                "azimuth": self.azimuth,
                "elevation": self.elevation,
                "range": self.rng,
            }
        )
        return ds

    def _get_moments(self):
        mdesc = self._mdesc
        moments = [k for k in self.groups if mdesc in k]
        moments_idx = np.argsort([int(s[len(mdesc) :]) for s in moments])
        moments_names = np.array(moments)[moments_idx].tolist()
        moments = [
            XRadMoment(
                ncfile=self.ncfile, ncpath="/".join([self.ncpath, mom]), parent=self
            )
            for mom in moments_names
        ]
        return moments

    def reset_data(self):
        """Reset .data xr.Dataset"""
        self._data = None

    @property
    def _mdesc(self):
        return self._get_mdesc()

    @property
    def chunks(self):
        """Return `chunks` setting."""
        return self._dask_kwargs.get("chunks")

    @property
    def parallel(self):
        """Return `parallel` setting."""
        return self._dask_kwargs.get("parallel")

    @property
    def mask_and_scale(self):
        """Return `mask_and_scale` setting."""
        return self._cf_kwargs.get("mask_and_scale")

    @property
    def decode_coords(self):
        """Return `decode_coords` setting."""
        return self._cf_kwargs.get("decode_coords")

    @property
    def decode_times(self):
        """Return `decode_times` setting."""
        return self._cf_kwargs.get("decode_times")

    @property
    def data(self):
        """Return and cache moments as combined xr.Dataset"""
        if self._data is None:
            self._data = self._merge_moments()

            # if self._data is not None:
            # if metadata declared in XRadTimeseries, load and assign
            if self.parent._meta is not None:
                vars = {}
                for k, v in self.parent._meta.items():
                    attr = self._get_attribute(v, attr=k)
                    if hasattr(attr, "ndim"):
                        attr = xr.DataArray(attr, dims=[self._dim0[0]])
                    vars[k] = attr
                self._data = self._data.assign(vars)

            if self.decode_coords:
                coords = self._get_coords().coords
                self._data = self._data.assign_coords(coords)
                self._data = self._data.sortby(self._dim0[0])
                self._data = self._data.pipe(_reindex_angle, self)
                sweep_mode = (
                    "azimuth_surveillance" if self._dim0[0] == "azimuth" else "rhi"
                )
                self._data = self._data.assign_coords({"sweep_mode": sweep_mode})
                self._data = self._data.assign_coords(self.parent.parent.site.coords)

            if self.mask_and_scale | self.decode_coords | self.decode_times:
                self._data = self._data.pipe(self._decode_cf)

        return self._data

    @property
    def coords(self):
        """Returns xr.Dataset containing coordinates."""
        # sort coords by azimuth, only necessary for gamic flavour
        # for odim is already sorted
        return self._get_coords().sortby(self._dim0[0])

    @property
    def moments(self):
        """Return list of moments."""
        return [f"{k.quantity}" for k in self]


class XRadSweepOdim(XRadSweep):
    """Class for holding one radar sweep

    Parameters
    ----------

    ncfile : {netCDF4.Dataset, h5py.File or h5netcdf.File object}
        File handle of file containing radar sweep
    ncpath : str
        path to sweep group
    """

    def __init__(self, ncfile, ncpath, parent=None, **kwargs):
        super().__init__(ncfile, ncpath, parent, **kwargs)

    def _get_a1gate(self):
        return self.where["a1gate"]

    def _get_angle_resolution(self):
        return self.azimuth.diff(self._dim0[0]).median(skipna=True).round(decimals=1)

    def _get_fixed_angle(self):
        # try RHI first
        angle_keys = ["az_angle", "azangle"]
        for ak in angle_keys:
            angle = self.where.get(ak, None)
            if angle is not None:
                break
        if angle is not None:
            angle = np.round(angle, decimals=1)
            self._dim0 = (self._dim0[1], self._dim0[0])
        else:
            angle = np.round(self.where["elangle"], decimals=1)
        return angle

    def _get_azimuth_how(self):
        how = self.how
        startaz = how["startazA"]
        stopaz = how["stopazA"]
        zero_index = np.where(stopaz < startaz)
        stopaz[zero_index[0]] += 360
        azimuth_data = (startaz + stopaz) / 2.0
        return azimuth_data

    def _get_azimuth_where(self):
        nrays = self.where["nrays"]
        res = 360.0 / nrays
        azimuth_data = np.arange(res / 2.0, 360.0, res, dtype="float32")
        return azimuth_data

    def _get_elevation_how(self):
        how = self.how
        startel = how["startelA"]
        stopel = how["stopelA"]
        elevation_data = (startel + stopel) / 2.0
        return elevation_data

    def _get_elevation_where(self):
        where = self.where
        nrays = where["nrays"]
        elangle = where["elangle"]
        elevation_data = np.ones(nrays, dtype="float32") * elangle
        return elevation_data

    def _get_time_how(self):
        how = self.how
        startT = how["startazT"]
        stopT = how["stopazT"]
        time_data = (startT + stopT) / 2.0
        return time_data

    def _get_time_what(self, nrays=None):
        what = self.what
        startdate = what["startdate"]
        starttime = what["starttime"]
        enddate = what["enddate"]
        endtime = what["endtime"]
        start = dt.datetime.strptime(startdate + starttime, "%Y%m%d%H%M%S")
        end = dt.datetime.strptime(enddate + endtime, "%Y%m%d%H%M%S")
        start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        end = end.replace(tzinfo=dt.timezone.utc).timestamp()
        if nrays is None:
            nrays = self.where["nrays"]
        if start == end:
            warnings.warn(
                "WRADLIB: Equal ODIM `starttime` and `endtime` "
                "values. Can't determine correct sweep start-, "
                "end- and raytimes.",
                UserWarning,
            )

            time_data = np.ones(nrays) * start
        else:
            delta = (end - start) / nrays
            time_data = np.arange(start + delta / 2.0, end, delta)
            time_data = np.roll(time_data, shift=+self.a1gate)
        return time_data

    def _get_ray_times(self, nrays=None):
        try:
            time_data = self._get_time_how()
            self._need_time_recalc = False
        except (AttributeError, KeyError, TypeError):
            time_data = self._get_time_what(nrays=nrays)
            self._need_time_recalc = True
        da = xr.DataArray(time_data, dims=[self._dim0[0]], attrs=time_attrs)
        return da

    def _get_azimuth(self):
        try:
            azimuth_data = self._get_azimuth_how()
        except (AttributeError, KeyError, TypeError):
            azimuth_data = self._get_azimuth_where()
        da = xr.DataArray(
            azimuth_data, dims=[self._dim0[0]], attrs=az_attrs_template.copy()
        )
        if not self._misc_kwargs["keep_azimuth"] and self._dim0[0] == "elevation":
            da = da.pipe(_fix_angle)
        return da

    def _get_elevation(self):
        try:
            elevation_data = self._get_elevation_how()
        except (AttributeError, KeyError, TypeError):
            elevation_data = self._get_elevation_where()
        da = xr.DataArray(
            elevation_data, dims=[self._dim0[0]], attrs=el_attrs_template.copy()
        )
        if not self._misc_kwargs["keep_elevation"] and self._dim0[0] == "azimuth":
            da = da.pipe(_fix_angle)
        return da

    def _get_mdesc(self):
        return "data"

    def _get_range(self):
        where = self.where
        ngates = where["nbins"]
        range_start = where["rstart"] * 1000.0
        bin_range = where["rscale"]
        cent_first = range_start + bin_range / 2.0
        range_data = np.arange(
            cent_first, range_start + bin_range * ngates, bin_range, dtype="float32"
        )
        range_attrs["meters_to_center_of_first_gate"] = cent_first
        range_attrs["meters_between_gates"] = bin_range
        da = xr.DataArray(range_data, dims=[self._dim1], attrs=range_attrs)
        return da

    def _merge_moments(self):
        ds = _open_mfmoments(
            self,
            chunks=self.chunks,
            preprocess=_preprocess_moment,
            parallel=self.parallel,
            mask_and_scale=self.mask_and_scale,
            decode_times=self.decode_times,
            decode_coords=self.decode_coords,
        )
        return ds

    def _get_nrays(self):
        return self.where["nrays"]

    def _get_nbins(self):
        return self.where["nbins"]

    def _get_time(self, point="start"):
        what = self.what
        startdate = what[f"{point}date"]
        starttime = what[f"{point}time"]
        start = dt.datetime.strptime(startdate + starttime, "%Y%m%d%H%M%S")
        start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        da = xr.DataArray(start, attrs=time_attrs)
        return da

    def _get_time_fast(self):
        ncid = self.ncid
        try:
            if self.is_netcdf:
                startdate = ncid["what"].getncattr("startdate")
                starttime = ncid["what"].getncattr("starttime")
            else:
                startdate = _maybe_decode(ncid["what"].attrs["startdate"])
                starttime = _maybe_decode(ncid["what"].attrs["starttime"])
        except (IndexError, KeyError):
            return None
        start = dt.datetime.strptime(startdate + starttime, "%Y%m%d%H%M%S")
        start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        return start


class XRadSweepGamic(XRadSweep):
    """Class for holding one radar sweep

    Parameters
    ----------
    ncfile : {netCDF4.Dataset, h5py.File or h5netcdf.File object}
        File handle of file containing radar sweep
    ncpath : str
        path to sweep group
    """

    def __init__(self, ncfile, ncpath, parent=None, **kwargs):
        super().__init__(ncfile, ncpath, parent, **kwargs)
        self._ray_header = None

    @property
    def ray_header(self):
        # todo: caching adds to memory footprint
        if self._ray_header is None:
            self._ray_header = self.ncid["ray_header"][:]
        return self._ray_header

    def _get_a1gate(self):
        return np.argsort(self.coords.rtime.values)[0]

    def _get_angle_resolution(self):
        return self.how["angle_step"]

    def _get_azimuth(self):
        azstart = self.ray_header["azimuth_start"]
        azstop = self.ray_header["azimuth_stop"]
        if self._dim0[0] == "azimuth":
            zero_index = np.where(azstop < azstart)
            azstop[zero_index[0]] += 360
        azimuth = (azstart + azstop) / 2.0
        da = xr.DataArray(azimuth, dims=[self._dim0[0]], attrs=az_attrs_template.copy())
        if not self._misc_kwargs["keep_azimuth"] and self._dim0[0] == "elevation":
            da = da.pipe(_fix_angle)
        return da

    def _get_elevation(self):
        elstart = self.ray_header["elevation_start"]
        elstop = self.ray_header["elevation_stop"]
        elevation = (elstart + elstop) / 2.0
        da = xr.DataArray(
            elevation, dims=[self._dim0[0]], attrs=el_attrs_template.copy()
        )
        if not self._misc_kwargs["keep_elevation"] and self._dim0[0] == "azimuth":
            da = da.pipe(_fix_angle)
        return da

    def _get_mdesc(self):
        return "moment_"

    def _get_range(self):
        range_samples = self.how["range_samples"]
        range_step = self.how["range_step"]
        bin_range = range_step * range_samples
        range_data = np.arange(
            bin_range / 2.0, bin_range * self.nbins, bin_range, dtype="float32"
        )
        range_attrs["meters_to_center_of_first_gate"] = bin_range / 2.0
        da = xr.DataArray(range_data, dims=[self._dim1], attrs=range_attrs)
        return da

    def _get_ray_times(self):
        times = self.ray_header["timestamp"] / 1e6
        attrs = {"units": "seconds since 1970-01-01T00:00:00Z", "standard_name": "time"}
        da = xr.DataArray(times, dims=[self._dim0[0]], attrs=attrs)
        return da

    def _get_fixed_angle(self):
        try:
            angle = np.round(self.how[self._dim0[1]], decimals=1)
        except KeyError:
            self._dim0 = (self._dim0[1], self._dim0[0])
            angle = np.round(self.how[self._dim0[1]], decimals=1)

        return angle

    def _merge_moments(self):
        if "h5" in self.engine:
            if Version(h5netcdf.__version__) < Version("0.8.0"):
                warnings.warn(
                    f"WRADLIB: 'h5netcdf>=0.8.0' needed to perform this "
                    f"operation. 'h5netcdf={h5netcdf.__version__} "
                    f"available.",
                    UserWarning,
                )
                return None
            if Version(xr.__version__) < Version("0.15.0"):
                warnings.warn(
                    f"WRADLIB: 'xarray>=0.15.0' needed to perform this "
                    f"operation. 'xarray={xr.__version__} "
                    f"available.",
                    UserWarning,
                )
                return None
            opener = h5netcdf.File
            opener_kwargs = {"phony_dims": "access"}
            store = xr.backends.H5NetCDFStore
        else:
            opener = netCDF4.Dataset
            opener_kwargs = {}
            store = xr.backends.NetCDF4DataStore

        if os.path.isfile(self.filename):
            ds0 = opener(self.filename, "r", **opener_kwargs)
        else:
            ds0 = self.ncfile

        ds = xr.open_dataset(store(ds0, self.ncpath), chunks=self.chunks)

        ds = ds.drop_vars("ray_header", errors="ignore")
        for mom in self:
            mom_name = mom.ncpath.split("/")[-1]
            dmom = ds[mom_name]
            name = dmom.moment.lower()
            try:
                name = GAMIC_NAMES[name]
            except KeyError:
                ds = ds.drop_vars(mom_name)
                continue

            # extract and translate attributes to cf
            attrs = collections.OrderedDict()
            dmax = np.iinfo(dmom.dtype).max
            dmin = np.iinfo(dmom.dtype).min
            minval = dmom.dyn_range_min
            maxval = dmom.dyn_range_max
            dtype = minval.dtype
            dyn_range = maxval - minval
            if maxval != minval:
                gain = dyn_range / (dmax - 1)
                minval -= gain
            else:
                gain = (dmax - dmin) / dmax
                minval = dmin
            gain = np.array([gain])[0].astype(dtype)
            minval = np.array([minval])[0].astype(dtype)
            undetect = np.array([dmin])[0].astype(dtype)
            attrs["scale_factor"] = gain
            attrs["add_offset"] = minval
            attrs["_FillValue"] = undetect
            attrs["_Undetect"] = undetect

            if self.decode_coords:
                attrs["coordinates"] = "elevation azimuth range"

            mapping = moments_mapping[name]
            attrs.update({key: mapping[key] for key in moment_attrs})
            # assign attributes to moment
            dmom.attrs = collections.OrderedDict()
            dmom.attrs.update(attrs)
            ds = ds.rename({mom_name: name.upper()})

        # fix dimensions
        dims = sorted(list(ds.dims.keys()), key=lambda x: int(x[len("phony_dim_") :]))
        ds = ds.rename({dims[0]: self._dim0[0], dims[1]: self._dim1})

        # todo: this sorts and reindexes the unsorted GAMIC dataset by azimuth
        # only if `decode_coords` is False
        # adding coord ->  sort -> reindex -> remove coord
        if not self.decode_coords:  # and (self._dim0[0] == "azimuth"):
            ds = (
                ds.assign_coords({self._dim0[0]: getattr(self, self._dim0[0])})
                .sortby(self._dim0[0])
                .pipe(_reindex_angle, self)
                .drop_vars(self._dim0[0])
            )
        return ds

    def _get_nrays(self):
        return self.how["ray_count"]

    def _get_nbins(self):
        return self.how["bin_count"]

    def _get_time(self):
        start = self.how["timestamp"]
        start = dateutil.parser.parse(start)
        start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        da = xr.DataArray(start, attrs=time_attrs)
        return da

    def _get_time_fast(self):
        ncid = self.ncid
        try:
            if self.is_netcdf:
                start = ncid["how"].getncattr("timestamp")
            else:
                start = ncid["how"].attrs["timestamp"]
                if Version(h5py.__version__) < Version("3.0.0"):
                    start = start.decode()
        except (IndexError, KeyError):
            return None
        start = dateutil.parser.parse(start)
        start = start.replace(tzinfo=dt.timezone.utc).timestamp()
        return start


class XRadTimeSeries(OdimH5GroupAttributeMixin, XRadBase):
    """Class for holding a timeseries of radar sweeps"""

    def __init__(self, **kwargs):
        super().__init__()
        self._data = None
        self._moments = None
        self._meta = None

    # override append and claim file for OdimH5GroupAttributeMixin
    def append(self, value):
        # do only for first file in this timeseries
        value._parent = self
        if not len(self):
            self._ncfile = value.ncfile
            self._ncpath = value.ncpath
        return super().append(value)

    def __repr__(self):
        summary = [f"<wradlib.{type(self).__name__}>"]
        dims = "Dimension(s):"
        dims_summary = [f"time: {len(self)}"]
        dims_summary.append(f"{self._seq[0]._dim0[0]}: {self._seq[0].nrays}")
        dims_summary.append(f"{self._seq[0]._dim1}: {self._seq[0].nbins}")
        dims_summary = ", ".join(dims_summary)
        summary.append(f"{dims} ({dims_summary})")
        angle = f"{self._seq[0]._dim0[1].capitalize()}(s):"
        angle_summary = self[0].fixed_angle
        summary.append(f"{angle} ({angle_summary:.1f})")

        return "\n".join(summary)

    def reset_data(self):
        self._data = None

    @property
    def data(self):
        if self._data is None:
            # moments handling
            # coords = set(['rtime', 'range', 'azimuth', 'elevation', 'time',
            #               'altitude', 'latitude', 'longitude', 'sweep_mode'])
            # get intersection and union
            moment_set = [set(t1.moments) for t1 in self]
            moment_set_i = set.intersection(*moment_set)
            moment_set_u = set.union(*moment_set)
            # drop variables not available in all datasets
            drop = moment_set_i ^ moment_set_u
            # keep = (moment_set_i | coords) ^ coords
            drop = list(self.check_moments().keys())
            if drop:
                warnings.warn(
                    f"wradlib: Moments {*drop,} are not available in all datasets "
                    "and will be dropped from the result.\n"
                    "This will be solved in xarray, see "
                    "https://github.com/pydata/xarray/pull/3545"
                )

            # todo: catch possible error and add precise ErrorMessage
            self._data = xr.concat(
                [
                    f.data.drop_vars(drop, errors="ignore")
                    for f in tqdm(
                        self, desc="Collecting", unit=" Timesteps", leave=None
                    )
                ],
                # data_vars=list(keep),
                dim="time",
            )
        return self._data

    def check_rays(self):
        nrays = [swp.nrays for swp in self]
        snrays = set(nrays)
        idx = []
        for nr in snrays:
            if nr % 360:
                idx.extend(np.argwhere(np.array(nrays) == nr).flatten().tolist())

        if len(snrays) > 1:
            warnings.warn(
                f"wradlib: number of rays differing between sweeps.\n" f"{snrays}"
            )
        return snrays, idx

    def check_moments(self):
        moments = [set([mom.quantity for mom in swp]) for swp in self]
        mi = set.intersection(*moments)
        mu = set.union(*moments)
        mp = mi ^ mu
        miss = {}
        for mom in mu:
            idx = []
            if mom in mp:
                for i, mset in enumerate(moments):
                    if mom not in mset:
                        idx.append(i)
                miss[mom] = idx
        return miss

    def set_moments(self, moments):
        if not isinstance(moments, list):
            pass
        else:
            self._moments = moments

    def set_metadata(self, metadata):
        if not isinstance(metadata, dict):
            pass
        else:
            self._meta = metadata


class XRadVolume(OdimH5GroupAttributeMixin, XRadBase):
    """Class for holding a volume of radar sweeps"""

    def __init__(self, **kwargs):
        super().__init__()
        self._data = None
        self._root = None

    def __repr__(self):
        summary = [f"<wradlib.{type(self).__name__}>"]
        dims = "Dimension(s):"
        dims_summary = f"sweep: {len(self)}"
        summary.append(f"{dims} ({dims_summary})")
        angle = f"{self[0][0]._dim0[1].capitalize()}(s):"
        angle_summary = [f"{k[0].fixed_angle:.1f}" for k in self]
        angle_summary = ", ".join(angle_summary)
        summary.append(f"{angle} ({angle_summary})")

        return "\n".join(summary)

    @property
    def root(self):
        """Return root object."""
        if self._root is None:
            self.assign_root()
        return self._root

    def assign_root(self):
        """(Re-)Create root object according CfRadial2 standard"""
        # assign root variables
        sweep_group_names = [f"sweep_{i}" for i in range(len(self))]

        try:
            sweep_fixed_angles = [ts[0].fixed_angle for ts in self]
        except AttributeError:
            sweep_fixed_angles = [ts.fixed_angle for ts in self]

        # extract time coverage
        times = np.array(
            [[t[0].ray_times.values.min(), t[-1].ray_times.values.max()] for t in self]
        ).flatten()
        time_coverage_start = min(times)
        time_coverage_end = max(times)

        time_coverage_start_str = str(time_coverage_start)[:19] + "Z"
        time_coverage_end_str = str(time_coverage_end)[:19] + "Z"

        # create root group from scratch
        root = xr.Dataset()  # data_vars=wrl.io.xarray.global_variables,
        # attrs=wrl.io.xarray.global_attrs)

        # take first dataset/file for retrieval of location
        site = self.site

        # assign root variables
        root = root.assign(
            {
                "volume_number": 0,
                "platform_type": str("fixed"),
                "instrument_type": "radar",
                "primary_axis": "axis_z",
                "time_coverage_start": time_coverage_start_str,
                "time_coverage_end": time_coverage_end_str,
                "latitude": site["latitude"].values,
                "longitude": site["longitude"].values,
                "altitude": site["altitude"].values,
                "sweep_group_name": (["sweep"], sweep_group_names),
                "sweep_fixed_angle": (["sweep"], sweep_fixed_angles),
            }
        )

        # assign root attributes
        attrs = collections.OrderedDict()
        attrs.update(
            {
                "version": "None",
                "title": "None",
                "institution": "None",
                "references": "None",
                "source": "None",
                "history": "None",
                "comment": "im/exported using wradlib",
                "instrument_name": "None",
            }
        )
        attrs["version"] = self.what["version"]
        root = root.assign_attrs(attrs)
        root = root.assign_attrs(self.attrs)
        self._root = root

    @property
    def site(self):
        """Return coordinates of radar site."""
        ds = xr.Dataset(coords=self.where).rename(
            {"height": "altitude", "lon": "longitude", "lat": "latitude"}
        )
        return ds

    @property
    def Conventions(self):
        """Return Conventions string."""
        try:
            conv = self.ncid.attrs["Conventions"]
        except KeyError:
            conv = None
        return conv

    def to_odim(self, filename, timestep=0):
        """Save volume to ODIM_H5/V2_2 compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        timestep : int
            timestep of wanted volume
        """
        if self.root:
            to_odim(self, filename, timestep=timestep)
        else:
            warnings.warn(
                "WRADLIB: No OdimH5-compliant data structure " "available. Not saving.",
                UserWarning,
            )

    def to_cfradial2(self, filename, timestep=0):
        """Save volume to CfRadial2 compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        timestep : int
            timestep wanted volume
        """
        if self.root:
            to_cfradial2(self, filename, timestep=timestep)
        else:
            warnings.warn(
                "WRADLIB: No CfRadial2-compliant data structure "
                "available. Not saving.",
                UserWarning,
            )

    def to_netcdf(self, filename, timestep=None, keys=None):
        """Save volume to netcdf compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        timestep : int, slice
            timestep/slice of wanted volume
        keys : list
            list of sweep_group_names which should be written to the file
        """
        if self.root:
            to_netcdf(self, filename, keys=keys, timestep=timestep)
        else:
            warnings.warn(
                "WRADLIB: No netcdf-compliant data structure " "available. Not saving.",
                UserWarning,
            )


@deprecation.deprecated(
    deprecated_in="1.10",
    removed_in="2.0",
    current_version=version.version,
    details="Use xarray BackendEntrypoint based functionality instead.",
)
def collect_by_time(obj):
    """Collect XRadSweep objects having same time

    Parameters
    ----------
    obj : list
        list of XRadSweep objects

    Returns
    -------
    out : XRadTimeSeries
        wrapper around list of XRadSweep objects
    """
    out = XRadTimeSeries()
    if isinstance(obj, XRadSweep):
        obj = [obj]
    times = [ds._get_time_fast() for ds in obj]
    unique_times = np.array(sorted(list(set(times))))
    if len(unique_times) == len(obj):
        out.extend(obj)
        out.sort(key=lambda x: x._get_time_fast())
    else:
        # runs only if several files for the same timestep are available
        # eg DWD's one sweep one moment files
        for t in unique_times:
            idx = np.argwhere(times == t).flatten()
            out1 = obj[idx[0]]
            [out1.extend(obj[i]) for i in idx[1:]]
            out.append(out1)
    return out


@deprecation.deprecated(
    deprecated_in="1.10",
    removed_in="2.0",
    current_version=version.version,
    details="Use xarray BackendEntrypoint based functionality instead.",
)
def collect_by_angle(obj):
    """Collect XRadSweep objects having same angle

    Parameters
    ----------
    obj : list
        list of XRadSweep objects

    Returns
    -------
    out : :class:`wradlib.io.xarray.XRadVolume`
        wrapper around nested list of XRadSweep objects
    """
    out = XRadVolume()
    angles = [ds.fixed_angle for ds in obj]
    unique_angles = list(set(angles))
    if len(unique_angles) == len(obj):
        out.extend(obj)
    else:
        for a in unique_angles:
            idx = np.argwhere(angles == a).flatten()
            merge_list = [obj[i] for i in idx]
            out.append(merge_list)
    return out


def _open_odim_sweep(filename, loader, **kwargs):
    """Returns list of XRadSweep objects

    Every sweep will be put into it's own class instance.
    """
    ld_kwargs = kwargs.get("ld_kwargs", {})
    if loader == "netcdf4":
        opener = netCDF4.Dataset
        attr = "groups"
    elif loader == "h5netcdf":
        opener = h5netcdf.File
        attr = "keys"
        ld_kwargs["phony_dims"] = "access"
    else:
        opener = h5py.File
        attr = "keys"

    dsdesc = "dataset"
    sweep_cls = XRadSweepOdim
    if "GAMIC" in kwargs.get("flavour", "ODIM"):
        if loader == "netcdf4":
            raise ValueError(
                "wradlib: GAMIC files can't be read using netcdf4"
                " loader. Use either 'h5py' or 'h5netcdf."
            )
        dsdesc = "scan"
        sweep_cls = XRadSweepGamic

    # open file
    if not isinstance(filename, str):
        if has_import(h5py) and opener == h5py.File:
            raise ValueError(
                "wradlib: file-like objects can't be read using h5py "
                "loader. Use either 'netcdf4' or 'h5netcdf'."
            )
        if has_import(netCDF4) and opener == netCDF4.Dataset:
            handle = opener(
                f"{str(filename)}", mode="r", memory=filename.read(), **ld_kwargs
            )
        else:
            handle = opener(filename, "r", **ld_kwargs)
    else:
        handle = opener(filename, "r", **ld_kwargs)

    # get group names
    fattr = getattr(handle, attr)
    if callable(fattr):
        groups = list(fattr())
    else:
        groups = list(fattr)

    # iterate over single sweeps
    # todo: if sorting does not matter, we can skip this
    sweeps = [k for k in groups if dsdesc in k]
    sweeps_idx = np.argsort([int(s[len(dsdesc) :]) for s in sweeps])
    sweeps = np.array(sweeps)[sweeps_idx].tolist()
    return [sweep_cls(handle, k, **kwargs) for k in sweeps]


@deprecation.deprecated(
    deprecated_in="1.10",
    removed_in="2.0",
    current_version=version.version,
    details="Use the appropriate `wradlib.io.open_{engine}_dataset` or `wradlib.io.open_{engine}_mfdataset` function.",
)
def open_odim(paths, loader="netcdf4", **kwargs):
    """Open multiple ODIM files as a XRadVolume structure.

    Parameters
    ----------
    paths : str or sequence
        Either a filename or string glob in the form `'path/to/my/files/*.h5'`
        or an explicit list of files to open.
    loader : {'netcdf4', 'h5py', 'h5netcdf'}
        Loader used for accessing file metadata, defaults to 'netcdf4'.
    kwargs : dict, optional
        Additional arguments passed on to :class:`wradlib.io.xarray.XRadSweep`.
    """
    if (loader == "h5netcdf") and (
        has_import(h5netcdf) and (Version(h5netcdf.__version__) < Version("0.8.0"))
    ):
        warnings.warn(
            f"WRADLIB: 'h5netcdf>=0.8.0' needed to perform this "
            f"operation. 'h5netcdf={h5netcdf.__version__} "
            f"available.",
            UserWarning,
        )

    if isinstance(paths, str):
        paths = glob.glob(paths)
    else:
        paths = np.array(paths).flatten().tolist()

    if loader not in ["netcdf4", "h5netcdf", "h5py"]:
        raise ValueError(f"wradlib: Unknown loader: {loader}")

    sweeps = []
    [
        sweeps.extend(_open_odim_sweep(f, loader, **kwargs))
        for f in tqdm(paths, desc="Open", unit=" Files", leave=None)
    ]
    angles = collect_by_angle(sweeps)
    for i in tqdm(range(len(angles)), desc="Collecting", unit=" Angles", leave=None):
        angles[i] = collect_by_time(angles[i])
    angles.sort(key=lambda x: x[0].time)
    for f in angles:
        f._parent = angles
    angles._ncfile = angles[0].ncfile
    angles._ncpath = "/"
    return angles


class XRadVolFile:
    """BaseClass for holding netCDF4.Dataset handles"""

    def __init__(self, filename=None, flavour=None, **kwargs):
        self._filename = filename
        self._nch = None
        self._flavour = None
        self._nch, self._flavour = self._check_file(filename, flavour)

    def _check_file(self, filename, flavour):
        raise NotImplementedError

    def __del__(self):
        if self._nch is not None:
            self._nch.close()

    @property
    def filename(self):
        return self._filename

    @property
    def nch(self):
        return self._nch

    @property
    def flavour(self):
        return self._flavour


class OdimH5File(XRadVolFile):
    """Class for holding netCDF4.Dataset handles of OdimH5 files

    Parameters
    ----------
    filename : str
        Source data file name.
    flavour : str
        Name of hdf5 flavour ('ODIM' or 'GAMIC'). Defaults to 'ODIM'.
    """

    def __init__(self, filename=None, flavour=None, **kwargs):
        super().__init__(filename=filename, flavour=flavour, **kwargs)

    def _check_file(self, filename, flavour):
        nch = netCDF4.Dataset(filename, diskless=True, persist=False)
        if nch.disk_format != "HDF5":
            raise TypeError(
                f"wradlib: File {filename} is neither 'NETCDF4' (using HDF5 groups) "
                "nor plain 'HDF5'."
            )
        if flavour is None:
            try:
                flavour = nch.Conventions
            except AttributeError as e:
                raise AttributeError(
                    f"wradlib: Missing 'Conventions' attribute in {filename} ./n"
                    "Use the 'flavour' kwarg to specify your source data."
                ) from e
            if "ODIM" not in flavour:
                raise AttributeError(
                    f"wradlib: 'Conventions' attribute '{flavour}' in {filename} is unknown./n"
                    "Use the 'flavour' kwarg to specify your source data."
                )

        if "ODIM" in flavour:
            self._dsdesc = "dataset"
            self._swmode = "product"
            self._mfmt = "data"
            self._msrc = "groups"
        elif "GAMIC" in flavour:
            self._dsdesc = "scan"
            self._swmode = "scan_type"
            self._mfmt = "moment_"
            self._msrc = "variables"
        else:
            raise AttributeError(
                f"wradlib: Unknown 'flavour' kwarg attribute: {flavour} ."
            )

        return nch, flavour

    @property
    def flavour(self):
        flv = ["ODIM", "GAMIC"]
        return [s for s in flv if s in self._flavour][0]


class NetCDF4File(XRadVolFile):
    """Class for holding netCDF4.Dataset handles of Cf/Radial files

    Parameters
    ----------
    filename : str
        Source data file name.
    flavour : str
        Name of flavour ('Cf/Radial' or 'Cf/Radial2').
    """

    def __init__(self, filename=None, flavour=None, **kwargs):
        super().__init__(filename=filename, flavour=flavour, **kwargs)

    def _check_file(self, filename, flavour):
        nch = netCDF4.Dataset(filename, diskless=True, persist=False)
        if flavour is None:
            try:
                Conventions = nch.Conventions
                version = nch.version
            except AttributeError as e:
                raise AttributeError(
                    f"wradlib: Missing 'Conventions' attribute in {filename} ./n"
                    "Use the 'flavour' kwarg to specify your source data."
                ) from e
            if "cf/radial" in Conventions.lower():
                if version == "2.0":
                    flavour = "Cf/Radial2"
                else:
                    flavour = "Cf/Radial"

        if flavour not in ["Cf/Radial", "Cf/Radial2"]:
            raise AttributeError(
                f"wradlib: Unknown 'flavour' kwarg attribute: {flavour} ."
            )

        return nch, flavour


class XRadVol(collections.abc.MutableMapping):
    """BaseClass for xarray based RadarVolumes

    Implements `collections.MutableMapping` dictionary.
    """

    def __init__(self, init_root=False):
        self._sweeps = {}
        self._nch = []
        self.root = None
        self._sweep_angles = []
        self._sweep_names = []
        if init_root:
            self._init_root()

    def __getitem__(self, key):
        if key == "root":
            warnings.warn(
                "WRADLIB: Use of `obj['root']` is deprecated, "
                "please use obj.root instead.",
                DeprecationWarning,
            )
            return self._root

        return self._sweeps[key]

    def __setitem__(self, key, value):
        if key in self._sweeps:
            self._sweeps[key] = value
        else:
            warnings.warn(
                "WRADLIB: Use class methods to add data. "
                "Direct setting is not allowed.",
                UserWarning,
            )

    def __delitem__(self, key):
        del self._sweeps[key]

    def __iter__(self):
        return iter(self._sweeps)

    def __len__(self):
        return len(self._sweeps)

    def __repr__(self):
        return self._sweeps.__repr__()

    def __del__(self):
        del self._root
        for k in list(self._sweeps):
            del k
        for k in self._nch:
            del k

    def _init_root(self):
        self.root = xr.Dataset(data_vars=global_variables, attrs=global_attrs)

    @property
    def root(self):
        """Return `root` dataset."""
        return self._root

    @root.setter
    def root(self, value):
        self._root = value

    @property
    def sweep_angles(self):
        if self.root is None:
            return self._sweep_angles
        return list(self.root.sweep_fixed_angle.values)

    @sweep_angles.setter
    def sweep_angles(self, value):
        if self.root is None:
            self._sweep_angles.append(value)

    @property
    def sweep_names(self):
        if self.root is None:
            return self._sweep_names
        else:
            return list(self.root.sweep_group_name.values)

    @sweep_names.setter
    def sweep_names(self, value):
        if self.root is None:
            self._sweep_names.append(value)

    @property
    def sweep(self):
        """Return sweep dimension count."""
        return self.root.dims["sweep"]

    @property
    def sweeps(self):
        """Return zip sweep names, sweep_angles"""
        return zip(self.sweep_names, self.sweep_angles)

    @property
    def location(self):
        """Return location of data source."""
        return (
            self.root.longitude.values.item(),
            self.root.latitude.values.item(),
            self.root.altitude.values.item(),
        )

    @property
    def Conventions(self):
        """Return CF/ODIM `Conventions`."""
        return self.root.Conventions

    @property
    def version(self):
        """Return CF/ODIM version"""
        return self.root.version

    def to_cfradial2(self, filename):
        """Save volume to CfRadial2.0 compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        """
        if self.root:
            to_cfradial2(self, filename)
        else:
            warnings.warn(
                "WRADLIB: No CfRadial2-compliant data structure "
                "available. Not saving.",
                UserWarning,
            )

    def to_odim(self, filename):
        """Save volume to ODIM_H5/V2_2 compliant file.

        Parameters
        ----------
        filename : str
            Name of the output file
        """
        if self.root:
            to_odim(self, filename)
        else:
            warnings.warn(
                "WRADLIB: No OdimH5-compliant data structure " "available. Not saving.",
                UserWarning,
            )

    def georeference(self, sweeps=None):
        """Georeference sweeps

        Parameters
        ----------
        sweeps : list
            list with sweep keys to georeference, defaults to all sweeps
        """
        if sweeps is None:
            sweeps = self

        for swp in sweeps:
            self[swp] = self[swp].pipe(geo_xarray.georeference_dataset)


class CfRadial(XRadVol):
    """Class for xarray based retrieval of CfRadial data files"""

    @deprecation.deprecated(
        deprecated_in="1.10",
        removed_in="2.0",
        current_version=version.version,
        details="Use xarray BackendEntrypoint based functionality instead.",
    )
    def __init__(self, filename=None, flavour=None, **kwargs):
        """Initialize xarray structure from Cf/Radial data structure.

        Parameters
        ----------
        filename : str
            Source data file name.
        flavour : str
            Name of flavour ('Cf/Radial' or 'Cf/Radial2').

        Keyword Arguments
        -----------------
        decode_times : bool
            If True, decode cf times to np.datetime64. Defaults to True.
        decode_coords : bool
            If True, use the ‘coordinates’ attribute on variable
            to assign coordinates. Defaults to True.
        mask_and_scale : bool
            If True, lazily scale (using scale_factor and add_offset)
            and mask (using _FillValue). Defaults to True.
        chunks : int or dict, optional
            If chunks is provided, it used to load the new dataset into dask
            arrays. chunks={} loads the dataset with dask using a single
            chunk for all arrays.
        georef : bool
            If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
            2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy)
        dim0 : str
            name of the ray-dimension of DataArrays and Dataset:
                * `time` - cfradial2 standard
                * `azimuth` - better for working with xarray
        """
        super().__init__()
        if not isinstance(filename, list):
            filename = [filename]
        for i, f in enumerate(filename):
            nch = NetCDF4File(f, flavour=flavour)
            self._nch.append(nch)
            if nch.flavour == "Cf/Radial2":
                self.assign_data_radial2(nch, **kwargs)
            else:
                self.assign_data_radial(nch, **kwargs)

    def assign_data_radial2(self, nch, **kwargs):
        """Assign from CfRadial2 data structure.

        Parameters
        ----------
        nch : object
            NetCDF4 File object

        Keyword Arguments
        -----------------
        decode_times : bool
            If True, decode cf times to np.datetime64. Defaults to True.
        decode_coords : bool
            If True, use the ‘coordinates’ attribute on variable
            to assign coordinates. Defaults to True.
        mask_and_scale : bool
            If True, lazily scale (using scale_factor and add_offset)
            and mask (using _FillValue). Defaults to True.
        chunks : int or dict, optional
            If chunks is provided, it used to load the new dataset into dask
            arrays. chunks={} loads the dataset with dask using a single
            chunk for all arrays.
        georef : bool
            If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
            2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy)
        dim0 : str
            name of the ray-dimension of DataArrays and Dataset:
                * `time` - cfradial2 standard
                * `azimuth` - better for working with xarray
        """
        # keyword argument handling
        georef = kwargs.pop("georef", False)
        dim0 = kwargs.pop("dim0", "time")

        self.root = _open_dataset(nch.nch, grp=None, **kwargs)
        sweepnames = self.root.sweep_group_name.values
        for sw in sweepnames:
            ds = _open_dataset(nch.nch, grp=sw, **kwargs)
            ds = ds.swap_dims({"time": dim0})
            coords = {
                "longitude": self.root.longitude,
                "latitude": self.root.latitude,
                "altitude": self.root.altitude,
                "azimuth": ds.azimuth,
                "elevation": ds.elevation,
            }
            ds = ds.assign_coords(**coords)

            # adding xyz aeqd-coordinates
            if georef:
                ds = geo_xarray.georeference_dataset(ds)

            self._sweeps[sw] = ds

    def assign_data_radial(self, nch, **kwargs):
        """Assign from CfRadial1 data structure.

        Keyword Arguments
        -----------------
        decode_times : bool
            If True, decode cf times to np.datetime64. Defaults to True.
        decode_coords : bool
            If True, use the ‘coordinates’ attribute on variable
            to assign coordinates. Defaults to True.
        mask_and_scale : bool
            If True, lazily scale (using scale_factor and add_offset)
            and mask (using _FillValue). Defaults to True.
        chunks : int or dict, optional
            If chunks is provided, it used to load the new dataset into dask
            arrays. chunks={} loads the dataset with dask using a single
            chunk for all arrays.
        georef : bool
            If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
            2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy)
        dim0 : str
            name of the ray-dimension of DataArrays and Dataset:
                * `time` - cfradial2 standard
                * `azimuth` - better for working with xarray
        """
        # keyword argument handling
        georef = kwargs.pop("georef", False)
        dim0 = kwargs.pop("dim0", "time")

        root = _open_dataset(nch.nch, grp=None, **kwargs)
        var = root.variables.keys()
        remove_root = var ^ root_vars
        remove_root &= var
        root1 = root.drop_vars(remove_root).rename({"fixed_angle": "sweep_fixed_angle"})
        sweep_group_name = []
        for i in range(root1.dims["sweep"]):
            sweep_group_name.append(f"sweep_{i + 1}")
        self.root = root1.assign({"sweep_group_name": (["sweep"], sweep_group_name)})

        keep_vars = sweep_vars1 | sweep_vars2 | sweep_vars3
        remove_vars = var ^ keep_vars
        remove_vars &= var
        data = root.drop_vars(remove_vars)
        data.attrs = {}
        start_idx = data.sweep_start_ray_index.values
        end_idx = data.sweep_end_ray_index.values
        data = data.drop_vars({"sweep_start_ray_index", "sweep_end_ray_index"})
        for i, sw in enumerate(sweep_group_name):
            tslice = slice(start_idx[i], end_idx[i])
            ds = data.isel(time=tslice, sweep=slice(i, i + 1)).squeeze("sweep")
            ds = ds.swap_dims({"time": dim0})
            ds.sweep_mode.load()
            coords = {
                "longitude": self.root.longitude,
                "latitude": self.root.latitude,
                "altitude": self.root.altitude,
                "azimuth": ds.azimuth,
                "elevation": ds.elevation,
                "sweep_mode": ds.sweep_mode.item().decode(),
            }
            ds = ds.assign_coords(**coords)

            # adding xyz aeqd-coordinates
            if georef:
                ds = geo_xarray.georeference_dataset(ds)

            self._sweeps[sw] = ds


class OdimH5(XRadVol):
    """Class for xarray based retrieval of ODIM_H5 data files"""

    @deprecation.deprecated(
        deprecated_in="1.10",
        removed_in="2.0",
        current_version=version.version,
        details="Use xarray BackendEntrypoint based functionality instead.",
    )
    def __init__(self, filename=None, flavour=None, **kwargs):
        """Initialize xarray structure from hdf5 data structure.

        Parameters
        ----------
        filename : str
            Source data file name.
        flavour : str
            Name of hdf5 flavour ('ODIM' or 'GAMIC'). Defaults to 'ODIM'.

        Keyword Arguments
        -----------------
        decode_times : bool
            If True, decode cf times to np.datetime64. Defaults to True.
        decode_coords : bool
            If True, use the ‘coordinates’ attribute on variable
            to assign coordinates. Defaults to True.
        mask_and_scale : bool
            If True, lazily scale (using scale_factor and add_offset)
            and mask (using _FillValue). Defaults to True.
        chunks : int or dict, optional
            If chunks is provided, it used to load the new dataset into dask
            arrays. chunks={} loads the dataset with dask using a single
            chunk for all arrays.
        georef : bool
            If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
            2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy)
        standard : str
            * `none` - data is read as verbatim as possible, no metadata
            * `odim` - data is read, odim metadata added to datasets
            * `cf-mandatory` - data is read according to cfradial2 standard
              importing mandatory metadata
            * `cf-full` - data is read according to cfradial2 standard
              importing all available cfradial2 metadata (not fully
              implemented)
        dim0 : str
            name of the ray-dimension of DataArrays and Dataset:
                * `time` - cfradial2 standard
                * `azimuth` - better for working with xarray
        """
        super().__init__()

        if not isinstance(filename, list):
            filename = [filename]

        if len(filename) == 0:
            raise ValueError("File list empty")

        for f in filename:
            self.assign_data(f, flavour=flavour, **kwargs)

        if "cf" in kwargs.get("standard", "cf-mandatory"):
            self.assign_root()

    def assign_data(self, filename, flavour=None, **kwargs):
        """Assign xarray dataset from hdf5 data structure.

        Parameters
        ----------
        filename : str
            Source data file name.
        flavour : str
            Name of hdf5 flavour ('ODIM' or 'GAMIC'). Defaults to 'ODIM'.

        Keyword Arguments
        -----------------
        decode_times : bool
            If True, decode cf times to np.datetime64. Defaults to True.
        decode_coords : bool
            If True, use the ‘coordinates’ attribute on variable
            to assign coordinates. Defaults to True.
        mask_and_scale : bool
            If True, lazily scale (using scale_factor and add_offset)
            and mask (using _FillValue). Defaults to True.
        chunks : int or dict, optional
            If chunks is provided, it used to load the new dataset into dask
            arrays. chunks={} loads the dataset with dask using a single
            chunk for all arrays.
        georef : bool
            If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
            2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy).
            Defaults to False.
        standard : str
            * `none` - data is read as verbatim as possible, no metadata
            * `odim` - data is read, odim metadata added to datasets
            * `cf-mandatory` - data is read according to cfradial2 standard
              importing mandatory metadata, default value
            * `cf-full` - data is read according to cfradial2 standard
              importing all available cfradial2 metadata (not fully
              implemented)
        dim0 : str
            name of the ray-dimension of DataArrays and Dataset:
                * `time` - cfradial2 standard, default value
                * `azimuth` - better for working with xarray
        """
        nch = OdimH5File(filename, flavour=flavour)
        self._nch.append(nch)

        # keyword argument handling
        decode_times = kwargs.get("decode_times", True)
        decode_coords = kwargs.get("decode_coords", True)
        mask_and_scale = kwargs.get("mask_and_scale", True)
        georef = kwargs.get("georef", False)
        standard = kwargs.get("standard", "cf-mandatory")
        dim0 = kwargs.get("dim0", "time")

        # retrieve and assign global groups /how, /what, /where
        groups = ["how", "what", "where"]
        how, what, where = _get_odim_groups(nch.nch, groups)
        rt_grps = {"how": how, "what": what, "where": where}

        # sweep group handling
        (src_swp_grp_name, swp_grp_name) = _get_odim_sweep_group_names(
            nch.nch, nch._dsdesc
        )

        # iterate sweeps in file
        for i, sweep in enumerate(src_swp_grp_name):
            # retrieve ds and assign datasetX how/what/where group attributes
            groups = [None, "how", "what", "where"]
            ds, ds_how, ds_what, ds_where = _get_odim_groups(nch.nch[sweep], groups)
            ds_grps = {"how": ds_how, "what": ds_what, "where": ds_where}

            # moments
            ds = _assign_odim_moments(ds, nch, sweep, **kwargs)

            # retrieve and assign gamic ray_header
            if nch.flavour == "GAMIC":
                rh = _get_gamic_ray_header(nch.filename, i)
                ds_grps["what"] = ds_grps["what"].assign(rh)

            # coordinates wrap-up
            vars = collections.OrderedDict()
            coords = collections.OrderedDict()
            if "cf" in standard or georef:
                coords["longitude"] = rt_grps["where"].attrs["lon"]
                coords["latitude"] = rt_grps["where"].attrs["lat"]
                coords["altitude"] = rt_grps["where"].attrs["height"]
            if "cf" in standard or georef:
                sweep_mode = _get_odim_sweep_mode(nch, ds_grps)
                coords["sweep_mode"] = sweep_mode
            if "cf" in standard or decode_coords or georef:
                coords.update(_get_odim_coordinates(nch, ds_grps))
                # georeference needs coordinate variables
                if georef:
                    geods = xr.Dataset(vars, coords)
                    geods = geo_xarray.georeference_dataset(geods)
                    coords.update(geods.coords)
            # time coordinate
            if "cf" in standard or decode_times:
                timevals = _get_odim_timevalues(nch, ds_grps)
                if decode_times:
                    coords["time"] = (["dim_0"], timevals, time_attrs)
                else:
                    coords["time"] = (["dim_0"], timevals)

            # assign global sweep attributes
            fixed_angle = _get_odim_fixed_angle(nch, ds_grps)
            if "cf" in standard:
                vars.update(
                    {
                        "sweep_number": i,
                        "sweep_mode": sweep_mode,
                        "follow_mode": "none",
                        "prt_mode": "fixed",
                        "fixed_angle": fixed_angle,
                    }
                )
            if "cf-full" in standard:
                full_vars = _get_odim_full_vars(nch, ds_grps)
                vars.update(full_vars)

            # assign variables and coordinates
            ds = ds.assign(vars)
            ds = ds.assign_coords(**coords)
            ds = ds.rename({"dim_0": dim0, "dim_1": "range"})

            # decode dataset if requested
            if decode_times or decode_coords or mask_and_scale:
                ds = xr.decode_cf(
                    ds,
                    decode_times=decode_times,
                    decode_coords=decode_coords,
                    mask_and_scale=mask_and_scale,
                )

            # determine if same sweep
            try:
                index = self.sweep_angles.index(fixed_angle)
            except ValueError:
                nidx = len(self._sweeps) + 1
                swp_grp_name = f"sweep_{nidx}"
                self._sweeps[swp_grp_name] = ds
                self.sweep_names.append(swp_grp_name)
                self.sweep_angles.append(fixed_angle)
            else:
                dictkey = self.sweep_names[index]
                self._sweeps[dictkey] = xr.merge([self._sweeps[dictkey], ds])

    def assign_root(self):
        # retrieve and assign global groups /how, /what, /where
        first = self._nch[0]

        groups = ["how", "what", "where"]
        how, what, where = _get_odim_groups(first.nch, groups)
        rt_grps = {"how": how, "what": what, "where": where}

        # assign root variables
        # extract time coverage
        tmin = [ds.time.values.min() for ds in self._sweeps.values()]
        time_coverage_start = min(tmin)
        tmax = [ds.time.values.max() for ds in self._sweeps.values()]
        time_coverage_end = max(tmax)

        time_coverage_start_str = str(time_coverage_start)[:19] + "Z"
        time_coverage_end_str = str(time_coverage_end)[:19] + "Z"

        # create root group from scratch
        root = xr.Dataset(data_vars=global_variables, attrs=global_attrs)

        # assign root variables
        root = root.assign(
            {
                "volume_number": 0,
                "platform_type": str("fixed"),
                "instrument_type": "radar",
                "primary_axis": "axis_z",
                "time_coverage_start": time_coverage_start_str,
                "time_coverage_end": time_coverage_end_str,
                "latitude": rt_grps["where"].attrs["lat"],
                "longitude": rt_grps["where"].attrs["lon"],
                "altitude": rt_grps["where"].attrs["height"],
                "sweep_group_name": (["sweep"], self.sweep_names),
                "sweep_fixed_angle": (["sweep"], self.sweep_angles),
            }
        )

        # assign root attributes
        attrs = _get_odim_root_attributes(first, rt_grps)
        root = root.assign_attrs(attrs)
        self.root = root


def _open_dataset(nch, grp=None, **kwargs):
    """Open netcdf4/hdf5 group as xarray dataset.

    Parameters
    ----------
    nch : handle
        netcdf4-file handle
    grp : str
        group to access

    Returns
    -------
    nch : handle
        xarray Dataset handle
    """
    if grp is not None:
        nch = nch.groups.get(grp, False)
    if nch:
        nch = xr.open_dataset(xr.backends.NetCDF4DataStore(nch), **kwargs)
    return nch


def _get_gamic_ray_header(filename, scan):
    """Returns GAMIC ray header dictionary.

    Parameters
    ----------
    filename : str
        filename of GAMIC file
    scan : int
        Number of scan in file

    Returns
    -------
    vars : dict
        OrderedDict of ray header items
    """
    # ToDo: move rayheader into own dataset
    h5 = h5py.File(filename, mode="r")
    ray_header = h5[f"scan{scan}/ray_header"][:]
    h5.close()
    vars = collections.OrderedDict()
    for name in ray_header.dtype.names:
        rh = ray_header[name]
        attrs = None
        vars.update({name: (["dim_0"], rh, attrs)})
    return vars


def _get_odim_sweep_group_names(nch, name):
    """Return sweep names.

    Returns source names and cfradial names.

    Parameters
    ----------
    nch : handle
        netCDF4 Dataset handle
    name : str
        Common part of source dataset names.

    Returns
    -------
    src : list
        list of source dataset names
    swg_grp_name : list
        list of corresponding cfradial sweep_group_name
    """
    src = [key for key in nch.groups.keys() if name in key]
    src.sort(key=lambda x: int(x[len(name) :]))
    swp_grp_name = [f"sweep_{i}" for i in range(1, len(src) + 1)]
    return src, swp_grp_name


def _get_odim_variables_moments(ds, moments=None, **kwargs):
    """Retrieve radar moments from dataset variables.

    Parameters
    ----------
    ds : xarray dataset
        source dataset
    moments : list
        list of moment strings

    Returns
    -------
    ds : xarray Dataset
        altered dataset
    """

    standard = kwargs.get("standard", "cf-mandatory")
    mask_and_scale = kwargs.get("mask_and_scale", True)
    decode_coords = kwargs.get("decode_coords", True)

    # fix dimensions
    dims = sorted(list(ds.dims.keys()), key=lambda x: int(x[len("phony_dim_") :]))

    ds = ds.rename({dims[0]: "dim_0", dims[1]: "dim_1"})

    for mom in moments:
        # open dataX dataset
        dmom = ds[mom]
        name = dmom.moment.lower()
        if "cf" in standard and name not in GAMIC_NAMES.keys():
            ds = ds.drop_vars(mom)
            continue

        # extract attributes
        dmax = np.iinfo(dmom.dtype).max
        dmin = np.iinfo(dmom.dtype).min
        minval = dmom.dyn_range_min
        maxval = dmom.dyn_range_max
        dtype = minval.dtype
        gain = (maxval - minval) / (dmax - 1)
        offset = minval - gain

        gain = gain.astype(dtype)
        offset = offset.astype(dtype)
        undetect = np.array([dmin])[0].astype(dtype)

        # create attribute dict
        attrs = collections.OrderedDict()
        # clean moment attributes
        if standard != "none":
            dmom.attrs = collections.OrderedDict()

        if standard in ["odim"]:
            attrs["gain"] = gain
            attrs["offset"] = offset
            attrs["nodata"] = undetect
            attrs["undetect"] = undetect

        # add cfradial moment attributes
        if "cf" in standard or mask_and_scale:
            attrs["scale_factor"] = gain
            attrs["add_offset"] = minval
            attrs["_FillValue"] = float(dmax)

        if "cf" in standard or decode_coords:
            attrs["coordinates"] = "elevation azimuth range"

        if "full" in standard:
            attrs["_Undetect"] = undetect

        if "cf" in standard:
            cfname = GAMIC_NAMES[name]
            for k, v in moments_mapping[cfname].items():
                attrs[k] = v
            name = attrs.pop("short_name")
            attrs.pop("gamic")

        # assign attributes to moment
        dmom.attrs.update(attrs)

        # keep original dataset name
        if standard != "none":
            ds = ds.rename({mom: name.upper()})

    return ds


def _get_odim_group_moments(nch, sweep, moments=None, **kwargs):
    """Retrieve radar moments from hdf groups.

    Parameters
    ----------
    nch : netCDF Dataset handle
    sweep : str
        sweep key
    moments : list
        list of moment strings

    Returns
    -------
    ds : dictionary
        moment datasets
    """

    standard = kwargs.get("standard", "cf-mandatory")
    mask_and_scale = kwargs.get("mask_and_scale", True)
    decode_coords = kwargs.get("decode_coords", True)
    chunks = kwargs.get("chunks", None)

    datas = {}
    for mom in moments:
        dmom_what = _open_dataset(nch[sweep][mom], "what", chunks=chunks)
        name = dmom_what.attrs.pop("quantity")
        if "cf" in standard and name not in moments_mapping.keys():
            continue
        dsmom = _open_dataset(nch[sweep], mom, chunks=chunks)

        # create attribute dict
        attrs = collections.OrderedDict()

        if standard in ["odim"]:
            attrs.update(dmom_what.attrs)

        # add cfradial moment attributes
        if "cf" in standard or mask_and_scale:
            attrs["scale_factor"] = dmom_what.attrs.get("gain")
            attrs["add_offset"] = dmom_what.attrs.get("offset")
            attrs["_FillValue"] = dmom_what.attrs.get("nodata")
        if "cf" in standard or decode_coords:
            attrs["coordinates"] = "elevation azimuth range"
        if "cf" in standard:
            for k, v in moments_mapping[name].items():
                attrs[k] = v
            # drop short_name
            attrs.pop("short_name")
            attrs.pop("gamic")
        if "full" in standard:
            attrs["_Undetect"] = dmom_what.attrs.get("undetect")

        # assign attributes
        dmom = dsmom.data.assign_attrs(attrs)

        # keep original dataset name
        if standard == "none":
            name = mom

        # fix dimensions
        dims = dmom.dims
        datas.update({name: dmom.rename({dims[0]: "dim_0", dims[1]: "dim_1"})})
    return datas


def _get_odim_groups(ncf, groups, **kwargs):
    """Get hdf groups.

    Parameters
    ----------
    ncf : netCDf4 Dataset handle
    groups : list
        list of groups-keys

    Returns
    -------
    out : tuple
        tuple of xarray datasets
    """
    return tuple(map(lambda x: _open_dataset(ncf, x, **kwargs), groups))


def _get_odim_moment_names(sweep, fmt=None, src=None):
    """Get moment names.

    Parameters
    ----------
    sweep : netCDf4 Group handle
    fmt : str
        dataset descriptor format
    src : str
        dataset location

    Returns
    -------
    out : :class:`numpy:numpy.ndarray`
        array of moment names
    """
    moments = [mom for mom in getattr(sweep, src).keys() if fmt in mom]
    moments_idx = np.argsort([int(s[len(fmt) :]) for s in moments])
    return np.array(moments)[moments_idx]


def _assign_odim_moments(ds, nch, sweep, **kwargs):
    """Assign radar moments to dataset.

    Parameters
    ----------
    nch : netCDF4.Dataset handle
        source netCDF4 Dataset
    ds : xarray dataset
        destination dataset
    sweep : str
        netcdf group name

    Keyword Arguments
    -----------------
    decode_times : bool
        If True, decode cf times to np.datetime64. Defaults to True.
    decode_coords : bool
        If True, use the ‘coordinates’ attribute on variable
        to assign coordinates. Defaults to True.
    mask_and_scale : bool
        If True, lazily scale (using scale_factor and add_offset)
        and mask (using _FillValue). Defaults to True.
    georef : bool
        If True, adds 2D AEQD x,y,z-coordinates, ground_range (gr) and
        2D (rays,bins)-coordinates for easy georeferencing (eg. cartopy)
    standard : str
        * `none` - data is read as verbatim as possible, no metadata
        * `odim` - data is read, odim metadata added to datasets
        * `cf-mandatory` - data is read according to cfradial2 standard
          importing mandatory metadata
        * `cf-full` - data is read according to cfradial2 standard
          importing all available cfradial2 metadata (not fully
          implemented)

    Returns
    -------
    ds : xarray dataset
        Dataset with assigned radar moments
    """
    moments = _get_odim_moment_names(nch.nch[sweep], fmt=nch._mfmt, src=nch._msrc)
    if nch.flavour == "ODIM":
        for name, dmom in _get_odim_group_moments(
            nch.nch, sweep, moments=moments, **kwargs
        ).items():
            ds[name] = dmom
    if nch.flavour == "GAMIC":
        ds = _get_odim_variables_moments(ds, moments=moments, **kwargs)

    return ds


def _get_odim_timevalues(nch, grps):
    """Retrieve TimeArray from source data.

    Parameters
    ----------
    nch : netCDF4.Dataset handle
        source netCDF4 Dataset
    grps : dict
        Dictionary of dataset hdf5 groups ('how', 'what', 'where')

    Returns
    -------
    timevals : :class:`numpy:numpy.ndarray`
            array of time values
    """
    if nch.flavour == "ODIM":
        try:
            timevals = grps["how"].odim.time_range
        except (KeyError, AttributeError):
            # timehandling if only start and end time is given
            start, end = grps["what"].odim.time_range2
            if start == end:
                warnings.warn(
                    "WRADLIB: Equal ODIM `starttime` and `endtime` "
                    "values. Can't determine correct sweep start-, "
                    "end- and raytimes.",
                    UserWarning,
                )
                timevals = np.ones(grps["where"].nrays) * start
            else:
                delta = (end - start) / grps["where"].nrays
                timevals = np.arange(start + delta / 2.0, end, delta)
                timevals = np.roll(timevals, shift=-grps["where"].a1gate)
    if nch.flavour == "GAMIC":
        timevals = grps["what"].gamic.time_range.values

    return timevals


def _get_odim_coordinates(nch, grps):
    """Retrieve coordinates according OdimH5 standard.

    Parameters
    ----------
    nch : netCDF4.Dataset handle
        source netCDF4 Dataset
    grps : dict
        Dictionary of dataset hdf5 groups ('how', 'what', 'where')

    Returns
    -------
    coords : dict
        Dictionary of coordinate arrays
    """
    flavour = nch.flavour.lower()
    coords = collections.OrderedDict()
    if flavour == "odim":
        rng = grps["where"]
        az = el = grps["how"]
    if flavour == "gamic":
        az = el = grps["what"]
        rng = grps["how"]
    try:
        coords["azimuth"] = getattr(az, flavour).azimuth_range
        coords["elevation"] = getattr(el, flavour).elevation_range
    except (KeyError, AttributeError):
        az = el = grps["where"]
        coords["azimuth"] = getattr(az, flavour).azimuth_range2
        coords["elevation"] = getattr(el, flavour).elevation_range2
    coords["range"] = getattr(rng, flavour).radial_range

    return coords


def _get_odim_sweep_mode(nch, grp):
    """Retrieve sweep mode

    Parameters
    ----------
    nch : netCDF4.Dataset handle
    grp : dict
        Dictionary of dataset hdf5 groups ('how', 'what', 'where')

    Returns
    -------
    out : str
        'azimuth_surveillance' or 'rhi'

    """
    odim_mode = grp["what"].attrs[nch._swmode]
    return "rhi" if odim_mode == "RHI" else "azimuth_surveillance"


def _get_odim_full_vars(nch, grps):
    """Retrieve available non mandatory variables from source data.

    Parameters
    ----------
    nch : netCDF4.Dataset handle
    grps : dict
        Dictionary of dataset hdf5 groups ('how', 'what', 'where')

    Returns
    -------
    full_vars : dict
        full cf-variables
    """
    full_vars = collections.OrderedDict()
    if nch.flavour == "ODIM":
        for k, v in cf_full_vars.items():
            full_vars[k] = getattr(getattr(grps["how"], nch.flavour.lower()), k)
    if nch.flavour == "GAMIC":
        pass

    return full_vars


def _get_odim_fixed_angle(nch, grps):
    """Retrieve fixed angle from source data.

    Parameters
    ----------
    nch : netCDF4.Dataset handle
    grps : dict
        Dictionary of dataset hdf5 groups ('how', 'what', 'where')

    Returns
    -------
    fixed-angle : float
        fixed angle of specific scan
    """
    mode = _get_odim_sweep_mode(nch, grps)
    if nch.flavour == "ODIM":
        ang = {"azimuth_surveillance": "elangle", "rhi": "azangle"}
        fixed_angle = getattr(grps["where"], ang[mode])
    if nch.flavour == "GAMIC":
        ang = {"azimuth_surveillance": "elevation", "rhi": "azimuth"}
        fixed_angle = grps["how"].attrs[ang[mode]]
    return fixed_angle


def _get_odim_root_attributes(nch, grps):
    """Retrieve root attributes according CfRadial2 standard.

    Parameters
    ----------
    grps : dict
        Dictionary of root hdf5 groups ('how', 'what', 'where')

    Returns
    -------
    attrs : dict
        Dictionary of root attributes
    """

    attrs = collections.OrderedDict()
    attrs.update(
        {
            "version": "None",
            "title": "None",
            "institution": "None",
            "references": "None",
            "source": "None",
            "history": "None",
            "comment": "im/exported using wradlib",
            "instrument_name": "None",
        }
    )
    attrs["version"] = grps["what"].attrs["version"]

    if nch.flavour == "ODIM":
        attrs["institution"] = grps["what"].attrs["source"]
        attrs["instrument"] = grps["what"].attrs["source"]
    if nch.flavour == "GAMIC":
        attrs["title"] = grps["how"].attrs["template_name"]
        attrs["instrument"] = grps["how"].attrs["host_name"]

    return attrs
