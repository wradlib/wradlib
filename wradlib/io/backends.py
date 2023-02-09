#!/usr/bin/env python
# Copyright (c) 2021-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Xarray backends
^^^^^^^^^^^^^^^
Reading radar data into xarray Datasets using ``xarray.open_dataset``
and ``xarray.open_mfdataset``.

Backends have been moved to ``xradar`` package. Here we keep stubs for backwards
compatibility.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "CfRadial1BackendEntrypoint",
    "CfRadial2BackendEntrypoint",
    "FurunoBackendEntrypoint",
    "GamicBackendEntrypoint",
    "IrisBackendEntrypoint",
    "OdimBackendEntrypoint",
    "RadolanBackendEntrypoint",
    "RainbowBackendEntrypoint",
]

__doc__ = __doc__.format("\n   ".join(__all__))

import io
import warnings

import numpy as np
from xarray.backends import NetCDF4DataStore
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
)
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import SerializableLock, ensure_lock
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import Frozen, FrozenDict, close_on_error
from xarray.core.variable import Variable
from xradar.io.backends import CfRadial1BackendEntrypoint as XCfRadial1BackendEntrypoint
from xradar.io.backends import FurunoBackendEntrypoint as XFurunoBackendEntrypoint
from xradar.io.backends import GamicBackendEntrypoint as XGamicBackendEntrypoint
from xradar.io.backends import IrisBackendEntrypoint as XIrisBackendEntrypoint
from xradar.io.backends import OdimBackendEntrypoint as XOdimBackendEntrypoint
from xradar.io.backends import RainbowBackendEntrypoint as XRainbowBackendEntrypoint
from xradar.io.backends.common import _fix_angle

from wradlib.io.radolan import _radolan_file
from wradlib.io.xarray import _assign_data_radial2
from wradlib.util import has_import, import_optional

h5netcdf = import_optional("h5netcdf")
netCDF4 = import_optional("netCDF4")
dask = import_optional("dask")

RADOLAN_LOCK = SerializableLock()


class RadolanArrayWrapper(BackendArray):
    """Wraps array of RADOLAN data."""

    def __init__(self, datastore, name, array):
        self.datastore = datastore
        self.name = name
        self.shape = array.shape
        self.dtype = array.dtype

    def _getitem(self, key):
        return self.datastore.ds.data[self.name][key]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._getitem,
        )


class RadolanDataStore(AbstractDataStore):
    """Implements ``xarray.AbstractDataStore`` read-only API for a RADOLAN files."""

    def __init__(
        self, filename_or_obj, lock=None, fillmissing=False, copy=False, ancillary=False
    ):
        if lock is None:
            lock = RADOLAN_LOCK
        self.lock = ensure_lock(lock)

        if isinstance(filename_or_obj, str):
            manager = CachingFileManager(
                _radolan_file,
                filename_or_obj,
                lock=lock,
                kwargs={
                    "fillmissing": fillmissing,
                    "copy": copy,
                    "ancillary": ancillary,
                },
            )
        else:
            if isinstance(filename_or_obj, bytes):
                filename_or_obj = io.BytesIO(filename_or_obj)
            dataset = _radolan_file(
                filename_or_obj, fillmissing=fillmissing, copy=copy, ancillary=ancillary
            )
            manager = DummyFileManager(dataset)

        self._manager = manager
        self._filename = self.ds.filename

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as ds:
            return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        encoding = {"source": self._filename}
        vdata = var.data
        if isinstance(vdata, np.ndarray):
            data = vdata
        else:
            data = indexing.LazilyOuterIndexedArray(
                RadolanArrayWrapper(self, name, vdata)
            )
        return Variable(var.dimensions, data, var.attributes, encoding)

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        )

    def get_attrs(self):
        return Frozen(self.ds.attributes)

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)

    def get_encoding(self):
        dims = self.get_dimensions()
        encoding = {"unlimited_dims": {k for k, v in dims.items() if v is None}}
        return encoding

    def close(self, **kwargs):
        self._manager.close(**kwargs)


class RadolanBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for RADOLAN data."""

    description = "Open RADOLAN in Xarray"
    url = "https://docs.wradlib.org/en/stable/notebooks/fileio/wradlib_radolan_backend.html"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        fillmissing=False,
        copy=False,
        ancillary=False,
    ):
        store = RadolanDataStore(
            filename_or_obj,
            fillmissing=fillmissing,
            copy=copy,
            ancillary=ancillary,
        )
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        return ds


class OdimBackendEntrypoint(XOdimBackendEntrypoint):
    """Xarray BackendEntrypoint for ODIM data."""

    available = has_import(h5netcdf)
    description = "Open ODIM_H5 (.h5, .hdf5) using h5netcdf in Xarray"
    url = (
        "https://docs.wradlib.org/en/stable/notebooks/fileio/wradlib_odim_backend.html"
    )
    name = "wradlib-odim"

    def open_dataset(
        self,
        *args,
        **kwargs,
    ):
        warnings.warn(
            "`wradlib-odim` xarray backend has been moved to `xradar` and will be removed in 2.0. "
            "Use `odim` backend from `xradar`-package.",
            category=FutureWarning,
            stacklevel=2,
        )
        reindex_angle = kwargs.get("reindex_angle", None)
        if reindex_angle is not False and not isinstance(reindex_angle, dict):
            if reindex_angle is True or reindex_angle is None:
                reindex_angle = 0.4
            reindex_angle = dict(
                tolerance=reindex_angle,
                start_angle=0,
                stop_angle=360,
                angle_res=1.0,
                direction=1,
            )
            kwargs["reindex_angle"] = reindex_angle
        keep_azimuth = kwargs.pop("keep_azimuth", False)
        keep_elevation = kwargs.pop("keep_elevation", False)
        ds = super().open_dataset(*args, **kwargs)
        try:
            ds.attrs["fixed_angle"] = ds.sweep_fixed_angle.values
        except AttributeError:
            ds.attrs["fixed_angle"] = ds.fixed_angle.values

        if not keep_azimuth:
            if ds.azimuth.dims[0] == "elevation":
                ds = ds.assign_coords({"azimuth": ds.azimuth.pipe(_fix_angle)})
        if not keep_elevation:
            if ds.elevation.dims[0] == "azimuth":
                ds = ds.assign_coords({"elevation": ds.elevation.pipe(_fix_angle)})

        ds = ds.rename({"time": "rtime"})
        ds = ds.assign_coords({"time": ds.rtime.min()})

        # backwards compat
        ds = ds.assign_coords({"sweep_mode": ds.sweep_mode.reset_coords(drop=True)})

        return ds


class GamicBackendEntrypoint(XGamicBackendEntrypoint):
    """Xarray BackendEntrypoint for GAMIC data."""

    available = has_import(h5netcdf)
    description = "Open GAMIC HDF5 (.h5, .hdf5, .mvol) using h5netcdf in Xarray"
    url = (
        "https://docs.wradlib.org/en/stable/notebooks/fileio/wradlib_gamic_backend.html"
    )
    name = "wradlib-gamic"

    def open_dataset(
        self,
        *args,
        **kwargs,
    ):
        warnings.warn(
            "`wradlib-gamic` xarray backend has been moved to `xradar` and will be removed in 2.0. "
            "Use `gamic` backend from `xradar`-package.",
            category=FutureWarning,
            stacklevel=3,
        )
        reindex_angle = kwargs.get("reindex_angle", None)
        if reindex_angle is not False and not isinstance(reindex_angle, dict):
            if reindex_angle is True or reindex_angle is None:
                reindex_angle = 0.4
            reindex_angle = dict(
                tolerance=reindex_angle,
                start_angle=0,
                stop_angle=360,
                angle_res=1.0,
                direction=1,
            )
            kwargs["reindex_angle"] = reindex_angle
        keep_azimuth = kwargs.pop("keep_azimuth", False)
        keep_elevation = kwargs.pop("keep_elevation", False)
        ds = super().open_dataset(*args, **kwargs)
        try:
            ds.attrs["fixed_angle"] = ds.sweep_fixed_angle.values
        except AttributeError:
            ds.attrs["fixed_angle"] = ds.fixed_angle.values

        if not keep_azimuth:
            if ds.azimuth.dims[0] == "elevation":
                ds = ds.assign_coords({"azimuth": ds.azimuth.pipe(_fix_angle)})
        if not keep_elevation:
            if ds.elevation.dims[0] == "azimuth":
                ds = ds.assign_coords({"elevation": ds.elevation.pipe(_fix_angle)})

        ds = ds.rename({"time": "rtime"})
        ds = ds.assign_coords({"time": ds.rtime.min()})

        # backwards compat
        ds = ds.assign_coords({"sweep_mode": ds.sweep_mode.reset_coords(drop=True)})

        return ds


class CfRadial1BackendEntrypoint(XCfRadial1BackendEntrypoint):
    """Xarray BackendEntrypoint for CfRadial1 data."""

    available = has_import(netCDF4)
    description = "Open CfRadial1 (.nc, .nc4) using netCDF4 in Xarray"
    url = "https://docs.wradlib.org/en/stable/notebooks/fileio/wradlib_cfradial1_backend.html"
    name = "wradlib-cfradial1"

    def open_dataset(
        self,
        *args,
        **kwargs,
    ):
        warnings.warn(
            "`wradlib-cfradial1` xarray backend has been moved to `xradar` and will be removed in 2.0. "
            "Use `cfradial1` backend from `xradar`-package.",
            category=FutureWarning,
            stacklevel=2,
        )
        reindex_angle = kwargs.get("reindex_angle", None)
        if reindex_angle is not False and not isinstance(reindex_angle, dict):
            if reindex_angle is True or reindex_angle is None:
                reindex_angle = 0.4
            reindex_angle = dict(
                tolerance=reindex_angle,
                start_angle=0,
                stop_angle=360,
                angle_res=1.0,
                direction=1,
            )
            kwargs["reindex_angle"] = reindex_angle
        ds = super().open_dataset(*args, **kwargs)
        try:
            ds.attrs["fixed_angle"] = ds.sweep_fixed_angle.values
        except AttributeError:
            ds.attrs["fixed_angle"] = ds.fixed_angle.values

        ds = ds.rename({"time": "rtime"})
        ds = ds.assign_coords({"time": ds.rtime.min()})

        # backwards compat
        ds = ds.assign_coords({"sweep_mode": ds.sweep_mode.reset_coords(drop=True)})

        return ds


class CfRadial2BackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for CfRadial2 data."""

    available = has_import(netCDF4)
    description = "Open CfRadial2 (.nc, .nc4) using netCDF4 in Xarray"
    url = "https://docs.wradlib.org/en/stable/notebooks/fileio/wradlib_cfradial2_backend.html"
    name = "wradlib-cfradial2"

    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        format=None,
        group=None,
    ):
        warnings.warn(
            "`wradlib-cfradial2` xarray backend has been moved to `xradar` and will be removed in 2.0. "
            "Use `xradar`-package.",
            category=FutureWarning,
            stacklevel=2,
        )
        if isinstance(filename_or_obj, io.IOBase):
            filename_or_obj.seek(0)

        store = NetCDF4DataStore.open(
            filename_or_obj,
            format=format,
            group=group,
            lock=None,
        )

        store_entrypoint = StoreBackendEntrypoint()

        with close_on_error(store):
            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )

        if group is not None:
            ds = _assign_data_radial2(ds)
            dim0 = list(set(ds.dims) & {"azimuth", "elevation"})[0]
            ds = ds.sortby(dim0)

        # backwards compat
        ds = ds.assign_coords({"sweep_mode": ds.sweep_mode.reset_coords(drop=True)})

        return ds


class IrisBackendEntrypoint(XIrisBackendEntrypoint):
    """Xarray BackendEntrypoint for IRIS/Sigmet data."""

    description = "Open IRIS/Sigmet files in Xarray"
    url = (
        "https://docs.wradlib.org/en/stable/notebooks/fileio/wradlib_iris_backend.html"
    )
    name = "wradlib-iris"

    def open_dataset(
        self,
        *args,
        **kwargs,
    ):
        warnings.warn(
            "`wradlib-iris` xarray backend has been moved to `xradar` and will be removed in 2.0. "
            "Use `iris` backend from `xradar`-package.",
            category=FutureWarning,
            stacklevel=2,
        )
        reindex_angle = kwargs.get("reindex_angle", None)
        if reindex_angle is not False and not isinstance(reindex_angle, dict):
            if reindex_angle is True or reindex_angle is None:
                reindex_angle = 0.4
            reindex_angle = dict(
                tolerance=reindex_angle,
                start_angle=0,
                stop_angle=360,
                angle_res=1.0,
                direction=1,
            )
            kwargs["reindex_angle"] = reindex_angle
        ds = super().open_dataset(*args, **kwargs)
        try:
            ds.attrs["fixed_angle"] = ds.sweep_fixed_angle.values
        except AttributeError:
            ds.attrs["fixed_angle"] = ds.fixed_angle.values

        ds = ds.rename({"time": "rtime"})
        ds = ds.assign_coords({"time": ds.rtime.min()})

        # backwards compat
        ds = ds.assign_coords({"sweep_mode": ds.sweep_mode.reset_coords(drop=True)})

        return ds


class RainbowBackendEntrypoint(XRainbowBackendEntrypoint):
    """Xarray BackendEntrypoint for Rainbow5 data."""

    description = "Open Rainbow5 files in Xarray"
    url = "https://docs.wradlib.org/en/stable/notebooks/fileio/wradlib_rainbow_backend.html"
    name = "wradlib-rainbow"

    def open_dataset(
        self,
        *args,
        **kwargs,
    ):
        warnings.warn(
            "`wradlib-rainbow` xarray backend has been moved to `xradar` and will be removed in 2.0. "
            "Use `rainbow` backend from `xradar`-package.",
            category=FutureWarning,
            stacklevel=2,
        )
        reindex_angle = kwargs.get("reindex_angle", None)
        if reindex_angle is not False and not isinstance(reindex_angle, dict):
            if reindex_angle is True or reindex_angle is None:
                reindex_angle = 0.4
            reindex_angle = dict(
                tolerance=reindex_angle,
                start_angle=0,
                stop_angle=360,
                angle_res=1.0,
                direction=1,
            )
            kwargs["reindex_angle"] = reindex_angle
        ds = super().open_dataset(*args, **kwargs)
        try:
            ds.attrs["fixed_angle"] = ds.sweep_fixed_angle.values
        except AttributeError:
            ds.attrs["fixed_angle"] = ds.fixed_angle.values

        ds = ds.rename({"time": "rtime"})
        ds = ds.assign_coords({"time": ds.rtime.min()})

        # backwards compat
        ds = ds.assign_coords({"sweep_mode": ds.sweep_mode.reset_coords(drop=True)})

        return ds


class FurunoBackendEntrypoint(XFurunoBackendEntrypoint):
    """Xarray BackendEntrypoint for Furuno data."""

    description = "Open FURUNO (.scn, .scnx) in Xarray"
    url = "https://docs.wradlib.org/en/stable/notebooks/fileio/wradlib_furuno_backend.html"
    name = "wradlib-furuno"

    def open_dataset(
        self,
        *args,
        **kwargs,
    ):
        warnings.warn(
            "`wradlib-furuno` xarray backend has been moved to `xradar` and will be removed in 2.0. "
            "Use `furuno` backend from `xradar`-package.",
            category=FutureWarning,
            stacklevel=2,
        )
        reindex_angle = kwargs.get("reindex_angle", None)
        if reindex_angle is not False and not isinstance(reindex_angle, dict):
            if reindex_angle is True or reindex_angle is None:
                reindex_angle = 0.4
            reindex_angle = dict(
                tolerance=reindex_angle,
                start_angle=0,
                stop_angle=360,
                angle_res=1.0,
                direction=1,
            )
            kwargs["reindex_angle"] = reindex_angle
        ds = super().open_dataset(*args, **kwargs)
        try:
            ds.attrs["fixed_angle"] = ds.sweep_fixed_angle.values
        except AttributeError:
            ds.attrs["fixed_angle"] = ds.fixed_angle.values

        ds = ds.rename({"time": "rtime"})
        ds = ds.assign_coords({"time": ds.rtime.min()})

        # backwards compat
        ds = ds.assign_coords({"sweep_mode": ds.sweep_mode.reset_coords(drop=True)})

        return ds
