#!/usr/bin/env python
# Copyright (c) 2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Xarray backends
^^^^^^^^^^^^^^^
Reading radar data into xarray Datasets using ``xarray.open_dataset``
and ``xarray.open_mfdataset``.

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

import datetime as dt
import io

import numpy as np
from packaging.version import Version
from xarray import Dataset
from xarray.backends import NetCDF4DataStore
from xarray.backends.common import (
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    find_root_and_group,
)
from xarray.backends.file_manager import CachingFileManager, DummyFileManager
from xarray.backends.locks import SerializableLock, ensure_lock
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.utils import Frozen, FrozenDict, close_on_error, is_remote_uri
from xarray.core.variable import Variable

from wradlib.io.furuno import FurunoFile
from wradlib.io.iris import IrisRawFile
from wradlib.io.radolan import _radolan_file
from wradlib.io.rainbow import RainbowFile
from wradlib.io.xarray import (
    _assign_data_radial,
    _assign_data_radial2,
    _fix_angle,
    _GamicH5NetCDFMetadata,
    _get_gamic_variable_name_and_attrs,
    _get_odim_variable_name_and_attrs,
    _OdimH5NetCDFMetadata,
    _reindex_angle,
    az_attrs_template,
    el_attrs_template,
    iris_mapping,
    moment_attrs,
    moments_mapping,
    rainbow_mapping,
    range_attrs,
    time_attrs,
)
from wradlib.util import has_import, import_optional

h5netcdf = import_optional("h5netcdf")
netCDF4 = import_optional("netCDF4")
dask = import_optional("dask")


RADOLAN_LOCK = SerializableLock()
HDF5_LOCK = SerializableLock()


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


class H5NetCDFArrayWrapper(BackendArray):
    """H5NetCDFArrayWrapper

    adapted from https://github.com/pydata/xarray/
    """

    __slots__ = ("datastore", "dtype", "shape", "variable_name")

    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

        array = self.get_array()
        self.shape = array.shape

        dtype = array.dtype
        if dtype is str:
            # use object dtype because that's the only way in numpy to
            # represent variable length strings; it also prevents automatic
            # string concatenation via conventions.decode_cf_variable
            dtype = np.dtype("O")
        self.dtype = dtype

    def __setitem__(self, key, value):
        with self.datastore.lock:
            data = self.get_array(needs_lock=False)
            data[key] = value
            if self.datastore.autoclose:
                self.datastore.close(needs_lock=False)

    def get_array(self, needs_lock=True):
        ds = self.datastore._acquire(needs_lock)
        return ds.variables[self.variable_name]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.OUTER_1VECTOR, self._getitem
        )

    def _getitem(self, key):
        # h5py requires using lists for fancy indexing:
        # https://github.com/h5py/h5py/issues/992
        key = tuple(list(k) if isinstance(k, np.ndarray) else k for k in key)
        with self.datastore.lock:
            array = self.get_array(needs_lock=False)
            return array[key]


def _get_h5netcdf_encoding(self, var):
    """get encoding from h5netcdf Variable

    adapted from https://github.com/pydata/xarray/
    """
    import h5py

    # netCDF4 specific encoding
    encoding = {
        "chunksizes": var.chunks,
        "fletcher32": var.fletcher32,
        "shuffle": var.shuffle,
    }

    # Convert h5py-style compression options to NetCDF4-Python
    # style, if possible
    if var.compression == "gzip":
        encoding["zlib"] = True
        encoding["complevel"] = var.compression_opts
    elif var.compression is not None:
        encoding["compression"] = var.compression
        encoding["compression_opts"] = var.compression_opts

    # save source so __repr__ can detect if it's local or not
    encoding["source"] = self._filename
    encoding["original_shape"] = var.shape

    vlen_dtype = h5py.check_dtype(vlen=var.dtype)
    if vlen_dtype is str:
        encoding["dtype"] = str
    elif vlen_dtype is not None:  # pragma: no cover
        # xarray doesn't support writing arbitrary vlen dtypes yet.
        pass
    else:
        encoding["dtype"] = var.dtype
    return encoding


class OdimSubStore(AbstractDataStore):
    """Store for reading ODIM data-moments via h5netcdf."""

    def __init__(
        self,
        store,
        group=None,
        lock=False,
    ):

        if not isinstance(store, OdimStore):
            raise TypeError(
                f"Wrong type {type(store)} for parameter store, "
                f"expected 'OdimStore'."
            )

        self._manager = store._manager
        self._group = group
        self._filename = store.filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return _OdimH5NetCDFMetadata(root, self._group.lstrip("/"))

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = root[self._group.lstrip("/")]
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):

        dimensions = self.root.get_variable_dimensions(var.dimensions)
        data = indexing.LazilyOuterIndexedArray(H5NetCDFArrayWrapper(name, self))
        encoding = _get_h5netcdf_encoding(self, var)
        encoding["group"] = self._group
        name, attrs = _get_odim_variable_name_and_attrs(name, self.root.what)

        return name, Variable(dimensions, data, attrs, encoding)

    def open_store_coordinates(self):
        return self.root.coordinates

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **dict(
                    [
                        self.open_store_variable(k, v)
                        for k, v in self.ds.variables.items()
                    ]
                ),
            }.items()
        )


class OdimStore(AbstractDataStore):
    """Store for reading ODIM dataset groups via h5netcdf."""

    def __init__(self, manager, group=None, lock=False):

        if isinstance(manager, (h5netcdf.File, h5netcdf.Group)):
            if group is None:
                root, group = find_root_and_group(manager)
            else:
                if type(manager) is not h5netcdf.File:
                    raise ValueError(
                        "must supply a h5netcdf.File if the group "
                        "argument is provided"
                    )
                root = manager
            manager = DummyFileManager(root)

        self._manager = manager
        self._group = group
        self._filename = self.filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
        self._substore = None
        self._need_time_recalc = False

    @classmethod
    def open(
        cls,
        filename,
        mode="r",
        format=None,
        group=None,
        lock=None,
        invalid_netcdf=None,
        phony_dims=None,
        decode_vlen_strings=True,
    ):
        if isinstance(filename, bytes):
            raise ValueError(
                "can't open netCDF4/HDF5 as bytes "
                "try passing a path or file-like object"
            )

        if format not in [None, "NETCDF4"]:
            raise ValueError("invalid format for h5netcdf backend")

        kwargs = {"invalid_netcdf": invalid_netcdf}
        if phony_dims is not None:
            if Version(h5netcdf.__version__) >= Version("0.8.0"):
                kwargs["phony_dims"] = phony_dims
            else:
                raise ValueError(
                    "h5netcdf backend keyword argument 'phony_dims' needs "
                    "h5netcdf >= 0.8.0."
                )
        if Version(h5netcdf.__version__) >= Version("0.10.0") and Version(
            h5netcdf.core.h5py.__version__
        ) >= Version("3.0.0"):
            kwargs["decode_vlen_strings"] = decode_vlen_strings

        if lock is None:
            if has_import(dask):
                lock = HDF5_LOCK
            else:
                lock = False

        manager = CachingFileManager(h5netcdf.File, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group, lock=lock)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def substore(self):
        if self._substore is None:
            with self._manager.acquire_context(False) as root:
                subgroups = [
                    "/".join([self._group, k])
                    for k in root[self._group].groups
                    # get data and quality groups
                    if "data" or "quality" in k
                ]
                substore = []
                substore.extend(
                    [
                        OdimSubStore(
                            self,
                            group=group,
                            lock=self.lock,
                        )
                        for group in subgroups
                    ]
                )
                self._substore = substore

        return self._substore

    def open_store_coordinates(self):
        return self.substore[0].open_store_coordinates()

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **dict(
                    [
                        (k, v)
                        for substore in self.substore
                        for k, v in substore.get_variables().items()
                    ]
                ),
                **self.open_store_coordinates(),
            }.items()
        )

    def get_attrs(self):
        dim, angle = self.substore[0].root.fixed_dim_and_angle
        attributes = {}
        attributes["fixed_angle"] = angle.item()
        return FrozenDict(attributes)


class OdimBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for ODIM data."""

    available = has_import(h5netcdf)

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
        group="dataset1",
        invalid_netcdf=None,
        phony_dims="access",
        decode_vlen_strings=True,
        keep_elevation=False,
        keep_azimuth=False,
        reindex_angle=None,
    ):

        if isinstance(filename_or_obj, io.IOBase):
            filename_or_obj.seek(0)

        store = OdimStore.open(
            filename_or_obj,
            format=format,
            group=group,
            invalid_netcdf=invalid_netcdf,
            phony_dims=phony_dims,
            decode_vlen_strings=decode_vlen_strings,
        )

        store_entrypoint = StoreBackendEntrypoint()

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

        ds.encoding["engine"] = "odim"

        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(_reindex_angle, store=store, tol=reindex_angle)

        if not keep_azimuth:
            if ds.azimuth.dims[0] == "elevation":
                ds = ds.assign_coords({"azimuth": ds.azimuth.pipe(_fix_angle)})
        if not keep_elevation:
            if ds.elevation.dims[0] == "azimuth":
                ds = ds.assign_coords({"elevation": ds.elevation.pipe(_fix_angle)})

        return ds


class GamicStore(AbstractDataStore):
    """Store for reading ODIM dataset groups via h5netcdf."""

    def __init__(self, manager, group=None, lock=False):

        if isinstance(manager, (h5netcdf.File, h5netcdf.Group)):
            if group is None:
                root, group = find_root_and_group(manager)
            else:
                if type(manager) is not h5netcdf.File:
                    raise ValueError(
                        "must supply a h5netcdf.File if the group "
                        "argument is provided"
                    )
                root = manager
            manager = DummyFileManager(root)

        self._manager = manager
        self._group = group
        self._filename = self.filename
        self.is_remote = is_remote_uri(self._filename)
        self.lock = ensure_lock(lock)
        self._need_time_recalc = False

    @classmethod
    def open(
        cls,
        filename,
        mode="r",
        format=None,
        group=None,
        lock=None,
        invalid_netcdf=None,
        phony_dims=None,
        decode_vlen_strings=True,
    ):
        if isinstance(filename, bytes):
            raise ValueError(
                "can't open netCDF4/HDF5 as bytes "
                "try passing a path or file-like object"
            )

        if format not in [None, "NETCDF4"]:
            raise ValueError("invalid format for h5netcdf backend")

        kwargs = {"invalid_netcdf": invalid_netcdf}
        if phony_dims is not None:
            if Version(h5netcdf.__version__) >= Version("0.8.0"):
                kwargs["phony_dims"] = phony_dims
            else:
                raise ValueError(
                    "h5netcdf backend keyword argument 'phony_dims' needs "
                    "h5netcdf >= 0.8.0."
                )
        if Version(h5netcdf.__version__) >= Version("0.10.0") and Version(
            h5netcdf.core.h5py.__version__
        ) >= Version("3.0.0"):
            kwargs["decode_vlen_strings"] = decode_vlen_strings

        if lock is None:
            if has_import(dask):
                lock = HDF5_LOCK
            else:
                lock = False

        manager = CachingFileManager(h5netcdf.File, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group, lock=lock)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return _GamicH5NetCDFMetadata(root, self._group.lstrip("/"))

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = root[self._group.lstrip("/")]
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        dimensions = self.root.get_variable_dimensions(var.dimensions)
        data = indexing.LazilyOuterIndexedArray(H5NetCDFArrayWrapper(name, self))
        encoding = _get_h5netcdf_encoding(self, var)
        encoding["group"] = self._group
        # cheat attributes
        if "moment" in name:
            name, attrs = _get_gamic_variable_name_and_attrs({**var.attrs}, var.dtype)
        elif "ray_header" in name:
            return self.root.coordinates(dimensions, data, encoding)
        else:
            return {}
        return {name: Variable(dimensions, data, attrs, encoding)}

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k, v in self.ds.variables.items()
            for k1, v1 in {
                **self.open_store_variable(k, v),
            }.items()
        )

    def get_attrs(self):
        dim, angle = self.root.fixed_dim_and_angle
        attributes = {"fixed_angle": angle.item()}
        return FrozenDict(attributes)


class GamicBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for GAMIC data."""

    available = has_import(h5netcdf)

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
        group="scan0",
        invalid_netcdf=None,
        phony_dims="access",
        decode_vlen_strings=True,
        keep_elevation=False,
        keep_azimuth=False,
        reindex_angle=None,
    ):

        if isinstance(filename_or_obj, io.IOBase):
            filename_or_obj.seek(0)

        store = GamicStore.open(
            filename_or_obj,
            format=format,
            group=group,
            invalid_netcdf=invalid_netcdf,
            phony_dims=phony_dims,
            decode_vlen_strings=decode_vlen_strings,
        )

        store_entrypoint = StoreBackendEntrypoint()

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

        ds.encoding["engine"] = "gamic"

        ds = ds.sortby(list(ds.dims.keys())[0])

        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(_reindex_angle, store=store, tol=reindex_angle)

        if not keep_azimuth:
            if ds.azimuth.dims[0] == "elevation":
                ds = ds.assign_coords({"azimuth": ds.azimuth.pipe(_fix_angle)})
        if not keep_elevation:
            if ds.elevation.dims[0] == "azimuth":
                ds = ds.assign_coords({"elevation": ds.elevation.pipe(_fix_angle)})

        return ds


class CfRadial1BackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for CfRadial1 data."""

    available = has_import(netCDF4)

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
        group="/",
    ):

        store = NetCDF4DataStore.open(
            filename_or_obj,
            format=format,
            group=None,
        )

        store_entrypoint = StoreBackendEntrypoint()

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

        if group != "/":
            ds = _assign_data_radial(ds, sweep=group)[0]

        return ds


class CfRadial2BackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for CfRadial2 data."""

    available = has_import(netCDF4)

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

        # 1. first open store with group=None
        # to get the root group and select wanted sweep/group
        # 2. open store with wanted sweep/group and merge with root

        if isinstance(filename_or_obj, io.IOBase):
            filename_or_obj.seek(0)

        store = NetCDF4DataStore.open(
            filename_or_obj,
            format=format,
            group=None,
            lock=False,
        )

        if group is not None:
            variables = store.get_variables()
            var = Dataset(variables)
            site = {
                key: loc
                for key, loc in var.items()
                if key in ["longitude", "latitude", "altitude"]
            }
            sweep_names = var.sweep_group_name.values
            idx = np.where(sweep_names == group)
            fixed_angle = var.sweep_fixed_angle.values[idx].item()

            store.close()

            if isinstance(filename_or_obj, io.IOBase):
                filename_or_obj.seek(0)

            store = NetCDF4DataStore.open(
                filename_or_obj,
                format=format,
                group=group,
                lock=False,
            )

        store_entrypoint = StoreBackendEntrypoint()

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
            ds = ds.assign_coords(site)
            ds.attrs["fixed_angle"] = fixed_angle
            ds = _assign_data_radial2(ds)
            dim0 = list(set(ds.dims) & {"azimuth", "elevation"})[0]
            ds = ds.sortby(dim0)

        return ds


class IrisArrayWrapper(BackendArray):
    """Wraps array of Iris RAW data."""

    def __init__(self, datastore, name, var):
        self.datastore = datastore
        self.group = var["sweep_number"]
        self.name = name
        # get rays and bins
        nrays = var["number_rays_file_written"]
        nbins = datastore.root.product_hdr["product_end"]["number_bins"]
        # todo: retrieve datatype from io.iris.SIGMET_DATA_TYPES
        # hint: source data for RAW files is int16
        # for now: assume floating point for all moments
        self.dtype = np.dtype("float32")
        # and for undecoded moments use int16
        prod = [v for v in datastore.root.data_types_dict if v["name"] == name]
        if prod and prod[0]["func"] is None:
            self.dtype = np.dtype("int16")
        if name == "DB_XHDR":
            self.dtype = np.dtype("O")
        if name in ["azimuth", "elevation"]:
            self.shape = (nrays,)
        elif name == "dtime":
            self.shape = (nrays,)
            self.dtype = np.dtype("uint16")
        elif name == "dtime_ms":
            self.shape = (nrays,)
            self.dtype = np.dtype("int32")
        else:
            self.shape = (nrays, nbins)

    def _getitem(self, key):
        # read the data and put it into dict
        self.datastore.root.get_moment(self.group, self.name)
        return self.datastore.ds["sweep_data"][self.name][key]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._getitem,
        )


class IrisStore(AbstractDataStore):
    """Store for reading IRIS sweeps via wradlib."""

    def __init__(self, manager, group=None):

        self._manager = manager
        self._group = group
        self._filename = self.filename
        self._need_time_recalc = False

    @classmethod
    def open(cls, filename, mode="r", group=None, **kwargs):
        manager = CachingFileManager(IrisRawFile, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return root

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            ds = root.data[self._group]
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        dim = self.root.first_dimension

        data = indexing.LazilyOuterIndexedArray(IrisArrayWrapper(self, name, var))
        encoding = {"group": self._group, "source": self._filename}

        mname = iris_mapping.get(name, name)
        mapping = moments_mapping.get(mname, {})
        attrs = {key: mapping[key] for key in moment_attrs if key in mapping}
        attrs[
            "coordinates"
        ] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"
        return mname, Variable((dim, "range"), data, attrs, encoding)

    def open_store_coordinates(self, var):
        azimuth = indexing.LazilyOuterIndexedArray(
            IrisArrayWrapper(self, "azimuth", var)
        )
        elevation = indexing.LazilyOuterIndexedArray(
            IrisArrayWrapper(self, "elevation", var)
        )

        # handle DB_XHDR time
        dtime = "dtime"
        time_prefix = ""
        if "DB_XHDR" in self.ds["ingest_data_hdrs"]:
            dtime = "dtime_ms"
            time_prefix = "milli"

        rtime = indexing.LazilyOuterIndexedArray(IrisArrayWrapper(self, dtime, var))
        time = (
            var["sweep_start_time"] - dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        ).total_seconds()
        encoding = {"group": self._group}
        rtime_attrs = {
            "units": f"{time_prefix}seconds since {var['sweep_start_time'].replace(tzinfo=None).isoformat()}Z",
            "standard_name": "time",
        }
        dim = self.root.first_dimension

        # get coordinates from IrisFile
        sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"
        lon_attrs = {
            "long_name": "longitude",
            "units": "degrees_east",
            "standard_name": "longitude",
        }
        lat_attrs = {
            "long_name": "latitude",
            "units": "degrees_north",
            "positive": "up",
            "standard_name": "latitude",
        }
        alt_attrs = {
            "long_name": "altitude",
            "units": "meters",
            "standard_name": "altitude",
        }
        lon, lat, alt = self.root.site_coords

        task = self.root.ingest_header["task_configuration"]["task_range_info"]
        range_first_bin = task["range_first_bin"]
        range_last_bin = task["range_last_bin"]
        if range_first_bin == 0:
            range_first_bin = task["step_output_bins"] / 2
            range_last_bin += task["step_output_bins"]
        range = (
            np.arange(
                range_first_bin,
                range_last_bin,
                task["step_output_bins"],
                dtype="float32",
            )[: task["number_output_bins"]]
            / 1e2
        )
        range_attrs["meters_to_center_of_first_gate"] = range_first_bin
        range_attrs["meters_between_gates"] = task["step_output_bins"]

        rtime = Variable((dim,), rtime, rtime_attrs, encoding)

        coords = {
            "azimuth": Variable((dim,), azimuth, az_attrs_template.copy(), encoding),
            "elevation": Variable(
                (dim,), elevation, el_attrs_template.copy(), encoding
            ),
            "rtime": rtime,
            "time": Variable((), time, time_attrs, encoding),
            "range": Variable(("range",), range, range_attrs),
            "longitude": Variable((), lon, lon_attrs),
            "latitude": Variable((), lat, lat_attrs),
            "altitude": Variable((), alt, alt_attrs),
            "sweep_mode": Variable((), sweep_mode),
        }

        # a1gate
        a1gate = np.where(rtime.values == rtime.values.min())[0][0]
        coords[dim].attrs["a1gate"] = a1gate
        # angle_res
        task_scan_info = self.root.ingest_header["task_configuration"]["task_scan_info"]
        coords[dim].attrs["angle_res"] = (
            task_scan_info["desired_angular_resolution"] / 1000.0
        )

        return coords

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **dict(
                    self.open_store_variable(k, v)
                    for k, v in self.ds["ingest_data_hdrs"].items()
                ),
                **self.open_store_coordinates(
                    list(self.ds["ingest_data_hdrs"].values())[0]
                ),
            }.items()
        )

    def get_attrs(self):
        ing_head = self.ds["ingest_data_hdrs"]
        data = ing_head[list(ing_head.keys())[0]]

        attributes = {"fixed_angle": np.round(data["fixed_angle"], 1)}
        # RHI limits
        if self.root.scan_mode == 2:
            tsi = self.root.ingest_header["task_configuration"]["task_scan_info"][
                "task_type_scan_info"
            ]
            ll = tsi["lower_elevation_limit"]
            ul = tsi["upper_elevation_limit"]
            attributes.update(
                {"elevation_lower_limit": ll, "elevation_upper_limit": ul}
            )
        return FrozenDict(attributes)


class IrisBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for IRIS/Sigmet data."""

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
        group=None,
        keep_elevation=False,
        keep_azimuth=False,
        reindex_angle=None,
    ):
        store = IrisStore.open(
            filename_or_obj,
            group=group,
            loaddata=False,
        )

        store_entrypoint = StoreBackendEntrypoint()

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

        if decode_coords and reindex_angle is not False:
            ds = ds.drop_vars("DB_XHDR", errors="ignore")
            ds = ds.pipe(_reindex_angle, store=store, tol=reindex_angle)
        else:
            ds = ds.sortby(store.root.first_dimension)

        ds.attrs.pop("elevation_lower_limit", None)
        ds.attrs.pop("elevation_upper_limit", None)

        return ds


class RainbowArrayWrapper(BackendArray):
    """Wraps array of RAINBOW5 data."""

    def __init__(self, datastore, name, var):
        self.datastore = datastore
        self.name = name

        # get rays and bins
        nrays = int(var.get("@rays", False))
        nbins = int(var.get("@bins", False))
        dtype = np.dtype(f"uint{var.get('@depth')}")
        self.dtype = dtype
        if nbins:
            self.shape = (nrays, nbins)
        else:
            self.shape = (nrays,)
        self.blobid = int(var["@blobid"])

    def _getitem(self, key):
        # read the data and put it into dict
        self.datastore.root.get_blob(self.blobid, self.datastore._group)
        if isinstance(self.name, int):
            return self.datastore.ds["slicedata"]["rayinfo"][self.name]["data"][key]
        else:
            return self.datastore.ds["slicedata"]["rawdata"]["data"][key]

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._getitem,
        )


class RainbowStore(AbstractDataStore):
    """Store for reading RAINBOW5 sweeps via wradlib."""

    def __init__(self, manager, group=None):

        self._manager = manager
        self._group = group
        self._filename = self.filename
        self._need_time_recalc = False

    @classmethod
    def open(cls, filename, mode="r", group=None, **kwargs):
        manager = CachingFileManager(RainbowFile, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return root

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            try:
                ds = root.header["scan"]["slice"][self._group]
            except KeyError:
                ds = root.header["scan"]["slice"]
        return ds

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, var):
        dim = self.root.first_dimension
        raw = var["slicedata"]["rawdata"]
        name = raw["@type"]

        data = indexing.LazilyOuterIndexedArray(RainbowArrayWrapper(self, name, raw))
        encoding = {"group": self._group, "source": self._filename}

        vmin = float(raw.get("@min"))
        vmax = float(raw.get("@max"))
        depth = int(raw.get("@depth"))
        scale_factor = (vmax - vmin) / (2**depth - 2)
        mname = rainbow_mapping.get(name, name)
        mapping = moments_mapping.get(mname, {})
        attrs = {key: mapping[key] for key in moment_attrs if key in mapping}
        attrs["add_offset"] = vmin - scale_factor
        attrs["scale_factor"] = scale_factor
        attrs["_FillValue"] = 0
        attrs[
            "coordinates"
        ] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"
        return {mname: Variable((dim, "range"), data, attrs, encoding)}

    def open_store_coordinates(self, var):

        dim = self.root.first_dimension
        ray = var["slicedata"]["rayinfo"]

        if not isinstance(ray, list):
            var["slicedata"]["rayinfo"] = [ray]
            ray = var["slicedata"]["rayinfo"]

        start = next(filter(lambda x: x["@refid"] == "startangle", ray), False)
        start_idx = ray.index(start)
        stop = next(filter(lambda x: x["@refid"] == "stopangle", ray), False)

        anglestep = self.root._get_rbdict_value(var, "anglestep", dtype=float)
        antdirection = self.root._get_rbdict_value(
            var, "antdirection", default=0, dtype=bool
        )

        encoding = {"group": self._group}
        startangle = indexing.LazilyOuterIndexedArray(
            RainbowArrayWrapper(self, start_idx, start)
        )

        step = anglestep
        # antdirection == True ->> negative angles
        # antdirection == False ->> positive angles
        if antdirection:
            step = -anglestep

        if dim == "azimuth":
            startaz = Variable((dim,), startangle, az_attrs_template.copy(), encoding)

            if stop:
                stop_idx = ray.index(stop)
                stopangle = indexing.LazilyOuterIndexedArray(
                    RainbowArrayWrapper(self, stop_idx, stop)
                )
                stopaz = Variable((dim,), stopangle, az_attrs_template.copy(), encoding)
                zero_index = np.where(startaz - stopaz > 5)
                stopazv = stopaz.values
                stopazv[zero_index[0]] += 360
                azimuth = (startaz + stopazv) / 2.0
                azimuth[azimuth >= 360] -= 360
            else:
                azimuth = startaz + step / 2.0

            elevation = np.ones_like(azimuth) * float(var["posangle"])
        else:
            startel = Variable((dim,), startangle, el_attrs_template.copy(), encoding)

            if stop:
                stop_idx = ray.index(stop)
                stopangle = indexing.LazilyOuterIndexedArray(
                    RainbowArrayWrapper(self, stop_idx, stop)
                )
                stopel = Variable((dim,), stopangle, el_attrs_template.copy(), encoding)
                elevation = (startel + stopel) / 2.0
            else:
                elevation = startel + step / 2.0

            azimuth = np.ones_like(elevation) * float(var["posangle"])

        dstr = var["slicedata"]["@date"]
        tstr = var["slicedata"]["@time"]

        timestr = f"{dstr}T{tstr}Z"
        time = dt.datetime.strptime(timestr, "%Y-%m-%dT%H:%M:%SZ")
        total_seconds = (time - dt.datetime(1970, 1, 1)).total_seconds()

        # range is in km
        start_range = self.root._get_rbdict_value(
            var, "startrange", default=0, dtype=float
        )
        start_range *= 1000.0

        stop_range = self.root._get_rbdict_value(var, "stoprange", dtype=float)
        stop_range *= 1000.0

        range_step = self.root._get_rbdict_value(var, "rangestep", dtype=float)
        range_step *= 1000.0
        rng = np.arange(
            start_range + range_step / 2,
            stop_range + range_step / 2,
            range_step,
            dtype="float32",
        )[: int(var["slicedata"]["rawdata"]["@bins"])]

        range_attrs["meters_to_center_of_first_gate"] = start_range + range_step / 2
        range_attrs["meters_between_gates"] = range_step

        # making-up ray times
        antspeed = self.root._get_rbdict_value(var, "antspeed", dtype=float)
        raytime = anglestep / antspeed
        raytimes = np.array(
            [
                dt.timedelta(seconds=x * raytime).total_seconds()
                for x in range(azimuth.shape[0] + 1)
            ]
        )

        diff = np.diff(raytimes) / 2.0
        rtime = raytimes[:-1] + diff
        rtime_attrs = {
            "units": f"seconds since {time.isoformat()}Z",
            "standard_name": "time",
        }

        rng = Variable(("range",), rng, range_attrs)
        azimuth = Variable((dim,), azimuth, az_attrs_template.copy(), encoding)
        elevation = Variable((dim,), elevation, el_attrs_template.copy(), encoding)
        rtime = Variable((dim,), rtime, rtime_attrs, encoding)
        time = Variable((), total_seconds, time_attrs, encoding)

        # get coordinates from RainbowFile
        sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"
        lon_attrs = {
            "long_name": "longitude",
            "units": "degrees_east",
            "standard_name": "longitude",
        }
        lat_attrs = {
            "long_name": "latitude",
            "units": "degrees_north",
            "positive": "up",
            "standard_name": "latitude",
        }
        alt_attrs = {
            "long_name": "altitude",
            "units": "meters",
            "standard_name": "altitude",
        }
        lon, lat, alt = self.root.site_coords

        coords = {
            "azimuth": azimuth,
            "elevation": elevation,
            "range": rng,
            "time": time,
            "rtime": rtime,
            "longitude": Variable((), lon, lon_attrs),
            "latitude": Variable((), lat, lat_attrs),
            "altitude": Variable((), alt, alt_attrs),
            "sweep_mode": Variable((), sweep_mode),
        }

        # a1gate, this might be off by 1 if reindexing is applied
        if dim == "azimuth":
            a1gate = np.argmin(azimuth[::-1].values)
        else:
            a1gate = np.argmin(elevation[::-1].values)
        coords[dim].attrs["a1gate"] = a1gate
        # angle_res
        coords[dim].attrs["angle_res"] = anglestep
        return coords

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **self.open_store_variable(self.ds),
                **self.open_store_coordinates(self.ds),
            }.items()
        )

    def get_attrs(self):
        attributes = {"fixed_angle": float(self.ds["posangle"])}
        return FrozenDict(attributes)


class RainbowBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for Rainbow5 data."""

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
        group=None,
        reindex_angle=None,
    ):
        store = RainbowStore.open(
            filename_or_obj,
            group=group,
            loaddata=False,
        )

        store_entrypoint = StoreBackendEntrypoint()

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

        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(_reindex_angle, store=store, tol=reindex_angle)

        return ds


class FurunoArrayWrapper(BackendArray):
    def __init__(
        self,
        data,
    ):
        self.data = data
        self.shape = data.shape
        self.dtype = np.dtype("uint16")

    def __getitem__(self, key: tuple):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER_1VECTOR,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        return self.data[key]


class FurunoStore(AbstractDataStore):
    """Store for reading Furuno sweeps via wradlib."""

    def __init__(self, manager, group=None):

        self._manager = manager
        self._group = group
        self._filename = self.filename
        self._need_time_recalc = False

    @classmethod
    def open(cls, filename, mode="r", group=None, **kwargs):
        manager = CachingFileManager(FurunoFile, filename, mode=mode, kwargs=kwargs)
        return cls(manager, group=group)

    @property
    def filename(self):
        with self._manager.acquire_context(False) as root:
            return root.filename

    @property
    def root(self):
        with self._manager.acquire_context(False) as root:
            return root

    def _acquire(self, needs_lock=True):
        with self._manager.acquire_context(needs_lock) as root:
            return root

    @property
    def ds(self):
        return self._acquire()

    def open_store_variable(self, name, var):
        dim = self.root.first_dimension

        data = indexing.LazilyOuterIndexedArray(FurunoArrayWrapper(var))
        encoding = {"group": self._group, "source": self._filename}
        if name == "PHIDP":
            add_offset = 360 * -32768 / 65535
            scale_factor = 360 / 65535
        elif name == "RHOHV":
            add_offset = 2 * -1 / 65534
            scale_factor = 2 / 65534
        elif name == "WRADH":
            add_offset = -1e-2
            scale_factor = 1e-2
        elif name in ["azimuth", "elevation"]:
            add_offset = 0
            scale_factor = 1e-2
        else:
            add_offset = -327.68
            scale_factor = 1e-2

        mapping = moments_mapping.get(name, {})
        attrs = {key: mapping[key] for key in moment_attrs if key in mapping}
        if name in ["azimuth", "elevation"]:
            attrs = (
                az_attrs_template.copy()
                if name == "azimuth"
                else el_attrs_template.copy()
            )
            attrs["add_offset"] = add_offset
            attrs["scale_factor"] = scale_factor
            dims = (dim,)
            if name == self.ds.first_dimension:
                attrs["a1gate"] = self.ds.a1gate
                attrs["angle_res"] = self.ds.angle_resolution
        else:
            if name != "QUAL":
                attrs["add_offset"] = add_offset
                attrs["scale_factor"] = scale_factor
                attrs["_FillValue"] = 0
            dims = (dim, "range")
        attrs[
            "coordinates"
        ] = "elevation azimuth range latitude longitude altitude time rtime sweep_mode"
        return Variable(dims, data, attrs, encoding)

    def open_store_coordinates(self):

        dim = self.ds.first_dimension

        # range
        start_range = 0
        if self.ds.version == 3:
            range_step = self.ds.header["resolution_range_direction"] / 100
        else:
            range_step = self.ds.header["resolution_range_direction"]
        stop_range = range_step * self.ds.header["number_range_direction_data"]
        rng = np.arange(
            start_range + range_step / 2,
            stop_range + range_step / 2,
            range_step,
            dtype="float32",
        )

        range_attrs["meters_to_center_of_first_gate"] = start_range + range_step / 2
        range_attrs["meters_between_gates"] = range_step
        rng = Variable(("range",), rng, range_attrs)

        # making-up ray times
        time = self.ds.header["scan_start_time"]
        stop_time = self.ds.header.get("scan_stop_time", time)
        num_rays = self.ds.header["number_sweep_direction_data"]
        total_seconds = (time - dt.datetime(1970, 1, 1)).total_seconds()

        # if no stop_time is available, get time from rotation speed
        if time == stop_time:
            raytime = self.ds.angle_resolution / (
                self.ds.header["antenna_rotation_speed"] * 1e-1 * 6
            )
            raytime = dt.timedelta(seconds=raytime)
        # otherwise, calculate from time difference
        else:
            raytime = (stop_time - time) / num_rays

        raytimes = np.array(
            [(x * raytime).total_seconds() for x in range(num_rays + 1)]
        )

        diff = np.diff(raytimes) / 2.0
        rtime = raytimes[:-1] + diff

        rtime_attrs = {
            "units": f"seconds since {time.isoformat()}Z",
            "standard_name": "time",
        }

        encoding = {}
        rng = Variable(("range",), rng, range_attrs)
        rtime = Variable((dim,), rtime, rtime_attrs, encoding)
        time = Variable((), total_seconds, time_attrs, encoding)

        # get coordinates from Furuno File
        sweep_mode = "azimuth_surveillance" if dim == "azimuth" else "rhi"
        lon_attrs = {
            "long_name": "longitude",
            "units": "degrees_east",
            "standard_name": "longitude",
        }
        lat_attrs = {
            "long_name": "latitude",
            "units": "degrees_north",
            "positive": "up",
            "standard_name": "latitude",
        }
        alt_attrs = {
            "long_name": "altitude",
            "units": "meters",
            "standard_name": "altitude",
        }
        lon, lat, alt = self.ds.site_coords

        coords = {
            "range": rng,
            "time": time,
            "rtime": rtime,
            "longitude": Variable((), lon, lon_attrs),
            "latitude": Variable((), lat, lat_attrs),
            "altitude": Variable((), alt, alt_attrs),
            "sweep_mode": Variable((), sweep_mode),
        }

        return coords

    def get_variables(self):
        return FrozenDict(
            (k1, v1)
            for k1, v1 in {
                **dict(
                    (k, self.open_store_variable(k, v)) for k, v in self.ds.data.items()
                ),
                **self.open_store_coordinates(),
            }.items()
        )

    def get_attrs(self):
        attributes = {"fixed_angle": float(self.ds.fixed_angle)}
        return FrozenDict(attributes)


class FurunoBackendEntrypoint(BackendEntrypoint):
    """Xarray BackendEntrypoint for Furuno data."""

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
        group=None,
        reindex_angle=None,
        obsmode=None,
    ):
        store = FurunoStore.open(
            filename_or_obj,
            group=group,
            loaddata=True,
            obsmode=obsmode,
        )

        store_entrypoint = StoreBackendEntrypoint()

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

        ds.encoding["engine"] = "furuno"

        ds = ds.sortby(list(ds.dims.keys())[0])

        if decode_coords and reindex_angle is not False:
            ds = ds.pipe(_reindex_angle, store=store, tol=reindex_angle)

        return ds
