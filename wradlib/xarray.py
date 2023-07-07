#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
wradlib Xarray Accessors
^^^^^^^^^^^^^^^^^^^^^^^^

Since version 2.0 wradlib makes increasing use of xarray Accessors.
Module `xarray` takes care of accessing wradlib functionality from
xarray DataArrays and Datasets.

.. currentmodule:: wradlib.xarray

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}

"""
__all__ = ["WradlibXarrayAccessor"]
__doc__ = __doc__.format("\n   ".join(__all__))

import re

import xarray as xr

import wradlib


@xr.register_dataarray_accessor("wrl")
@xr.register_dataset_accessor("wrl")
class WradlibXarrayAccessor:
    """Xarray Accessor for wradlib module functions"""

    __slots__ = [
        "_obj",
        "_atten",
        "_classify",
        "_comp",
        "_dp",
        "_georef",
        "_ipol",
        "_qual",
        "_trafo",
        "_util",
        "_vis",
        "_zr",
    ]

    def __init__(self, xarray_obj):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self._obj = xarray_obj

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __repr__(self):
        return re.sub(r"<.+>", f"<{self.__class__.__name__}>", str(self._obj))

    @property
    def atten(self):
        """SubAccessor for :class:`wradlib.atten.AttenMethods`."""
        if self._atten is None:
            self._atten = wradlib.atten.AttenMethods(self._obj)
        return self._atten

    @property
    def classify(self):
        """SubAccessor for :class:`wradlib.classify.ClassifyMethods`."""
        if self._classify is None:
            self._classify = wradlib.classify.ClassifyMethods(self._obj)
        return self._classify

    @property
    def comp(self):
        """SubAccessor for :class:`wradlib.comp.CompMethods`."""
        if self._comp is None:
            self._comp = wradlib.comp.CompMethods(self._obj)
        return self._comp

    @property
    def dp(self):
        """SubAccessor for :class:`wradlib.dp.DpMethods`."""
        if self._dp is None:
            self._dp = wradlib.dp.DpMethods(self._obj)
        return self._dp

    @property
    def georef(self):
        """SubAccessor for :class:`wradlib.georef.GeorefMethods`."""
        if self._georef is None:
            self._georef = wradlib.georef.GeorefMethods(self._obj)
        return self._georef

    @property
    def ipol(self):
        """SubAccessor for :class:`wradlib.ipol.IpolMethods`."""
        if self._ipol is None:
            self._ipol = wradlib.ipol.IpolMethods(self._obj)
        return self._ipol

    @property
    def qual(self):
        """SubAccessor for :class:`wradlib.qual.QualMethods`."""
        if self._qual is None:
            self._qual = wradlib.qual.QualMethods(self._obj)
        return self._qual

    @property
    def trafo(self):
        """SubAccessor for :class:`wradlib.trafo.TrafoMethods`."""
        if self._trafo is None:
            self._trafo = wradlib.trafo.TrafoMethods(self._obj)
        return self._trafo

    @property
    def util(self):
        """SubAccessor for :class:`wradlib.util.UtilMethods`."""
        if self._util is None:
            self._util = wradlib.util.UtilMethods(self._obj)
        return self._util

    @property
    def vis(self):
        """SubAccessor for :class:`wradlib.vis.VisMethods`."""
        if self._vis is None:
            self._vis = wradlib.vis.VisMethods(self._obj)
        return self._vis

    @property
    def zr(self):
        """SubAccessor for :class:`wradlib.zr.ZRMethods`."""
        if self._vis is None:
            self._vis = wradlib.zr.ZRMethods(self._obj)
        return self._vis


if __name__ == "__main__":
    print("wradlib: Calling module <xarray> as main...")
