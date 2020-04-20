#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
wradlib Xarray Accessors
^^^^^^^^^^^^^^^^^^^^^^^^

Module xarray takes care of accessing wradlib functionality from
xarray DataArrays and Datasets

.. currentmodule:: wradlib.xarray

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = ["VisMethods"]
__doc__ = __doc__.format("\n   ".join(__all__))

import re

import xarray as xr

from wradlib import vis


class XarrayMethods:
    def __init__(self, xarray_obj, module):
        self._obj = xarray_obj
        namespace = vars(module)
        module.__name__.split(".")[-1]
        (name for name in namespace if name[:1] != "_")
        for name in getattr(module, "__xr__"):
            func = namespace[name]
            if "xr_" in name:
                name = name[3:]
            setattr(self, name, func.__get__(self._obj, self.__class__))


class VisMethods(XarrayMethods):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj, vis)


@xr.register_dataarray_accessor("wrl")
class WradlibDataArrayAccessor:
    """DataArray Accessor for wradlib module functions"""

    __slots__ = ["_obj", "_vis"]

    def __init__(self, xarray_obj):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self._obj = xarray_obj

    def __getattr__(self, attr):
        return getattr(self._obj, attr)

    def __repr__(self):
        return re.sub(r"<.+>", f"<{self.__class__.__name__}>", str(self._obj))

    @property
    def vis(self):
        if self._vis is None:
            self._vis = VisMethods(self._obj)
        return self._vis


if __name__ == "__main__":
    print("wradlib: Calling module <xarray> as main...")
