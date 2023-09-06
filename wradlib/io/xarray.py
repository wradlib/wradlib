#!/usr/bin/env python
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Xarray based Data I/O
^^^^^^^^^^^^^^^^^^^^^

Note
----
    The Xarray backend code has been moved to `xradar <https://xradar.rtfd.io>`__-package.

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""
__all__ = [
    "WradlibVariable",
]
__doc__ = __doc__.format("\n   ".join(__all__))


class WradlibVariable:
    """Minimal variable wrapper."""

    def __init__(self, dims, data, attrs):
        self._dimensions = dims
        self._data = data
        self._attrs = attrs

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def data(self):
        return self._data

    @property
    def attributes(self):
        return self._attrs
