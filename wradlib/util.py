#-------------------------------------------------------------------------------
# Name:        util
# Purpose:
#
# Author:      heistermann
#
# Created:     26.10.2011
# Copyright:   (c) heistermann 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python
"""
Utility functions
^^^^^^^^^^^^^^^^^

Module util provides a set of useful helpers which are currently not attributable
to the other modules

.. currentmodule:: wradlib.util

.. autosummary::
   :nosignatures:
   :toctree: generated/

   aggregate_in_time

"""
import numpy as np


def aggregate_in_time(src, dtimes_src, dtimes_trg):
    """Aggregate time series data to a coarser temporal resolution.

    Parameters
    ----------
    src : array
        xxx
    dtimes_src : array of datetime objects
        xxx
    dtimes_trg : array of datetime objects
        xxx

    Returns
    -------
    output : array
        xxx

    """
    return None


if __name__ == '__main__':
    print 'wradlib: Calling module <util> as main...'
