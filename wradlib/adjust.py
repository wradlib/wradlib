#-------------------------------------------------------------------------------
# Name:         adjust
# Purpose:
#
# Authors:      Maik Heistermann, Stephan Jacobi and Thomas Pfaff
#
# Created:      26.10.2011
# Copyright:    (c) Maik Heistermann, Stephan Jacobi and Thomas Pfaff 2011
# Licence:      The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

"""
Adjustment
^^^^^^^^^^

Adjusting remotely sensed spatial data by ground truth (gage observations)

The main objective of this module is the adjustment of radar-based QPE
by rain gage observations. However, this module can also be applied to adjust
satellite rainfall by rain gage observations, remotely sensed soil moisture
patterns by ground truthing moisture sensors or any spatial point pattern
which ought to be adjusted by selcted point measurements.

Basically, we only need two data sources:

- point observations (e.g. rain gage observations)

- set of (potentially irregular) unadjusted point values (e.g. remotely sensed rainfall)

.. autosummary::
   :nosignatures:
   :toctree: generated/


"""

# site packages
import numpy as np

# wradlib modules
import wradlib.ipol as ipol

class AdjustBase():
    """
    The basic adjustment class
    """
    def __init__(self):
        pass

if __name__ == '__main__':
    print 'wradlib: Calling module <adjust> as main...'
