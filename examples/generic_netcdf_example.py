#-------------------------------------------------------------------------------
# Name:        generic_netcdf_example.py
# Purpose:
#
# Author:      Maik Heistermann
#
# Created:     16/01/2015
# Copyright:   (c) Maik Heistermann 2015
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib
import numpy as np
import os


# In this example, we test reading NetCDF radar files from different sources

def read_and_overview(filename):
    """Read NetCDF using read_generic_netcdf and print upper level dictionary keys
    """
    test = wradlib.io.read_generic_netcdf(filename)
    print "\nPrint keys for file %s" % os.path.basename(filename)
    for key in test.keys():
        print key

def ex_read_generic_netcdf():
    # CfRadial example from TITAN homepage
    #   http://www.ral.ucar.edu/projects/titan/docs/radial_formats
    filename = os.path.dirname(__file__) + '/' + 'data/cfrad.20080604_002217_000_SPOL_v36_SUR.nc'
    read_and_overview(filename)

    # Example PPI from Py-ART repository
    #    https://github.com/ARM-DOE/pyart/blob/master/pyart/testing/data/example_cfradial_ppi.nc
    filename = os.path.dirname(__file__) + '/' + 'data/example_cfradial_ppi.nc'
    read_and_overview(filename)
    # Example RHI from Py-ART repository
    #    https://github.com/ARM-DOE/pyart/blob/master/pyart/testing/data/example_cfradial_rhi.nc
    filename = os.path.dirname(__file__) + '/' + 'data/example_cfradial_rhi.nc'
    read_and_overview(filename)

    # Example EDGE NetCDF export format
    filename = os.path.dirname(__file__) + '/' + 'data/edge_netcdf.nc'
    read_and_overview(filename)

# =======================================================
if __name__ == '__main__':
    ex_read_generic_netcdf()
