#!/usr/bin/env python
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import wradlib
import os


# In this example, we test reading NetCDF radar files from different sources
def read_and_overview(filename):
    """Read NetCDF using read_generic_netcdf and print upper level dictionary keys
    """
    test = wradlib.io.read_generic_netcdf(filename)
    print("\nPrint keys for file %s" % os.path.basename(filename))
    for key in test.keys():
        print(key)


def ex_read_generic_netcdf():
    # CfRadial example from TITAN homepage
    #   http://www.ral.ucar.edu/projects/titan/docs/radial_formats
    filename = 'netcdf/cfrad.20080604_002217_000_SPOL_v36_SUR.nc'
    filename = wradlib.util.get_wradlib_data_file(filename)
    read_and_overview(filename)

    # Example PPI from Py-ART repository
    # https://github.com/ARM-DOE/pyart/
    filename = 'netcdf/example_cfradial_ppi.nc'
    filename = wradlib.util.get_wradlib_data_file(filename)
    read_and_overview(filename)
    # Example RHI from Py-ART repository
    # https://github.com/ARM-DOE/pyart/
    filename = 'netcdf/example_cfradial_rhi.nc'
    filename = wradlib.util.get_wradlib_data_file(filename)
    read_and_overview(filename)

    # Example EDGE NetCDF export format
    filename = 'netcdf/edge_netcdf.nc'
    filename = wradlib.util.get_wradlib_data_file(filename)
    read_and_overview(filename)


# =======================================================
if __name__ == '__main__':
    ex_read_generic_netcdf()
