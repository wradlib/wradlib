#-------------------------------------------------------------------------------
# Name:        vis_example
# Purpose:
#
# Author:      heistermann
#
# Created:     26.10.2011
# Copyright:   (c) heistermann 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib.vis as vis
import wradlib.georef as georef


if __name__ == '__main__':

    import numpy as np

    # EXAMPLE 1: Simple polar plot of e.g. reflectivity
    testdata = np.loadtxt('data/polar_dBZ_tur.gz')
    classes = [0, 20, 30, 40, 50, 55, 60, 65]
    vis.polar_plot(testdata, title='Reflectivity', unit='dBZ', colormap='spectral', classes=classes, extend='max')

    # EXAMPLE 2: Plot polar data on a map for Tuerkheim
##    testdata = np.loadtxt('data/polar_dBZ_tur.gz')
##    r = np.arange(1,129)
##    az = np.linspace(0,360,361)[0:-1]
##    #   drs:  51.12527778 ; fbg: 47.87444444 ; tur: 48.58611111 ; muc: 48.3372222
##    #   drs:  13.76972222 ; fbg: 8.005 ; tur: 9.783888889 ; muc: 11.61277778
##    sitecoords = (48.5861, 9.7839)
##    polygons = georef.polar2polyvert(r, az, sitecoords)
##    baseplot = vis.PolarBasemap(polygons, sitecoords, r, az)
##    baseplot(testdata)

    # EXAMPLE 3: Trying the same for the Philippines
    import netCDF4 as nc
    #   specify file path
    fname='E:/data/philippines/radar/Netcdf 0926_2011_00H/SUB-20110926-000549-01-Z.nc'
    #   read the data from file
    testdata = nc.Dataset(fname)
    r = np.linspace(0.5, testdata.getncattr('MaximumRange-value'), len(testdata.dimensions['Gate']))
    az = np.round(np.array(testdata.variables['Azimuth']),0)
    #   drs:  51.12527778 ; fbg: 47.87444444 ; tur: 48.58611111 ; muc: 48.3372222
    #   drs:  13.76972222 ; fbg: 8.005 ; tur: 9.783888889 ; muc: 11.61277778
    sitecoords = (testdata.Latitude, testdata.Longitude)
    polygons = georef.polar2polyvert(r, az, sitecoords)
    baseplot = vis.PolarBasemap(polygons, sitecoords, r, az)
##    data = np.array(testdata.variables[testdata.TypeName])
##    data = np.where(data==testdata.MissingData, 0., data)
    data = np.ma.masked_values(testdata.variables[testdata.TypeName], testdata.MissingData)
    baseplot(data)



