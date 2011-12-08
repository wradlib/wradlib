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
##    testdata = np.loadtxt('data/polar_dBZ_tur.gz')
##    vis.polar_plot(testdata, title='Reflectivity (dBZ)')

    # EXAMPLE 2: Plot polar data on a map
    testdata = np.loadtxt('data/polar_dBZ_tur.gz')
    r = np.arange(1,129)
    az = np.linspace(0,360,361)[0:-1]
    #   drs:  51.12527778 ; fbg: 47.87444444 ; tur: 48.58611111 ; muc: 48.3372222
    #   drs:  13.76972222 ; fbg: 8.005 ; tur: 9.783888889 ; muc: 11.61277778
    sitecoords = (48.5861, 9.7839)
    polygons = georef.polar2polyvert(r, az, sitecoords)
    baseplot = vis.PolarBasemap(polygons, sitecoords, r, az)
    baseplot(testdata)




