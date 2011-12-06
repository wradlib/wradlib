# -*- coding: UTF-8 -*-
#-------------------------------------------------------------------------------
# Name:        georef_example
# Purpose:
#
# Author:      Maik Heistermann
#
# Created:     28.10.2011
# Copyright:   (c) Maik Heistermann 2011
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python


import wradlib.georef as georef


if __name__ == '__main__':
    import numpy as np
    import pylab as pl

    # --------------------------------------------------------------------------
    # EXAMPLE 1: Full workflow for georeferencing radar data

    # 1st step: generate the centroid coordinates of the radar bins
    r  = np.array([0.,   0., 111., 111., 111., 111.,])
    az = np.array([0., 180.,   0.,  90., 180., 270.,])
    csite = (48.0, 9.0)
    lat1, lon1= georef.pol2latlon(r, az, csite)

    # 2nd step: project the centroid coordinates to Gaus-Krueger Zone 3
    gk3 = '''
    +proj=tmerc +lat_0=0 +lon_0=9 +k=1 +x_0=3500000 +y_0=0 +ellps=bessel
    +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs
    '''
    x, y = georef.project(latc, lonc, projstr)

    # 3rd step: generate polygon vertices
    coords = np.arange(4).reshape((2,2))
    coords = np.transpose(np.vstack((np.arange(5), np.repeat(1,5))))
    coords = np.transpose(np.vstack((np.arange(5), np.arange(5))))
    delta = 0.5
    vertices = georef.centroid2polyvert(coords, delta)

    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(coords[:,0], coords[:,1], 'o')
    for vertice in vertices:
        ax.plot(vertice[:,0], vertice[:,1], 'r+')
    ax.set_xlim(left=-1, right=8)
    ax.set_ylim(bottom=-1, top=8)
    pl.show()

