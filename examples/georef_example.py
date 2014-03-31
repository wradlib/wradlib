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
    import matplotlib as mpl

    # --------------------------------------------------------------------------
    # EXAMPLE 1: Full workflow for georeferencing radar data

    # 1st step: generate the centroid coordinates of the radar bins
    #   define the polar coordinates and the site coordinates in lat/lon
    r = np.arange(1,129)*1000
    az = np.linspace(0,360,361)[0:-1]
    #   drs:  51.12527778 ; fbg: 47.87444444 ; tur: 48.58611111 ; muc: 48.3372222
    #   drs:  13.76972222 ; fbg: 8.005 ; tur: 9.783888889 ; muc: 11.61277778
    sitecoords = (48.5861, 9.7839)

    #   these are the polgon vertices of the radar bins
    polygons = georef.polar2polyvert(r, az, sitecoords)

    #   these are the corresponding centroids
    cent_lon, cent_lat = georef.polar2centroids(r, az, sitecoords)

    # plot the vertices and the centroids in one plot
    fig = pl.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    polycoll = mpl.collections.PolyCollection(polygons,closed=True, facecolors='None')
    ax.add_collection(polycoll, autolim=True)
    ax.plot(cent_lon, cent_lat, 'r+')
    ax.axis('tight')
    pl.title('Zoom in to compare polygons and centroids.')
    pl.show()


    # 2nd step: project the centroid coordinates to Gauss-Krueger Zone 3
    #   this is the proj.4 projection string
    gk3 = '''
    +proj=tmerc +lat_0=0 +lon_0=9 +k=1 +x_0=3500000 +y_0=0 +ellps=bessel
    +towgs84=598.1,73.7,418.2,0.202,0.045,-2.455,6.7 +units=m +no_defs
    '''
    # use it for projecting the centroids to Gauss-Krueger 3
    x, y = georef.project(cent_lat, cent_lon, gk3)

    # export the projected centroid coordinates
    f = open('centroids.tab', 'w')
    f.write('x\ty\n')
    np.savetxt(f, np.hstack( (x.reshape((-1,1)),y.reshape((-1,1))) ), fmt='%.2f', delimiter='\t')
    f.close()

    print 'Exit.'