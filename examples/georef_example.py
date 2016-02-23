# -*- coding: UTF-8 -*-
# -------------------------------------------------------------------------------
# Name:        georef_example
# Purpose:
#
# Author:      Maik Heistermann
#
# Created:     28.10.2011
# Copyright:   (c) Maik Heistermann 2011
# Licence:     The MIT License
# -------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib.georef as georef
import numpy as np
import matplotlib.pyplot as pl
#pl.interactive(True)
import matplotlib as mpl


def ex_georef():
    # --------------------------------------------------------------------------
    # EXAMPLE 1: Full workflow for georeferencing radar data

    # 1st step: generate the centroid coordinates of the radar bins
    #   define the polar coordinates and the site coordinates in lat/lon
    r = np.arange(1, 129) * 1000
    az = np.linspace(0, 360, 361)[0:-1]
    #   drs:  51.12527778 ; fbg: 47.87444444 ; tur: 48.58611111 ; muc: 48.3372222
    #   drs:  13.76972222 ; fbg: 8.005 ; tur: 9.783888889 ; muc: 11.61277778
    sitecoords = (9.7839, 48.5861)

    #   these are the polgon vertices of the radar bins
    polygons = georef.polar2polyvert(r, az, sitecoords)

    #   these are the corresponding centroids
    cent_lon, cent_lat = georef.polar2centroids(r, az, sitecoords)

    # plot the vertices and the centroids in one plot
    fig = pl.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    polycoll = mpl.collections.PolyCollection(polygons, closed=True, facecolors='None')
    ax.add_collection(polycoll, autolim=True)
    ax.plot(cent_lon, cent_lat, 'r+')
    ax.axis('tight')
    pl.title('Zoom in to compare polygons and centroids.')
    pl.show()

    # 2nd step: reproject the centroid coordinates to Gauss-Krueger Zone 3
    #   by using the EPSG-Number 31467
    # use it for projecting the centroids to Gauss-Krueger 3
    proj_gk3 = georef.epsg_to_osr(31467)
    x, y = georef.reproject(cent_lon, cent_lat, projection_targe=proj_gk3)

    # export the projected centroid coordinates
    # f = open('centroids.tab', 'w')
    f = 'centroids.tab'
    # f.write('x\ty\n')
    np.savetxt(f, np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1)))), fmt='%.2f', header='x\ty', delimiter='\t')
    # f.close()

    print('Exit.')


if __name__ == '__main__':
    ex_georef()
