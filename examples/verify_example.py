# -*- coding: UTF-8 -*-
# -------------------------------------------------------------------------------
# Name:        verify_example
# Purpose:
#
# Author:      Maik Heistermann
#
# Created:     28.10.2011
# Copyright:   (c) Maik Heistermann 2011
# Licence:     The MIT License
# -------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib
import numpy as np
import matplotlib.pyplot as pl
#pl.interactive(True)


def ex_verify():
    kdp_true = np.sin(3 * np.arange(0, 10, 0.1))
    phidp_true = np.cumsum(kdp_true)
    phidp_raw = phidp_true + np.random.uniform(-1, 1, len(phidp_true))
    gaps = np.concatenate([range(10, 20), range(30, 40), range(60, 80)])
    phidp_raw[gaps] = np.nan
    kdp_re = wradlib.dp.kdp_from_phidp_linregress(phidp_raw)
    line1 = pl.plot(np.ma.masked_invalid(phidp_true), "b--", label="phidp_true")
    line2 = pl.plot(np.ma.masked_invalid(phidp_raw), "b-", label="phidp_raw")
    line3 = pl.plot(kdp_true, "g-", label="kdp_true")
    line4 = pl.plot(np.ma.masked_invalid(kdp_re), "r-", label="kdp_reconstructed")
    lgnd = pl.legend(("phidp_true", "phidp_raw", "kdp_true", "kdp_reconstructed"))
    print('Done')
    pl.show()
    # import numpy as np
    # import pylab as pl
    # # just making sure that the plots immediately pop up
    # #pl.interactive(True)
    # import matplotlib as mpl
    #
    # # ------------------------------------------------------------------------------
    # # EXAMPLE 1: Extract bin values of a polar radar data set at rain gage locations
    #
    # # define the polar coordinates and the site coordinates in lat/lon
    # r = np.arange(1,129)
    # az = np.linspace(0,360,361)[0:-1]
    # sitecoords = (9.7839, 48.5861)
    # # import the polar example radar dataset
    # testdata = np.loadtxt(os.path.dirname(__file__) + '/' + 'data/polar_R_tur.gz')
    # # the rain gages are in Gauss-Krueger Zone 3 coordinates, so we need the
    # #   corresponding epsg-code
    # proj = georef.epsg_to_osr(31467)
    # # these are the coordinates of the rain gages in Gauss-Krueger 3 coordinates
    # x, y = np.array([3522175, 3453575, 3456725, 3498590]), np.array([5410600, 5433600, 5437860, 5359710])
    # # <nnear> nearest radar bin shall be extracted
    # nnear = 9
    # # create an instance of PolarNeighbours
    # polarneighbs = verify.PolarNeighbours(r, az, sitecoords, proj, x, y, nnear)
    # radar_at_gages = polarneighbs.extract(testdata)
    # print( radar_at_gages)
    #
    # binx, biny = polarneighbs.get_bincoords()
    # binx_nn, biny_nn = polarneighbs.get_bincoords_at_points()
    #
    # # plot the centroids of all radar bins (red plus signs), the rain gages (blue circles)
    # #   and the nnear neighbouring radar bins (blue plus signs)
    # fig = pl.figure(figsize=(12,12))
    # ax = fig.add_subplot(111)
    # ax.plot(binx, biny, 'r+')
    # ax.plot(binx_nn, biny_nn, 'b+')
    # ax.plot(x, y, 'bo')
    # ax.axis('tight')
    # pl.title('USE THE ZOOM TOOL TO INSPECT THE NEIGHBOURHOOD OF THE GAGES!')
    # pl.show()

    print('Exit.')


if __name__ == '__main__':
    ex_verify()
