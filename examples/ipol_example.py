#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      heistermann
#
# Created:     28.10.2011
# Copyright:   (c) heistermann 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

from wradlib.ipol import *

if __name__ == '__main__':

    import datetime as dt
    import pylab as pl
    import numpy as np

    xsrc = np.loadtxt('data/bin_coords_tur.gz')
#    xtrg = np.loadtxt('../examples/data/target_coords.dat')
    xtrg = np.meshgrid(np.linspace(3300000.0, 3300000.0+900000,100), np.linspace(5200000.0, 5200000.0+900000.,100))
#    xtrg = np.transpose(np.vstack((xtrg[0].ravel(), xtrg[1].ravel())))
    vals = np.loadtxt('data/polar_R_tur.gz').ravel()

    print 'Building our object takes:'
    t0 = dt.datetime.now()
    ip = Idw(xsrc, xtrg)
    print dt.datetime.now()-t0
    print 'Calling the object takes:'
    t0 = dt.datetime.now()
    test = ip(vals)
    print dt.datetime.now()-t0
    pl.scatter(xtrg[0].ravel(), xtrg[1].ravel(), c=test.ravel(), s=5, edgecolor='none')
    pl.show()
    pl.close()

    xsrc = np.arange(10)[:,None]
    xtrg = np.linspace(0,20,40)[:,None]
    vals = np.sin(xsrc).ravel()
    ip = Idw(xsrc, xtrg)
    test = ip(vals)
    pl.plot(xsrc.ravel(), vals, 'b+')
    pl.plot(xtrg.ravel(), test, 'r+')
    pl.show()
    pl.close()

    xsrc = np.vstack((np.array([4,7,3,15]), np.array([8,18,17,3]))).transpose()
    xtrg = np.meshgrid( np.linspace(0,20,40), np.linspace(0,20,40))
    vals = np.random.uniform(size=len(xsrc))
    ip = Idw(xsrc, xtrg)
    test = ip(vals)
    pl.scatter(xtrg[0], xtrg[1], c=test.ravel(), s=20, edgecolor='none')
    pl.scatter(xsrc[:,0], xsrc[:,1], c=vals.ravel(), s=50, marker='s')
    pl.colorbar()
    pl.show()
    pl.close()

