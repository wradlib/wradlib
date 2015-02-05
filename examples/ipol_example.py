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
import os

def ex_ipol():

    import datetime as dt
    import pylab as pl
    pl.interactive(True)
    import numpy as np

    xsrc = np.loadtxt(os.path.dirname(__file__) + '/' + 'data/bin_coords_tur.gz')
#    xtrg = np.loadtxt('../examples/data/target_coords.dat')
    xtrg = np.meshgrid(np.linspace(3300000.0, 3300000.0+900000,100), np.linspace(5200000.0, 5200000.0+900000.,100))
#    xtrg = np.transpose(np.vstack((xtrg[0].ravel(), xtrg[1].ravel())))
    vals = np.loadtxt(os.path.dirname(__file__) + '/' + 'data/polar_R_tur.gz').ravel()

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

    # one dimensional in space
    xsrc = np.arange(10)[:,None]
    xtrg = np.linspace(0,20,40)[:,None]
    vals = np.sin(xsrc).ravel()
    ip = Idw(xsrc, xtrg)
    test = ip(vals)
    pl.plot(xsrc.ravel(), vals, 'b+')
    pl.plot(xtrg.ravel(), test, 'r+')
    pl.show()
    pl.close()

    # two-dimensional in space
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

    # --------------------------------------------------------------------------
    # Using the convenience function ipol.interpolation in order to deal with missing values
    #    1st test: for 1 dimension in space and two dimensions of the source value array
    src = np.arange(10)[:,None]
    trg = np.linspace(0,20,40)[:,None]
    vals = np.hstack((np.sin(src), 10.+np.sin(src)))
    # here we introduce missing values only in the second dimension
    vals[3:5,1] = np.nan
    ipol_result = interpolate(src, trg, vals, Idw, nnearest=2)
    # plot if you like
    import pylab as pl
    pl.plot(trg, ipol_result, 'b+')
    pl.plot(src, vals, 'ro')
    pl.show()

    #    2nd test: for 2 dimensions in space and two dimensions of the source value array
    src = np.vstack((np.array([4,7,3,15]), np.array([8,18,17,3]))).transpose()
    trg = np.meshgrid( np.linspace(0,20,100), np.linspace(0,20,100))
    trg = np.vstack((trg[0].ravel(), trg[1].ravel())).transpose()
    vals = np.round(np.random.uniform(size=(len(src),2)),1)
    result = interpolate(src, trg, vals, Idw, nnearest=4)
    vals_with_nan = vals.copy()
    vals_with_nan[1,0] = np.nan
    vals_with_nan[1:3,1] = np.nan
    result_with_nan = interpolate(src, trg, vals_with_nan, Idw, nnearest=4)
    import pylab as pl
    vmin = np.concatenate((vals.ravel(), result.ravel())).min()
    vmax = np.concatenate((vals.ravel(), result.ravel())).max()
    def plotall(ax, trg, src, interp, pts, title):
        ix = np.where(np.isfinite(pts))
        ax.scatter(trg[: ,0],trg[: ,1], c=interp.ravel(), s=20, edgecolor='none', vmin=vmin, vmax=vmax)
        ax.scatter(src[ix,0],src[ix,1], c=pts.ravel()[ix], s=20, marker='s', vmin=vmin, vmax=vmax)
        ax.set_title(title)
    fig = pl.figure()
    ax = fig.add_subplot(221)
    plotall(ax, trg, src, result[:,0], vals[:,0], '1st dim: no NaNs')
    ax = fig.add_subplot(222)
    plotall(ax, trg, src, result[:,1], vals[:,1], '2nd dim: no NaNs')
    ax = fig.add_subplot(223)
    plotall(ax, trg, src, result_with_nan[:,0], vals_with_nan[:,0], '1st dim: one NaN')
    ax = fig.add_subplot(224)
    plotall(ax, trg, src, result_with_nan[:,1], vals_with_nan[:,1], '2nd dim: two NaN')
    pl.show()

if __name__ == '__main__':
    ex_ipol()