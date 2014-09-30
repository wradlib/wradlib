#------------------------------------------------------------------------------
# Name:        load_radolan_example.py
# Purpose:     show how to load radolan composites
#
# Author:      Kai Muehlbauer
#
# Created:     24.09.2014
# Copyright:   (c) Kai Muehlbauer 2014
# Licence:     The MIT License
#------------------------------------------------------------------------------

import wradlib as wrl
import matplotlib.pyplot as pl
import numpy as np
import matplotlib as mpl
import os


def ex_load_radolan():
    pg_filename = os.path.dirname(__file__) + '/' + 'data/raa00-pc_10015-1408030905-dwd---bin.gz'
    rw_filename = os.path.dirname(__file__) + '/' + 'data/raa01-rw_10000-1408030950-dwd---bin.gz'

    # load radolan files
    pgdata, pgattrs = wrl.io.read_RADOLAN_composite(pg_filename, missing=255)
    rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)

    # print the available attributes
    print("PG Attributes:", pgattrs)
    print("RW Attributes:", rwattrs)

    # do some masking
    pgdata = np.ma.masked_equal(pgdata, 255)
    sec = rwattrs['secondary']
    rwdata.flat[sec] = -9999
    rwdata = np.ma.masked_equal(rwdata, -9999)

    # plot the images side by side
    pl.figure(figsize=(12,8))
    pl.subplot(121, aspect='equal')
    # consider 2km grid resolution
    x = np.arange(0, pgdata.shape[0]*2 + 1, 2)
    y = np.arange(0, pgdata.shape[1]*2 + 1, 2)
    X, Y = np.meshgrid(x, y)
    # color-scheme taken from DWD "legend_radar_products_pc.pdf"
    colors = ['lightgrey', 'yellow', 'lightblue', 'magenta', 'green', 'red', 'darkblue', 'darkred']
    cmap = mpl.colors.ListedColormap(colors, name=u'DWD-pc-scheme')
    bounds = np.arange(len(colors) + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    pl.pcolormesh(X, Y, pgdata, cmap=cmap, norm=norm)
    pl.xlim(0, max(x))
    pl.ylim(0, max(y))
    # harmonize ticklabel
    pl.xticks(np.arange(min(x), max(x)+1, 100))
    pl.yticks(np.arange(min(y), max(y)+1, 100))

    # add colorbar and do some magic for proper visualisation
    cb = pl.colorbar(shrink=0.5, norm=norm, boundaries=bounds)
    loc = bounds + .5
    cb.set_ticks(loc)
    labels = bounds[:-1]
    cb.set_ticklabels(labels)
    cl = cb.ax.get_yticklabels()
    cl[-1].set_text('9')
    cb.ax.set_yticklabels([elem.get_text() for elem in cl])

    pl.title('RADOLAN PG Product \n' + pgattrs['datetime'].isoformat())

    pl.subplot(122, aspect='equal')
    x = np.arange(0, rwdata.shape[0] + 1, 1)
    y = np.arange(0, rwdata.shape[1] + 1, 1)
    X, Y = np.meshgrid(x, y)
    # using spectral to better see spatial patterns
    cmap = pl.cm.spectral
    pl.pcolormesh(X, Y, rwdata, cmap=cmap)
    pl.xlim(0, max(x))
    pl.ylim(0, max(y))
    pl.colorbar(shrink=0.5)
    pl.title('RADOLAN RW Product \n' + rwattrs['datetime'].isoformat())
    pl.tight_layout()

    pl.show()

# =======================================================
if __name__ == '__main__':
    ex_load_radolan()