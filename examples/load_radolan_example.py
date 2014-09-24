#------------------------------------------------------------------------------
# Name:        load_rainbow_example.py
# Purpose:     show how to load and work with Gematronik Rainbow5 format
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
#import datetime
 

def ex_load_radolan():

    pg_filename = os.path.dirname(__file__) + '/' + '/data/raa00-pc_10015-1408030905-dwd---bin.gz'
    rw_filename = os.path.dirname(__file__) + '/' + '/data/raa01-rw_10000-1408030950-dwd---bin.gz'

    # load radolan files
    pgdata, pgattrs = wrl.io.read_RADOLAN_composite(pg_filename)
    rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)

    # print the available attributes
    print("PG Attributes:", pgattrs)
    print("RW Attributes:", rwattrs)

    # plot the images side by side
    pl.subplot(121, aspect='equal')
    x = np.arange(0,pgdata.shape[0]+1,1)
    y = np.arange(0,pgdata.shape[1]+1,1)
    X,Y = np.meshgrid(x,y)
    cmap = pl.cm.jet
    bounds = [0,1,2,3,4,5,6,9]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    pl.pcolormesh(X, Y, pgdata, cmap=cmap, norm=norm)
    pl.xlim(0,max(x))
    pl.ylim(0,max(y))
    pl.colorbar(shrink=0.5)
    pl.title('RADOLAN PG Product \n' + pgattrs['datetime'].isoformat())

    pl.subplot(122, aspect='equal')
    x = np.arange(0,rwdata.shape[0]+1,1)
    y = np.arange(0,rwdata.shape[1]+1,1)
    X,Y = np.meshgrid(x,y)
    bounds = np.linspace(0, np.max(rwdata))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    pl.pcolormesh(X, Y, rwdata, cmap=cmap, norm=norm, vmax=np.max(rwdata), vmin=0)
    pl.xlim(0,max(x))
    pl.ylim(0,max(y))
    pl.colorbar(shrink=0.5)
    pl.title('RADOLAN RW Product \n' + rwattrs['datetime'].isoformat())
    pl.tight_layout()

    pl.show()

# =======================================================
if __name__ == '__main__':
    ex_load_radolan()