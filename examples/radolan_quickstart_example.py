#------------------------------------------------------------------------------
# Name:        radolan_quickstart_example.py
# Purpose:     quick plot of radolan rw-product
#
# Author:      Kai Muehlbauer
#
# Created:     11.02.2015
# Copyright:   (c) Kai Muehlbauer 2015
# Licence:     The MIT License
#------------------------------------------------------------------------------

import wradlib as wrl
import numpy as np
import matplotlib.pyplot as pl
import os


def ex_radolan_quickstart():

    # load radolan files
    rw_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-rw_10000-1408102050-dwd---bin.gz'
    print(rw_filename)
    rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)

    # print the available attributes
    print("RW Attributes:", rwattrs)

    # do some masking
    sec = rwattrs['secondary']
    rwdata.flat[sec] = -9999
    rwdata = np.ma.masked_equal(rwdata, -9999)

    # Get coordinates
    radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)
    x = radolan_grid_xy[:, :, 0]
    y = radolan_grid_xy[:, :, 1]

    # plot data
    pl.pcolormesh(x, y, rwdata, cmap="spectral")
    # add colorbar and title
    cb = pl.colorbar(shrink=0.75)
    cb.set_label("mm/h")
    pl.title('RADOLAN RW Product Polar Stereo \n' + rwattrs['datetime'].isoformat())
    pl.grid(color='r')

    pl.show()

# =======================================================
if __name__ == '__main__':
    ex_radolan_quickstart()