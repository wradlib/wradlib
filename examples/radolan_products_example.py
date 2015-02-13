#------------------------------------------------------------------------------
# Name:        radolan_products_example.py
# Purpose:     plotting several radolan composite products
#
# Author:      Kai Muehlbauer
#
# Created:     11.02.2015
# Copyright:   (c) Kai Muehlbauer 2015
# Licence:     The MIT License
#------------------------------------------------------------------------------

import wradlib as wrl
import matplotlib.pyplot as pl
import numpy as np
import os


def ex_radolan_products():

    # load radolan file
    rx_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-rx_10000-1408102050-dwd---bin.gz'
    ex_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-ex_10000-1408102050-dwd---bin.gz'
    rw_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-rw_10000-1408102050-dwd---bin.gz'
    sf_filename = os.path.dirname(__file__) + '/' + 'data/radolan/raa01-sf_10000-1408102050-dwd---bin.gz'

    rxdata, rxattrs = wrl.io.read_RADOLAN_composite(rx_filename)
    exdata, exattrs = wrl.io.read_RADOLAN_composite(ex_filename)
    rwdata, rwattrs = wrl.io.read_RADOLAN_composite(rw_filename)
    sfdata, sfattrs = wrl.io.read_RADOLAN_composite(sf_filename)

    # mask invalid values
    sec = rwattrs['secondary']
    rwdata.flat[sec] = -9999
    sec = sfattrs['secondary']
    sfdata.flat[sec] = -9999

    rxdata = np.ma.masked_equal(rxdata, -9999) / 2 - 32.5
    exdata = np.ma.masked_equal(exdata, -9999) / 2 - 32.5
    rwdata = np.ma.masked_equal(rwdata, -9999)
    sfdata = np.ma.masked_equal(sfdata, -9999)

    # Get coordinates
    radolan_grid_xy = wrl.georef.get_radolan_grid(900, 900)
    radolan_egrid_xy = wrl.georef.get_radolan_grid(1500, 1400)

    x = radolan_grid_xy[:, :, 0]
    y = radolan_grid_xy[:, :, 1]

    xe = radolan_egrid_xy[:, :, 0]
    ye = radolan_egrid_xy[:, :, 1]

    # plot RX product
    fig = pl.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pm = ax.pcolormesh(x, y, rxdata, cmap='spectral')
    cb = fig.colorbar(pm, shrink=0.75)
    cb.set_label("dBZ")
    pl.xlabel("x [km]")
    pl.ylabel("y [km]")
    pl.title('RADOLAN RX Product \n' + rxattrs['datetime'].isoformat())
    pl.xlim((x[0, 0],x[-1, -1]))
    pl.ylim((y[0, 0],y[-1, -1]))
    pl.grid(color='r')

    # plot EX product
    fig = pl.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pm = ax.pcolormesh(xe, ye, exdata, cmap='spectral')
    cb = fig.colorbar(pm, shrink=0.75)
    cb.set_label("dBZ")
    pl.xlabel("x [km]")
    pl.ylabel("y [km]")
    pl.title('RADOLAN EX Product \n' + exattrs['datetime'].isoformat())
    pl.xlim((xe[0, 0],xe[-1, -1]))
    pl.ylim((ye[0, 0],ye[-1, -1]))
    pl.grid(color='r')

    # plot RW product
    fig = pl.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pm = ax.pcolormesh(x, y, rwdata, cmap='spectral')
    cb = fig.colorbar(pm, shrink=0.75)
    cb.set_label("mm/h")
    pl.xlabel("x [km]")
    pl.ylabel("y [km]")
    pl.title('RADOLAN RW Product \n' + rwattrs['datetime'].isoformat())
    pl.xlim((x[0, 0],x[-1, -1]))
    pl.ylim((y[0, 0],y[-1, -1]))
    pl.grid(color='r')

    # plot SF product
    fig = pl.figure()
    ax = fig.add_subplot(111, aspect='equal')
    pm = ax.pcolormesh(x, y, sfdata, cmap='spectral')
    cb = fig.colorbar(pm, shrink=0.75)
    cb.set_label("mm / 24h")
    pl.xlabel("x [km]")
    pl.ylabel("y [km]")
    pl.title('RADOLAN SF Product \n' + sfattrs['datetime'].isoformat())
    pl.xlim((x[0, 0],x[-1, -1]))
    pl.ylim((y[0, 0],y[-1, -1]))
    pl.grid(color='r')

    pl.show()

# =======================================================
if __name__ == '__main__':
    ex_radolan_products()