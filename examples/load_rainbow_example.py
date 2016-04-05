# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import wradlib as wrl
import matplotlib.pyplot as pl
# pl.interactive(True)
import numpy as np
import os


def ex_load_rainbow():
    filename = os.path.join(os.path.dirname(__file__),
                            'data/2013070308340000dBuZ.azi')

    # load rainbow file contents to dict
    rbdict = wrl.io.read_Rainbow(filename)

    # get azimuthal data
    azi = rbdict['volume']['scan']['slice']['slicedata']['rayinfo']['data']
    azidepth = float(rbdict['volume']['scan']['slice']
                     ['slicedata']['rayinfo']['@depth'])
    azirange = float(rbdict['volume']['scan']['slice']
                     ['slicedata']['rayinfo']['@rays'])
    azires = float(rbdict['volume']['scan']['slice']['anglestep'])
    azi = (azi * azirange / 2**azidepth) * azires

    # create range array
    stoprange = float(rbdict['volume']['scan']['slice']['stoprange'])
    rangestep = float(rbdict['volume']['scan']['slice']['rangestep'])
    r = np.arange(0, stoprange, rangestep)

    # get reflectivity data
    data = rbdict['volume']['scan']['slice']['slicedata']['rawdata']['data']
    datadepth = float(rbdict['volume']['scan']['slice']
                      ['slicedata']['rawdata']['@depth'])
    datamin = float(rbdict['volume']['scan']['slice']
                    ['slicedata']['rawdata']['@min'])
    datamax = float(rbdict['volume']['scan']['slice']
                    ['slicedata']['rawdata']['@max'])
    data = datamin + data * (datamax - datamin) / 2 ** datadepth

    # get annotation data
    unit = rbdict['volume']['scan']['slice']['slicedata']['rawdata']['@type']
    time = rbdict['volume']['scan']['slice']['slicedata']['@time']
    date = rbdict['volume']['scan']['slice']['slicedata']['@date']
    lon = rbdict['volume']['sensorinfo']['lon']
    lat = rbdict['volume']['sensorinfo']['lat']
    sensortype = rbdict['volume']['sensorinfo']['@type']
    sensorname = rbdict['volume']['sensorinfo']['@name']

    # plot data with annotation
    cgax, caax, paax, pm = wrl.vis.plot_cg_ppi(data, r=r, az=azi)

    title = '{0} {1} {2} {3}\n{4}E {5}N'.format(sensortype, sensorname, date,
                                                time, lon, lat)
    t = pl.title(title, fontsize=12)
    t.set_y(1.1)
    cbar = pl.gcf().colorbar(pm, pad=0.075)
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    pl.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
            ha='right')
    cbar.set_label('reflectivity [' + unit + ']')
    pl.tight_layout()
    pl.show()


# =======================================================
if __name__ == '__main__':
    ex_load_rainbow()
