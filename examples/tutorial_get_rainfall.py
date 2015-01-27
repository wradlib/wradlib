#-------------------------------------------------------------------------------
# Name:        tutorial_get_rainfall.py
# Purpose:
#
# Author:      heistermann
#
# Created:     27/01/2015
# Copyright:   (c) Maik Heistermann 2015
# Licence:     The MIT License
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import wradlib
import numpy as np
import matplotlib.pyplot as plt
import os


def read_data(dtimes):
    """Helper to read raw data from multiple time steps <dtimes>
    """
    data  = np.empty((len(dtimes),360,128))
    for i, dtime in enumerate(dtimes):
        f = os.path.dirname(__file__) + "/" + "data/raa00-dx_10908-%s-fbg---bin" % dtime
        data[i], attrs = wradlib.io.readDX(f)
    return data


def ex_reflectivity_to_rainfall():
    # ------
    # PART 1
    # ------
    # This is an array of typical reflectivity values (dBZ)
    dBZ = np.array([20., 30., 40., 45., 50., 55.])
    print dBZ
    # Reflectivity factor Z
    Z = wradlib.trafo.idecibel(dBZ)
    print Z
    # Rainfall intensity using the Marshal-Palmer Z(R) parameters
    R = wradlib.zr.z2r(Z, a=200., b=1.6)
    print np.round(R, 2)
    # Convert to rainfall depth assuming a duration of 300 seconds
    depth = wradlib.trafo.r2depth(R, 300)
    print np.round(depth, 2)

    # ------
    # PART 2
    # ------
    # read data from radar Feldberg for three consecutive 5 minute time steps
    dtimes = ["0806021735","0806021740","0806021745"]
    dBZ = read_data(dtimes)
    Z = wradlib.trafo.idecibel(dBZ)
    R = wradlib.zr.z2r(Z, a=200., b=1.6)
    depth = wradlib.trafo.r2depth(R, 300)
    # accumulate 15 minute rainfall depth over all three 5 minute intervals
    accum = np.sum(depth, axis=0)
    # plot PPI of 15 minute rainfall depth
    ax, cf = wradlib.vis.plot_ppi(accum, cmap="spectral")
    plt.xlabel("Easting from radar (km)")
    plt.ylabel("Northing from radar (km)")
    plt.title("Radar Feldberg\n15 min. rainfall depth, 2008-06-02 17:30-17:45 UTC")
    cb = plt.colorbar(cf, shrink=0.8)
    cb.set_label("mm")
    plt.xlim(-128,128)
    plt.ylim(-128,128)
    plt.grid(color="grey")

    plt.show()
    plt.close()

if __name__ == '__main__':
    ex_reflectivity_to_rainfall()