# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:        plot_cg_rhi_example.py
# Purpose:     show a few examples on how to use wradlib.vis.plot_cg_rhi
#
# Author:      Kai Muehlbauer
#
# Created:     25.02.2014
# Copyright:   (c) Kai Muehlbauer 2014
# Licence:     The MIT License
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
# well, it's a wradlib example
import wradlib

# reading in data, range and theta arrays from special rhi hdf5 file
file = '../../../examples/data/polar_rhi_dBZ_bonn.h5'
data, meta = wradlib.io.from_hdf5(file, dataset='data')
r, meta = wradlib.io.from_hdf5(file, dataset='range')
th, meta = wradlib.io.from_hdf5(file, dataset='theta')
# mask data array for better presentation
mask_ind = np.where(data <= np.nanmin(data))
data[mask_ind] = np.nan
ma = np.ma.array(data, mask=np.isnan(data))
# ----------------------------------------------------------------
# the simplest call, plot cg rhi in new window
cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, refrac=False,
                                               subplot=111)
t = plt.title('Simple CG RHI')
t.set_y(1.05)
plt.tight_layout()
plt.show()
