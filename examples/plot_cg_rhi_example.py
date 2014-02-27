# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        plot_cg_rhi_example.py
# Purpose:     show a few examples on how to use wradlib.vis.plot_cg_rhi
#
# Author:      Kai Muehlbauer
#
# Created:     25.02.2014
# Copyright:   (c) Kai Muehlbauer 2014
# Licence:     The MIT License
#-------------------------------------------------------------------------------
import numpy as np
# importing most matplotlib routines at once
import pylab as pl
# well, it's a wradlib example
import wradlib

# reading in data, range and theta arrays from special rhi hdf5 file
data, meta = wradlib.io.from_hdf5('data/polar_rhi_dBZ_bonn.h5', dataset='data')
r, meta = wradlib.io.from_hdf5('data/polar_rhi_dBZ_bonn.h5', dataset='range')
th, meta = wradlib.io.from_hdf5('data/polar_rhi_dBZ_bonn.h5', dataset='theta')

# mask data array for better presentation
mask_ind = np.where(data <= np.nanmin(data))
data[mask_ind] = np.nan
ma = np.ma.array(data, mask=np.isnan(data))

# change to subplot2 to get one figure with 3 subplots stacked horizontally
# the positions of title and text annotation need adjustment
subplot1 = [111, 111, 111]
subplot2 = [311, 312, 313]
subplot = subplot1

# the simplest call, plot cg rhi in new window
# cgax - curvilienar grid axis
# caax - twin cartesian axis
# paax - polar axis for plotting
# pm - pcolormesh
cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma,
                                               subplot=subplot[0])
t = pl.title('Plain CG RHI')
t.set_y(1.05)
pl.tight_layout()

# now lets plot with given range and theta arrays
# and plot some annotation and colorbar
cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma,
                                               r, th, autoext=True,
                                               subplot=subplot[1])
t = pl.title('Decorated CG RHI')
t.set_y(1.05)
cbar = pl.gcf().colorbar(pm)
caax.set_xlabel('range [m]')
caax.set_ylabel('height [m]')
pl.text(1.0, 1.05, 'elevation', transform=caax.transAxes,
        va='bottom', ha='right')
cbar.set_label('reflectivity [dBZ]')
pl.tight_layout()

# now lets zoom into the data and apply our range_factor
# and plot some annotation and colorbar
cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma,
                                               r, th, rf=1e3, autoext=True,
                                               subplot=subplot[2])
t = pl.title('Decorated and Zoomed CG RHI')
t.set_y(1.05)
cgax.set_ylim(0, 15)
cbar = pl.gcf().colorbar(pm)
caax.set_xlabel('range [km]')
caax.set_ylabel('height [km]')
pl.text(1.0, 1.05, 'elevation', transform=caax.transAxes,
        va='bottom', ha='right')
cbar.set_label('reflectivity [dBZ]')
pl.tight_layout()

# show the plots
pl.show()
