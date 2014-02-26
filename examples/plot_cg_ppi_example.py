#-------------------------------------------------------------------------------
# Name:        plot_cg_ppi_example.py
# Purpose:     show a few examples on how to use wradlib.vis.plot_cg_ppi
#
# Author:      Kai Muehlbauer
#
# Created:     25.02.2014
# Copyright:   (c) Kai Muehlbaier 2014
# Licence:     The MIT License
#-------------------------------------------------------------------------------
import numpy as np
# importing most matplotlib routines at once
import pylab as pl
# well, it's a wradlib example
import wradlib

# load a polar scan and create range and azimuth arrays accordingly
data = np.loadtxt('data/polar_dBZ_tur.gz')
r = np.arange(0, data.shape[1])
az = np.arange(0, data.shape[0])

# mask data array for better presentation
mask_ind = np.where(data <= np.nanmin(data))
data[mask_ind] = np.nan
ma = np.ma.array(data, mask=np.isnan(data))

# change to subplot2 to get one figure with 4 subplots 2x2
# the positions of title and text annotation and the padding of
# colorbar etc may need adjustment
subplot1 = [111, 111, 111, 111]
subplot2 = [221, 222, 223, 224]
subplot = subplot1

# plot simple CG PPI
wradlib.vis.plot_cg_ppi(ma, subplot=subplot[0])
t = pl.title('Simple CG PPI')
t.set_y(1.05)
pl.tight_layout()

# now let's just plot a sector of data
# for this, we need to give the ranges and azimuths explicitly
# and one more than we pass on in the data, because we also may not use
# the autoext-feature, and otherwise the last row and column of our data
# would not be plotted
wradlib.vis.plot_cg_ppi(ma[200:250, 40:80], r[40:81], az[200:251],
                        autoext=False, subplot=subplot[1])
t = pl.title('Sector CG PPI')
t.set_y(1.05)
pl.tight_layout()

# now let's plot with given range and theta arrays
# and plot some annotation and colorbar
cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma, r, az, autoext=True,
                                               subplot=subplot[2])
t = pl.title('Decorated CG RHI')
t.set_y(1.05)
cbar = pl.gcf().colorbar(pm, pad=0.075)
caax.set_xlabel('x_range [km]')
caax.set_ylabel('y_range [km]')
pl.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='right')
cbar.set_label('reflectivity [dBZ]')
pl.tight_layout()

# now let's just plot a sector of data
# and plot some annotation and colorbar
# create an floating axis for range
cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma[200:250, 40:80],
                                               r[40:81], az[200:251],
                                               autoext=False,
                                               subplot=subplot[3])
t = pl.title('Decorated Sector CG PPI')
t.set_y(1.05)
cbar = pl.gcf().colorbar(pm, pad=0.075)
caax.set_xlabel('x_range [km]')
caax.set_ylabel('y_range [km]')
pl.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='right')
cbar.set_label('reflectivity [dBZ]')
cgax.axis["lat"] = cgax.new_floating_axis(0, 240)
cgax.axis["lat"].set_ticklabel_direction('-')
cgax.axis["lat"].label.set_text("range [km]")
cgax.axis["lat"].label.set_rotation(180)
cgax.axis["lat"].label.set_pad(10)
pl.tight_layout()

pl.show()
