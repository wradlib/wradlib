# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Name:        plot_cg_rhi_example.py
# Purpose:     show a few examples on how to use wradlib.vis.plot_cg_rhi
#
# Author:      Kai Muehlbauer
#
# Created:     25.02.2014
# Copyright:   (c) Kai Muehlbauer 2014
# Licence:     The MIT License
#------------------------------------------------------------------------------
import numpy as np
# importing most matplotlib routines at once
import pylab as pl
pl.interactive(True)
# well, it's a wradlib example
import wradlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter
import os

def ex_plot_cg_rhi():

    # reading in data, range and theta arrays from special rhi hdf5 file
    filename = os.path.dirname(__file__) + '/' + 'data/polar_rhi_dBZ_bonn.h5'
    data, meta = wradlib.io.from_hdf5(filename, dataset='data')
    r, meta = wradlib.io.from_hdf5(filename, dataset='range')
    th, meta = wradlib.io.from_hdf5(filename, dataset='theta')
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))

    # cgax - curvelinear grid axis
    # Main axis

    # caax - twin cartesian axis
    # secondary axis for cartesian coordinates (plotting, labeling etc.)

    # paax - polar axis for plotting
    # here all plotting in polar data is done

    # pm - pcolormesh
    # actual plot mappable

    # Remark #1:
    # The tight_layout function is great, but may not lead to
    # satisfactory results in the first place. So labels and annotations
    # may need adjustment

    # Remark #2:
    # This examples makes heavy use of new matlotlib functionality. See
    # function help for more information.

    #----------------------------------------------------------------
    # First, creation of three simple figures
    # figure #1
    # the simplest call, plot cg rhi in new window
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, refrac=False,
                                                   subplot=111)
    t = pl.title('Plain CG RHI')
    t.set_y(1.05)
    pl.tight_layout()

    #----------------------------------------------------------------
    # figure #2
    # now lets plot with given range and theta arrays
    # and plot some annotation and colorbar
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma,
                                                   r, th, autoext=True,
                                                   refrac=False, subplot=111)
    t = pl.title('Decorated CG RHI')
    t.set_y(1.05)
    cbar = pl.gcf().colorbar(pm)
    caax.set_xlabel('range [m]')
    caax.set_ylabel('height [m]')
    pl.text(1.0, 1.05, 'elevation', transform=caax.transAxes,
            va='bottom', ha='right')
    cbar.set_label('reflectivity [dBZ]')
    pl.tight_layout()

    # plot some additional polar and cartesian data
    # cgax and caax plot both cartesian data
    # paax plots polar data
    # plot on cartesian axis
    caax.plot(38640, 10350, 'ro', label="caax")
    caax.plot(25000, 20000, 'ro')
    xxx = wradlib.georef.arc_distance_n(30000, 10)
    yyy = wradlib.georef.beam_height_n(30000, 10)
    caax.plot(xxx, yyy, 'ro')
    # plot on polar axis
    xx, yy = np.meshgrid(10, 30000)
    paax.plot(xx, yy, 'bo')
    paax.plot(20, 30000, 'bo', label="paax")
    # plot on cg axis (same as on cartesian axis)
    cgax.plot(20000, 10000, 'go', label="cgax")
    # legend on main cg axis
    cgax.legend()

    #----------------------------------------------------------------
    # figure #3
    # now lets zoom into the data and apply our range_factor (to km)
    # and plot some annotation and colorbar
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma,
                                                   r, th, rf=1e3, autoext=True,
                                                   refrac=False, subplot=111)
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

    #----------------------------------------------------------------
    # figure #4
    # plot figure #1-3 in one plot
    # stacked vertically
    pl.figure()
    # figure #4-1
    # First, creation of three simple figures
    # the simplest call, plot cg rhi in new window
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, refrac=False,
                                                   subplot=311)
    t = pl.title('Plain CG RHI')
    t.set_y(1.05)
    pl.tight_layout()

    # figure #4-2
    # now lets plot with given range and theta arrays
    # and plot some annotation and colorbar
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma,
                                                   r, th, autoext=True,
                                                   refrac=False,
                                                   subplot=312)
    t = pl.title('Decorated CG RHI')
    t.set_y(1.05)
    cbar = pl.gcf().colorbar(pm)
    caax.set_xlabel('range [m]')
    caax.set_ylabel('height [m]')
    pl.text(1.0, 1.05, 'elevation', transform=caax.transAxes,
            va='bottom', ha='right')
    cbar.set_label('reflectivity [dBZ]')
    pl.tight_layout()

    # figure #4-3
    # now lets zoom into the data and apply our range_factor (to km)
    # and plot some annotation and colorbar
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma,
                                                   r, th, rf=1e3, autoext=True,
                                                   refrac=False, subplot=313)
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

    #----------------------------------------------------------------
    # figure #5
    # create figure with GridSpec
    pl.figure()
    gs = gridspec.GridSpec(3, 3)
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, refrac=False,
                                                   subplot=gs[0, :])
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, refrac=False,
                                                   subplot=gs[1, :-1])
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, refrac=False,
                                                   subplot=gs[1:, -1])
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, refrac=False,
                                                   subplot=gs[-1, 0])
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, refrac=False,
                                                   subplot=gs[-1, -2])
    pl.tight_layout()
    t = pl.gcf().suptitle('GridSpec CG Example')
    pl.tight_layout()

    #----------------------------------------------------------------
    # figure #6
    # create figure with co-located x and y-axis
    # using axesgrid1 toolkit
    x = np.random.randn(ma.shape[1])
    y = np.random.randn(ma.shape[1])
    cgax, caax, paax, cgpm = wradlib.vis.plot_cg_rhi(ma, refrac=False, )
    divider = make_axes_locatable(cgax)
    axHistX = divider.append_axes("top", size=1.2, pad=0.1, sharex=caax)
    axHistY = divider.append_axes("right", size=1.2, pad=0.1, sharey=caax)
    # make some labels invisible
    axHistX.xaxis.set_major_formatter(NullFormatter())
    axHistY.yaxis.set_major_formatter(NullFormatter())
    axHistX.hist(x)
    if not pl.matplotlib.__version__=="1.2.1":
        # There is a bug in matplotlib 1.2.1,
        # see https://github.com/matplotlib/matplotlib/pull/1985
        axHistY.hist(y, orientation='horizontal')
    else:
        axHistY.text(0.5, 0.5, "Does not work with\nmatplotlib 1.2.1",
        horizontalalignment="center", rotation=90, fontsize=15, color="red")
    t = pl.gcf().suptitle('AxesDivider CG Example')
    pl.tight_layout()

    #----------------------------------------------------------------
    # figure #7
    # compare plots with refraction True and refraction False
    pl.figure()
    # figure #7-1
    # check how refraction is doing
    # and plot some annotation and colorbar
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, r, th,
                                                   rf=1e3, autoext=True,
                                                   refrac=True, subplot=211)
    t = pl.title('CG RHI refrac=True')
    t.set_y(1.1)
    cbar = pl.gcf().colorbar(pm)
    caax.set_xlabel('range [km]')
    caax.set_ylabel('height [km]')
    pl.text(1.0, 1.15, 'elevation', transform=caax.transAxes,
            va='bottom', ha='right')
    cbar.set_label('reflectivity [dBZ]')
    # zoom
    cgax.set_xlim(35, 45)
    cgax.set_ylim(10, 12)
    # plot additional data point ((theta, range)
    paax.plot(15, 42, 'bo', label="paax")
    caax. plot(40, 11, 'ro', label="caax")
    cgax.legend()
    pl.tight_layout()
    pl.subplots_adjust(hspace=0.4)


    # figure #7-2
    # check how refraction is doing
    # and plot some annotation and colorbar
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, r, th,
                                                   rf=1e3, autoext=True,
                                                   refrac=False, subplot=212)
    t = pl.title('CG RHI refrac=False')
    t.set_y(1.1)
    cbar = pl.gcf().colorbar(pm)
    caax.set_xlabel('range [km]')
    caax.set_ylabel('height [km]')
    pl.text(1.0, 1.15, 'elevation', transform=caax.transAxes,
            va='bottom', ha='right')
    cbar.set_label('reflectivity [dBZ]')
    # zoom
    cgax.set_xlim(35, 45)
    cgax.set_ylim(10, 12)
    # plot additional data point ((theta, range)
    paax.plot(15, 42, 'bo', label="paax")
    caax. plot(40, 11, 'ro', label="caax")
    cgax.legend()
    pl.tight_layout()

    # show the plots
    pl.show()

if __name__ == '__main__':
    ex_plot_cg_rhi()