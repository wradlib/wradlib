#------------------------------------------------------------------------------
# Name:        plot_cg_ppi_example.py
# Purpose:     show a few examples on how to use wradlib.vis.plot_cg_ppi
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
import unittest
import os

def ex_plot_cg_ppi():

    pl.interactive(True)
    # load a polar scan and create range and azimuth arrays accordingly
    data = np.loadtxt(os.path.dirname(__file__) + '/' + 'data/polar_dBZ_tur.gz')
    r = np.arange(0, data.shape[1])
    az = np.arange(0, data.shape[0])
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
    # satisfactory results in the first place. So labels, annotations
    # and/or axes may need adjustment

    # Remark #2:
    # This examples makes heavy use of new matlotlib functionality. See
    # function help for more information.

    #----------------------------------------------------------------
    # First, creation of four simple figures
    # figure #1
    # the simplest call, plot cg ppi in new window
    # plot simple CG PPI
    wradlib.vis.plot_cg_ppi(ma, refrac=False)
    t = pl.title('Simple CG PPI')
    t.set_y(1.05)
    pl.tight_layout()

    #----------------------------------------------------------------
    # figure #2
    # now let's just plot a sector of data
    # for this, we need to give the ranges and azimuths explicitly
    # and one more than we pass on in the data, because we also may not use
    # the autoext-feature, and otherwise the last row and column of our data
    # would not be plotted
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma[200:250, 40:80],
                                                   r[40:81], az[200:251],
                                                   autoext=False, refrac=False)
    t = pl.title('Sector CG PPI')
    t.set_y(1.05)
    pl.tight_layout()

    # plot some additional polar and cartesian data
    # cgax and caax plot both cartesian data
    # paax plots polar data
    # plot on cartesian axis
    caax.plot(-60, -60, 'ro', label="caax")
    caax.plot(-50, -70, 'ro')
    # plot on polar axis
    xx, yy = np.meshgrid(230, 90)
    paax.plot(xx, yy, 'bo')
    paax.plot(220, 90, 'bo', label="paax")
    # plot on cg axis (same as on cartesian axis)
    cgax.plot(-50, -60, 'go', label="cgax")
    # legend on main cg axis
    cgax.legend()

    #----------------------------------------------------------------
    # figure #3
    # now let's plot with given range and theta arrays
    # and plot some annotation and colorbar
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma, r, az, autoext=True,
                                                   refrac=False)
    t = pl.title('Decorated CG PPI')
    t.set_y(1.05)
    cbar = pl.gcf().colorbar(pm, pad=0.075)
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    pl.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
            ha='right')
    cbar.set_label('reflectivity [dBZ]')
    pl.tight_layout()

    #----------------------------------------------------------------
    # figure #4
    # now let's just plot a sector of data
    # and plot some annotation and colorbar
    # create an floating axis for range
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma[200:250, 40:80],
                                                   r[40:81], az[200:251],
                                                   autoext=False,
                                                   refrac=False)
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

    #----------------------------------------------------------------
    # figure #5
    # plot figure #1-4 in one figure 2x2 grid
    pl.figure()
    # figure #5-1
    # the simplest call, plot cg ppi in new window
    # plot simple CG PPI
    wradlib.vis.plot_cg_ppi(ma, refrac=False, subplot=221)
    t = pl.title('Simple CG PPI')
    t.set_y(1.05)
    pl.tight_layout()

    #----------------------------------------------------------------
    # figure #5-2
    # now let's just plot a sector of data
    # for this, we need to give the ranges and azimuths explicitly
    # and one more than we pass on in the data, because we also may not use
    # the autoext-feature, and otherwise the last row and column of our data
    # would not be plotted
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma[200:250, 40:80],
                                                   r[40:81], az[200:251],
                                                   autoext=False, refrac=False,
                                                   subplot=222)
    t = pl.title('Sector CG PPI')
    t.set_y(1.05)
    pl.tight_layout()

    #----------------------------------------------------------------
    # figure #5-3
    # now let's plot with given range and theta arrays
    # and plot some annotation and colorbar
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma, r, az, autoext=True,
                                                   refrac=False, subplot=223)
    t = pl.title('Decorated CG PPI')
    t.set_y(1.05)
    cbar = pl.gcf().colorbar(pm, pad=0.075)
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    pl.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
            ha='right')
    cbar.set_label('reflectivity [dBZ]')
    pl.tight_layout()

    #----------------------------------------------------------------
    # figure #5-4
    # now let's just plot a sector of data
    # and plot some annotation and colorbar
    # create an floating axis for range
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma[200:250, 40:80],
                                                   r[40:81], az[200:251],
                                                   autoext=False,
                                                   refrac=False,
                                                   subplot=224)
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


    #----------------------------------------------------------------
    # figure #6
    # create figure with GridSpec
    pl.figure()
    gs = gridspec.GridSpec(5, 5)
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma, refrac=False,
                                                   subplot=gs[0:3, 0:3])
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma, refrac=False,
                                                   subplot=gs[0:3, 3:5])
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma, refrac=False,
                                                   subplot=gs[3:5, 0:3])
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma, refrac=False,
                                                   subplot=gs[3:5, 3:5])
    t = pl.gcf().suptitle('GridSpec CG Example')
    pl.tight_layout()

    #----------------------------------------------------------------
    # figure #7
    # create figure with co-located x and y-axis
    # using axesgrid1 toolkit
    x = np.random.randn(ma.shape[1])
    y = np.random.randn(ma.shape[1])
    cgax, caax, paax, cgpm = wradlib.vis.plot_cg_ppi(ma, refrac=False, )
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

    pl.show()

if __name__ == '__main__':
    ex_plot_cg_ppi()
    #testcase = unittest.FunctionTestCase(plot_cg_ppi_example)
    #unittest.TextTestRunner(verbosity=2).run(testcase)