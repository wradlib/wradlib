# ------------------------------------------------------------------------------
# Name:        plot_cg_ppi_example.py
# Purpose:     show a few examples on how to use wradlib.vis.plot_cg_ppi
#
# Author:      Kai Muehlbauer
#
# Created:     25.02.2014
# Copyright:   (c) Kai Muehlbauer 2014
# Licence:     The MIT License
# ------------------------------------------------------------------------------
import numpy as np
# importing most matplotlib routines at once
import matplotlib.pyplot as pl
#pl.interactive(True)
# well, it's a wradlib example
import wradlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter,FuncFormatter, MaxNLocator
import os


def ex_plot_cg_ppi():
    # pl.interactive(True)
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

    # ----------------------------------------------------------------
    # First, creation of four simple figures
    # figure #1
    # the simplest call, plot cg ppi in new window
    # plot simple CG PPI
    wradlib.vis.plot_cg_ppi(ma, refrac=False)
    t = pl.title('Simple CG PPI')
    t.set_y(1.05)
    pl.tight_layout()

    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # figure #7
    # create figure with co-located x and y-axis
    # using axesgrid1 toolkit
    def mip_formatter(x, pos):
        x = x / 1000.
        fmt_str = '{:g}'.format(x)
        if np.abs(x) > 0 and np.abs(x) < 1:
            return fmt_str.replace("0", "", 1)
        else:
            return fmt_str

    # angle of *cut* through ppi and scan elev.
    angle = 0.0
    elev = 0.0

    data = np.loadtxt(os.path.dirname(__file__) + '/' + 'data/polar_dBZ_tur.gz')
    # we need to have meter here for the georef function inside mip
    d1 = np.arange(data.shape[1], dtype=np.float) * 1000
    d2 = np.arange(data.shape[0], dtype=np.float)
    data = np.roll(data, (d2 >= angle).nonzero()[0][0], axis=0)

    # calculate max intensity proj
    xs, ys, mip1 = wradlib.util.maximum_intensity_projection(data, r=d1, az=d2, angle=angle, elev=elev)
    xs, ys, mip2 = wradlib.util.maximum_intensity_projection(data, r=d1, az=d2, angle=90+angle, elev=elev)

    # normal cg plot
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(data, r=d1, az=d2, refrac=True)
    cgax.set_xlim(-np.max(d1),np.max(d1))
    cgax.set_ylim(-np.max(d1),np.max(d1))
    caax.xaxis.set_major_formatter(FuncFormatter(mip_formatter))
    caax.yaxis.set_major_formatter(FuncFormatter(mip_formatter))
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')

    # axes divider section
    divider = make_axes_locatable(cgax)
    axMipX = divider.append_axes("top", size=1.2, pad=0.1, sharex=cgax)
    axMipY = divider.append_axes("right", size=1.2, pad=0.1, sharey=cgax)

    # special handling for labels etc.
    cgax.axis["right"].major_ticklabels.set_visible(False)
    cgax.axis["top"].major_ticklabels.set_visible(False)
    axMipX.xaxis.set_major_formatter(NullFormatter())
    axMipX.yaxis.set_major_formatter(FuncFormatter(mip_formatter))
    axMipX.yaxis.set_major_locator(MaxNLocator(5))
    axMipY.yaxis.set_major_formatter(NullFormatter())
    axMipY.xaxis.set_major_formatter(FuncFormatter(mip_formatter))
    axMipY.xaxis.set_major_locator(MaxNLocator(5))

    # plot max intensity proj
    ma = np.ma.array(mip1, mask=np.isnan(mip1))
    axMipX.pcolormesh(xs, ys, ma)
    ma = np.ma.array(mip2, mask=np.isnan(mip2))
    axMipY.pcolormesh(ys.T, xs.T, ma.T)

    # set labels, limits etc
    axMipX.set_xlim(-np.max(d1),np.max(d1))
    axMipX.set_ylim(0, wradlib.georef.beam_height_n(d1[-2], elev))
    axMipY.set_xlim(0, wradlib.georef.beam_height_n(d1[-2], elev))
    axMipY.set_ylim(-np.max(d1),np.max(d1))
    axMipX.set_ylabel('height [km]')
    axMipY.set_xlabel('height [km]')
    axMipX.grid(True)
    axMipY.grid(True)
    t = pl.gcf().suptitle('AxesDivider CG-MIP Example')

    pl.show()


if __name__ == '__main__':
    ex_plot_cg_ppi()
