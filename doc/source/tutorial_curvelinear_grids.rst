*****************************
Plot data to curvelinear grid
*****************************

=======
Preface
=======

If you are working with radar station data, it is almost ever only available as polar data.
This means you have a 2D-array, one dimension holding the azimuth (**PPI**) or elevation
(**RHI**) angle values and the other holding the range values.

In wradlib it is assumed that the first dimension is over the azimuth/elevation angles,
while the second dimension is over the range bins.

=======================
Create Curvelinear Grid
=======================

The creation process of the curvelinear grid is bundled in the helper function :func:`wradlib.vis.create_cg`.
I will not dwell too much on that, just this far: :func:`wradlib.vis.create_cg` uses a derived Axes implementation.

:func:`wradlib.vis.create_cg` takes scan type ('PPI' or 'RHI') as argument, figure object and grid definition are
optional. The grid creation process generates three axes objects and set some reasonable starting values for labeling.

The returned objects are ``cgax``, ``caax`` and ``paax``.

* cgax : matplotlib toolkit axisartist Axes object
        curvelinear Axeswhich holds the angle-range-grid
* caax : matplotlib Axes object (twin to cgax)
        Cartesian Axes (x-y-grid) for plotting cartesian data
* paax : matplotlib Axes object (parasite to cgax)
        The parasite axes object for plotting polar data

A typical invocation of :func:`wradlib.vis.create_cg` for a **PPI** is::

    # create curvelinear axes
    cgax, caax, paax = create_cg('PPI', fig, subplot)

For plotting actual polar data two functions exist, depending on whether your data holds a **PPI**
(:func:`wradlib.vis.plot_cg_ppi`) or an **RHI** (:func:`wradlib.vis.plot_cg_rhi`).

.. note:: 1. Other than most plotting functions you cannot give an axes object as an argument. All necessary
 axes objects are created on the fly. You may give an figure object and/or an subplot specification as parameter.
 For further information on howto plot multiple cg plots in one figure, have a look at the special section
 :ref:`multiplots`.

 2. When using the ``refrac`` keyword with :func:`wradlib.vis.plot_cg_rhi` the data is plotted to the cartesian
 axis ``caax``.

.. seealso:: :func:`wradlib.vis.create_cg`, :func:`wradlib.vis.plot_cg_ppi`, :func:`wradlib.vis.plot_cg_rhi`

 If you want to learn more about the matplotlib features used with :func:`wradlib.vis.create_cg`, have a look into

 * `Matplotlib AXISARTIST namespace <http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html>`_
 * `Matplotlib AxesGrid Toolkit <http://matplotlib.org/mpl_toolkits/axes_grid/index.html>`_
 * `The Matplotlib AxesGrid Toolkit User’s Guide <http://matplotlib.org/mpl_toolkits/axes_grid/users/index.html>`_

==============
Plotting on CG
==============

* :ref:`plot-cg-ppi`
    - :ref:`simple-cg-ppi`
    - :ref:`deco-cg-ppi`
    - :ref:`sector-cg-ppi`
    - :ref:`special-markers`
    - :ref:`special-specials`
* :ref:`plot-cg-rhi`
* :ref:`multiplots`
    - :ref:`mp-builtin`
    - :ref:`mp-gridspec`
    - :ref:`mp-axesdivider`

.. _plot-cg-ppi:

Plot CG PPI
===========

:func:`wradlib.vis.plot_cg_ppi` is used in this section. There is also an
:download:`example file <../../examples/plot_cg_ppi_example.py>` in the examples
section, which covers most of the shown plotting capabilities.

.. _simple-cg-ppi:

Simple CG PPI
-------------

First we will look into plotting a **PPI**. We start with importing the necessary modules::

    import matplotlib.pyplot as plt
    import numpy as np
    import wradlib

Next, we will load a polar scan from the examples/data folder and prepare it::

    # load a polar scan and create range and azimuth arrays accordingly
    data = np.loadtxt('data/polar_dBZ_tur.gz')
    r = np.arange(0, data.shape[1])
    az = np.arange(0, data.shape[0])
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))

The plotting routine would be invoked like this::

    wradlib.vis.plot_cg_ppi(ma, refrac=False)
    t = plt.title('Simple CG PPI')
    t.set_y(1.05)
    plt.tight_layout()
    plt.show()

For this simple example, we do not need the returned axes. This is the image we will get:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import wradlib
    # load a polar scan and create range and azimuth arrays accordingly
    filename = wradlib.util.get_wradlib_data_file('misc/polar_dBZ_tur.gz')
    data = np.loadtxt(filename)
    r = np.arange(0, data.shape[1])
    az = np.arange(0, data.shape[0])
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    wradlib.vis.plot_cg_ppi(ma, refrac=False)
    t = plt.title('Simple CG PPI')
    t.set_y(1.05)
    plt.tight_layout()
    plt.draw()
    plt.show()

.. _deco-cg-ppi:

Decorated CG PPI
----------------

Now we will make use of some of the capabilities of this curvelinear axes.

The plotting routine would be invoked like thi, adding range and azimuth arrays and using the ``autoext`` feature::

    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma, r, az, autoext=True,
                                               refrac=False)
    t = plt.title('Decorated CG PPI')
    t.set_y(1.05)
    cbar = plt.gcf().colorbar(pm, pad=0.075)
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    plt.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='right')
    cbar.set_label('reflectivity [dBZ]')
    plt.tight_layout()
    plt.show()

You see, that for labeling x- and y-axis the cartesian axis is used. The `azimuth` label
is set via :func:`text`. Also a colorbar is easily added. This is the image we will get now:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import wradlib
    # load a polar scan and create range and azimuth arrays accordingly
    filename = wradlib.util.get_wradlib_data_file('misc/polar_dBZ_tur.gz')
    data = np.loadtxt(filename)
    r = np.arange(0, data.shape[1])
    az = np.arange(0, data.shape[0])
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma, r, az, autoext=True,
                                               refrac=False)
    t = plt.title('Decorated CG PPI')
    t.set_y(1.05)
    cbar = plt.gcf().colorbar(pm, pad=0.075)
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    plt.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='right')
    cbar.set_label('reflectivity [dBZ]')
    plt.tight_layout()
    plt.draw()
    plt.show()

.. _sector-cg-ppi:

Sector CG PPI
-------------

What if I want to plot only an interesting sector of the whole **PPI**? Not as easy, one might think.
Here we go::

    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma[200:250, 40:80],
                                               r[40:81], az[200:251],
                                               autoext=False,
                                               refrac=False)
    t = plt.title('Decorated Sector CG PPI')
    t.set_y(1.05)
    cbar = plt.gcf().colorbar(pm, pad=0.075)
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    plt.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='right')
    cbar.set_label('reflectivity [dBZ]')

We also can generate a so called floating axis using the ``cgax`` now::

    cgax.axis["lat"] = cgax.new_floating_axis(0, 240)
    cgax.axis["lat"].set_ticklabel_direction('-')
    cgax.axis["lat"].label.set_text("range [km]")
    cgax.axis["lat"].label.set_rotation(180)
    cgax.axis["lat"].label.set_pad(10)
    plt.tight_layout()
    plt.draw()
    plt.show()

Let's have a look at the plot:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import wradlib
    # load a polar scan and create range and azimuth arrays accordingly
    filename = wradlib.util.get_wradlib_data_file('misc/polar_dBZ_tur.gz')
    data = np.loadtxt(filename)
    r = np.arange(0, data.shape[1])
    az = np.arange(0, data.shape[0])
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma[200:250, 40:80],
                                               r[40:81], az[200:251],
                                               autoext=False,
                                               refrac=False)
    t = plt.title('Decorated Sector CG PPI')
    t.set_y(1.05)
    cbar = plt.gcf().colorbar(pm, pad=0.075)
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    plt.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='right')
    cbar.set_label('reflectivity [dBZ]')
    cgax.axis["lat"] = cgax.new_floating_axis(0, 240)
    cgax.axis["lat"].set_ticklabel_direction('-')
    cgax.axis["lat"].label.set_text("range [km]")
    cgax.axis["lat"].label.set_rotation(180)
    cgax.axis["lat"].label.set_pad(10)
    plt.tight_layout()
    plt.draw()
    plt.show()

.. _special-markers:

Special Markers
---------------

One more good thing about curvelinear axes is that you can plot polar as well as cartesian data. However,
you have to be careful, where to plot. Polar data has to be plottet to the parasite axis (``paax``). Cartesian
data can be plottet to ``caax``, although you can also plot cartesian data to the main ``cgax``.

Anyway, it is easy to overlay your polar data, with other station data (e.g. gauges).
Taking the former sector example, we can plot some additional **stations**::

    #plot on cartesian axis
    caax.plot(-60, -60, 'ro', label="caax")
    caax.plot(-50, -70, 'ro')
    # plot on polar axis
    paax.plot(220, 90, 'bo', label="paax")
    # plot on cg axis (same as on cartesian axis)
    cgax.plot(-50, -60, 'go', label="cgax")
    # legend on main cg axis
    cgax.legend()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import wradlib
    # load a polar scan and create range and azimuth arrays accordingly
    filename = wradlib.util.get_wradlib_data_file('misc/polar_dBZ_tur.gz')
    data = np.loadtxt(filename)
    r = np.arange(0, data.shape[1])
    az = np.arange(0, data.shape[0])
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma[200:250, 40:80],
                                               r[40:81], az[200:251],
                                               autoext=False,
                                               refrac=False)
    t = plt.title('Decorated Sector CG PPI')
    t.set_y(1.05)
    cbar = plt.gcf().colorbar(pm, pad=0.075)
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    plt.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='right')
    cbar.set_label('reflectivity [dBZ]')
    cgax.axis["lat"] = cgax.new_floating_axis(0, 240)
    cgax.axis["lat"].set_ticklabel_direction('-')
    cgax.axis["lat"].label.set_text("range [km]")
    cgax.axis["lat"].label.set_rotation(180)
    cgax.axis["lat"].label.set_pad(10)
    #plot on cartesian axis
    caax.plot(-60, -60, 'ro', label="caax")
    caax.plot(-50, -70, 'ro')
    # plot on polar axis
    paax.plot(220, 90, 'bo', label="paax")
    # plot on cg axis (same as on cartesian axis)
    cgax.plot(-50, -60, 'go', label="cgax")
    # legend on main cg axis
    cgax.legend()
    plt.tight_layout()
    plt.draw()
    plt.show()

.. _special-specials:

Special Specials
----------------

But there is more to know, when using the curvelinear grids! As an example, you can get access to the underlying
``cgax`` ``grid_helper`` to change the azimuth and range resolution::

    from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
    gh = cgax.get_grid_helper()
    # set azimuth resolution to 20deg
    locs = [i for i in np.arange(0., 359., 5.)]
    gh.grid_finder.grid_locator1 = FixedLocator(locs)
    gh.grid_finder.tick_formatter1 = DictFormatter(dict([(i, r"${0:.0f}^\circ$".format(i)) for i in locs]))
    gh.grid_finder.grid_locator2._nbins = 20
    gh.grid_finder.grid_locator2._steps = [1, 1.5, 2, 2.5, 5, 10]

The use of ``FixedLocator`` and ``DictFormatter`` should be clear. The use of ``_nbins`` and ``_steps`` is
a bit of head-twisting. With ``_steps`` you can set the possible divisions of the range. In connection with
the ``_nbins`` the range grid is created depending on maximum range. In the above situation with ``_nbins``
set to 10 we get an range grid resolution of 25 (divider 2.5). When setting steps to 20 we get a resolution
of 15 (divider 1.5). Choosing 30 lead to resolution of 10 (divider 1/10). So it may be good to play around
a bit, for wanted results.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import wradlib
    from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
    import mpl_toolkits.axisartist.angle_helper as angle_helper
    # load a polar scan and create range and azimuth arrays accordingly
    filename = wradlib.util.get_wradlib_data_file('misc/polar_dBZ_tur.gz')
    data = np.loadtxt(filename)
    r = np.arange(0, data.shape[1])
    az = np.arange(0, data.shape[0])
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma[200:250, 40:80],
                                               r[40:81], az[200:251],
                                               autoext=False,
                                               refrac=False)
    t = plt.title('Decorated Sector CG PPI')
    t.set_y(1.05)
    cbar = plt.gcf().colorbar(pm, pad=0.075)
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    plt.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='right')
    cbar.set_label('reflectivity [dBZ]')
    gh = cgax.get_grid_helper()
    # set azimuth resolution to 15deg
    locs = [i for i in np.arange(0., 359., 5.)]
    gh.grid_finder.grid_locator1 = FixedLocator(locs)
    gh.grid_finder.tick_formatter1 = DictFormatter(dict([(i, r"${0:.0f}^\circ$".format(i)) for i in locs]))
    gh.grid_finder.grid_locator2._nbins = 30
    gh.grid_finder.grid_locator2._steps = [1, 1.5, 2, 2.5, 5, 10]
    cgax.axis["lat"] = cgax.new_floating_axis(0, 240)
    cgax.axis["lat"].set_ticklabel_direction('-')
    cgax.axis["lat"].label.set_text("range [km]")
    cgax.axis["lat"].label.set_rotation(180)
    cgax.axis["lat"].label.set_pad(10)
    plt.tight_layout()
    plt.draw()
    plt.show()

As you might have noticed the cartesian grid remained the same and the azimuth labels are bit overplottet.
But matplotlib would be not matplotlib if there would be no solution. First we take care of the labeling.
We push the titel a bit higher to get space and toggle the ``caax`` labels to right and top::

    t = plt.title('Very Special Sector CG PPI')
    t.set_y(1.1)
    caax.toggle_axisline()

Then we **toggle** "left" and "right" and "top" and "bottom" axis behaviour::

    # make ticklabels of left and bottom axis visible,
    cgax.axis["left"].major_ticklabels.set_visible(True)
    cgax.axis["bottom"].major_ticklabels.set_visible(True)
    cgax.axis["left"].get_helper().nth_coord_ticks = 0
    cgax.axis["bottom"].get_helper().nth_coord_ticks = 0
    # and also set tickmarklength to zero for better presentation
    cgax.axis["right"].major_ticks.set_ticksize(0)
    cgax.axis["top"].major_ticks.set_ticksize(0)

    # make ticklabels of right and top axis unvisible,
    # because we use the caax for them
    cgax.axis["right"].major_ticklabels.set_visible(False)
    cgax.axis["top"].major_ticklabels.set_visible(False)
    # and also set tickmarklength to zero for better presentation
    cgax.axis["right"].major_ticks.set_ticksize(0)
    cgax.axis["top"].major_ticks.set_ticksize(0)

We also have to put the colorbar a bit to the side and alter the location of the azimuth label::

    cbar = plt.gcf().colorbar(pm, pad=0.1)
    plt.text(0.025, -0.065, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='left')

Everything else stays the same. So now we have:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import wradlib
    from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
    from matplotlib.ticker import MaxNLocator
    # load a polar scan and create range and azimuth arrays accordingly
    filename = wradlib.util.get_wradlib_data_file('misc/polar_dBZ_tur.gz')
    data = np.loadtxt(filename)
    r = np.arange(0, data.shape[1])
    az = np.arange(0, data.shape[0])
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma[200:250, 40:80],
                                               r[40:81], az[200:251],
                                               autoext=False,
                                               refrac=False)
    t = plt.title('Very Special Sector CG PPI')
    t.set_y(1.1)
    cbar = plt.gcf().colorbar(pm, pad=0.1)
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    caax.toggle_axisline()
    caax.grid(True)
    # make ticklabels of left and bottom axis visible,
    cgax.axis["left"].major_ticklabels.set_visible(True)
    cgax.axis["bottom"].major_ticklabels.set_visible(True)
    cgax.axis["left"].get_helper().nth_coord_ticks = 0
    cgax.axis["bottom"].get_helper().nth_coord_ticks = 0
    # and also set tickmarklength to zero for better presentation
    cgax.axis["right"].major_ticks.set_ticksize(0)
    cgax.axis["top"].major_ticks.set_ticksize(0)
    # make ticklabels of left and bottom axis unvisible,
    # because we are drawing them
    cgax.axis["right"].major_ticklabels.set_visible(False)
    cgax.axis["top"].major_ticklabels.set_visible(False)
    # and also set tickmarklength to zero for better presentation
    cgax.axis["right"].major_ticks.set_ticksize(0)
    cgax.axis["top"].major_ticks.set_ticksize(0)
    plt.text(0.025, -0.065, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='left')
    cbar.set_label('reflectivity [dBZ]')
    gh = cgax.get_grid_helper()
    # set azimuth resolution to 5deg
    locs = [i for i in np.arange(0., 359., 5.)]
    gh.grid_finder.grid_locator1 = FixedLocator(locs)
    gh.grid_finder.tick_formatter1 = DictFormatter(dict([(i, r"${0:.0f}^\circ$".format(i)) for i in locs]))
    #gh.grid_finder.grid_locator1 = FixedLocator([i for i in np.arange(0, 359, 5, dtype=np.int32)])
    #gh.grid_finder.grid_locator1 = LocatorDMS(15)
    gh.grid_finder.grid_locator2._nbins = 30
    gh.grid_finder.grid_locator2._steps = [1, 1.5, 2, 2.5, 5, 10]
    cgax.axis["lat"] = cgax.new_floating_axis(0, 240)
    cgax.axis["lat"].set_ticklabel_direction('-')
    cgax.axis["lat"].label.set_text("range [km]")
    cgax.axis["lat"].label.set_rotation(180)
    cgax.axis["lat"].label.set_pad(10)
    plt.tight_layout()
    plt.draw()
    plt.show()

Ups, we forgot to adapt the ticklabels of the cartesian axes::

    from matplotlib.ticker import MaxNLocator
    caax.xaxis.set_major_locator(MaxNLocator(15))
    caax.yaxis.set_major_locator(MaxNLocator(15))

With little effort we got a better (IMHO) representation:

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import wradlib
    from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
    from matplotlib.ticker import MaxNLocator
    # load a polar scan and create range and azimuth arrays accordingly
    filename = wradlib.util.get_wradlib_data_file('misc/polar_dBZ_tur.gz')
    data = np.loadtxt(filename)
    r = np.arange(0, data.shape[1])
    az = np.arange(0, data.shape[0])
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    cgax, caax, paax, pm = wradlib.vis.plot_cg_ppi(ma[200:250, 40:80],
                                               r[40:81], az[200:251],
                                               autoext=False,
                                               refrac=False)
    t = plt.title('Very Special Sector CG PPI')
    t.set_y(1.1)
    cbar = plt.gcf().colorbar(pm, pad=0.1)
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    caax.toggle_axisline()
    caax.grid(True)
    caax.xaxis.set_major_locator(MaxNLocator(15))
    caax.yaxis.set_major_locator(MaxNLocator(15))
    # make ticklabels of left and bottom axis visible,
    cgax.axis["left"].major_ticklabels.set_visible(True)
    cgax.axis["bottom"].major_ticklabels.set_visible(True)
    cgax.axis["left"].get_helper().nth_coord_ticks = 0
    cgax.axis["bottom"].get_helper().nth_coord_ticks = 0
    # and also set tickmarklength to zero for better presentation
    cgax.axis["right"].major_ticks.set_ticksize(0)
    cgax.axis["top"].major_ticks.set_ticksize(0)
    # make ticklabels of left and bottom axis unvisible,
    # because we are drawing them
    cgax.axis["right"].major_ticklabels.set_visible(False)
    cgax.axis["top"].major_ticklabels.set_visible(False)
    # and also set tickmarklength to zero for better presentation
    cgax.axis["right"].major_ticks.set_ticksize(0)
    cgax.axis["top"].major_ticks.set_ticksize(0)
    plt.text(0.025, -0.065, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='left')
    cbar.set_label('reflectivity [dBZ]')
    gh = cgax.get_grid_helper()
    # set azimuth resolution to 5deg
    locs = [i for i in np.arange(0., 359., 5.)]
    gh.grid_finder.grid_locator1 = FixedLocator(locs)
    gh.grid_finder.tick_formatter1 = DictFormatter(dict([(i, r"${0:.0f}^\circ$".format(i)) for i in locs]))
    gh.grid_finder.grid_locator2._nbins = 30
    gh.grid_finder.grid_locator2._steps = [1, 1.5, 2, 2.5, 5, 10]
    cgax.axis["lat"] = cgax.new_floating_axis(0, 240)
    cgax.axis["lat"].set_ticklabel_direction('-')
    cgax.axis["lat"].label.set_text("range [km]")
    cgax.axis["lat"].label.set_rotation(180)
    cgax.axis["lat"].label.set_pad(10)
    plt.tight_layout()
    plt.draw()
    plt.show()

.. _plot-cg-rhi:

Plot CG RHI
===========

:func:`wradlib.vis.plot_cg_rhi` is used in this section. There is also an
:download:`example file <../../examples/plot_cg_rhi_example.py>` in the examples
section, which covers most of the shown plotting capabilities.

An CG RHI plot is a little different compared to an CG PPI plot. I covers only one quadrant and
the data is plottet counterclockwise from "east" (3 o'clock) to "north" (12 o'clock).

Everything else is much the same and you can do whatever you want as shown in the section :ref:`plot-cg-ppi`.

So just a quick example of an cg rhi plot with some decorations::

    import matplotlib.pyplot as plt
    import numpy as np
    # well, it's a wradlib example
    import wradlib
    from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
    # reading in data, range and theta arrays from special rhi hdf5 file
    file = wradlib.util.get_wradlib_data_file('hdf5/polar_rhi_dBZ_bonn.h5')
    data, meta = wradlib.io.from_hdf5(file, dataset='data')
    r, meta = wradlib.io.from_hdf5(file, dataset='range')
    th, meta = wradlib.io.from_hdf5(file, dataset='theta')
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    #----------------------------------------------------------------
    # the simplest call, plot cg rhi in new window
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, r=r, th=th, rf=1e3, refrac=False,
                                           subplot=111)
    t = plt.title('Decorated CG RHI')
    t.set_y(1.05)
    cgax.set_ylim(0,12)
    cbar = plt.gcf().colorbar(pm, pad=0.05)
    cbar.set_label('reflectivity [dBZ]')
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    plt.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='right')
    gh = cgax.get_grid_helper()
    # set theta to some nice values
    #gh.grid_finder.grid_locator1 = FixedLocator([i for i in np.arange(0, 359, 5)])
    locs = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.,
                    15., 16., 17., 18., 20., 22., 25., 30., 35.,  40., 50., 60., 70., 80., 90.]
    gh.grid_finder.grid_locator1 = FixedLocator(locs)
    gh.grid_finder.tick_formatter1 = DictFormatter(dict([(i, r"${0:.0f}^\circ$".format(i)) for i in locs]))
    plt.tight_layout()
    plt.plot()
    plt.show()

As you can see, the ``grid_locator1`` for the theta angles is overwritten and now the grid is much finer.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    # well, it's a wradlib example
    import wradlib
    from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
    # reading in data, range and theta arrays from special rhi hdf5 file
    file = wradlib.util.get_wradlib_data_file('hdf5/polar_rhi_dBZ_bonn.h5')
    data, meta = wradlib.io.from_hdf5(file, dataset='data')
    r, meta = wradlib.io.from_hdf5(file, dataset='range')
    th, meta = wradlib.io.from_hdf5(file, dataset='theta')
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    #----------------------------------------------------------------
    # the simplest call, plot cg rhi in new window
    cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma, r=r, th=th, rf=1e3, refrac=False,
                                               subplot=111)
    t = plt.title('Decorated CG RHI')
    t.set_y(1.05)
    cgax.set_ylim(0,12)
    cbar = plt.gcf().colorbar(pm, pad=0.05)
    cbar.set_label('reflectivity [dBZ]')
    caax.set_xlabel('x_range [km]')
    caax.set_ylabel('y_range [km]')
    plt.text(1.0, 1.05, 'azimuth', transform=caax.transAxes, va='bottom',
        ha='right')
    gh = cgax.get_grid_helper()
    # set theta to some nice values
    #gh.grid_finder.grid_locator1 = FixedLocator([i for i in np.arange(0, 359, 5)])
    locs = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.,
                    15., 16., 17., 18., 20., 22., 25., 30., 35.,  40., 50., 60., 70., 80., 90.]
    gh.grid_finder.grid_locator1 = FixedLocator(locs)
    gh.grid_finder.tick_formatter1 = DictFormatter(dict([(i, r"${0:.0f}^\circ$".format(i)) for i in locs]))
    plt.tight_layout()
    plt.plot()
    plt.show()

.. _multiplots:

Plotting on Grids
=================

There are serveral possibilities to plot multiple cg plots in one figure. Since both plotting routines
are equipped with the same mechanisms it is concentrated mostly on **RHI** plots.

.. note:: Using the :func:`tight_layout` and :func:`subplots_adjust` functions most alignment problems
 can be avoided.

* :ref:`mp-builtin`
* :ref:`mp-gridspec`
* :ref:`mp-axesdivider`

.. _mp-builtin:

The Built-In Method
-------------------

Using the matplotlib grid definition for the parameter ``subplot``, we can easily plot two or more plots
in one figure on a regular grid::

    subplots = [221, 222, 223, 224]
    for sp in subplots:
        cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma,
                                                       r, th, rf=1e3, autoext=True,
                                                       refrac=False, subplot=sp)
        t = plt.title('CG RHI #%(sp)d' %locals())
        t.set_y(1.1)
        cgax.set_ylim(0, 15)
        cbar = plt.gcf().colorbar(pm, pad=0.125)
        caax.set_xlabel('range [km]')
        caax.set_ylabel('height [km]')
        gh = cgax.get_grid_helper()
        # set theta to some nice values
        locs = [0., 5., 10., 15., 20., 30., 40., 60., 90.]
        gh.grid_finder.grid_locator1 = FixedLocator(locs)
        gh.grid_finder.tick_formatter1 = DictFormatter(dict([(i, r"${0:.0f}^\circ$".format(i)) for i in locs]))
        cbar.set_label('reflectivity [dBZ]')
    plt.tight_layout()

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    # well, it's a wradlib example
    import wradlib
    from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
    # reading in data, range and theta arrays from special rhi hdf5 file
    file = wradlib.util.get_wradlib_data_file('hdf5/polar_rhi_dBZ_bonn.h5')
    data, meta = wradlib.io.from_hdf5(file, dataset='data')
    r, meta = wradlib.io.from_hdf5(file, dataset='range')
    th, meta = wradlib.io.from_hdf5(file, dataset='theta')
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    subplots = [221, 222, 223, 224]
    for sp in subplots:
        cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma,
                                                       r, th, rf=1e3, autoext=True,
                                                       refrac=False, subplot=sp)
        t = plt.title('CG RHI #%(sp)d' %locals())
        t.set_y(1.1)
        cgax.set_ylim(0, 15)
        cbar = plt.gcf().colorbar(pm, pad=0.125)
        caax.set_xlabel('range [km]')
        caax.set_ylabel('height [km]')
        gh = cgax.get_grid_helper()
        # set theta to some nice values
        locs = [0., 5., 10., 15., 20., 30., 40., 60., 90.]
        gh.grid_finder.grid_locator1 = FixedLocator(locs)
        gh.grid_finder.tick_formatter1 = DictFormatter(dict([(i, r"${0:.0f}^\circ$".format(i)) for i in locs]))
        cbar.set_label('reflectivity [dBZ]')
    plt.tight_layout()
    plt.draw()
    plt.show()

.. _mp-gridspec:

The GridSpec Method
-------------------

Here the abilities of `Matplotlib GridSpec <http://matplotlib.org/users/gridspec.html>`_ are used.
Now we can also plot on irregular grids. Just create your grid as follows::

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 3)

Then you can take the GridSpec object as an input to the parameter ``subplot``::

    subplots = [gs[0, :], gs[1, :-1], gs[1:, -1], gs[-1, 0], gs[-1, -2]]
    for i, sp in enumerate(subplots):
        cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma,
                                                       r, th, rf=1e3, autoext=True,
                                                       refrac=False, subplot=sp)
        t = plt.title('CG RHI #%(i)d' %locals())
        t.set_y(1.1)
        cgax.set_ylim(0, 15)
        cbar = plt.gcf().colorbar(pm, pad=0.125)
        caax.set_xlabel('range [km]')
        caax.set_ylabel('height [km]')
        gh = cgax.get_grid_helper()
        # set theta to some nice values
        locs = [0., 5., 10., 15., 20., 30., 40., 60., 90.]
        gh.grid_finder.grid_locator1 = FixedLocator(locs)
        gh.grid_finder.tick_formatter1 = DictFormatter(dict([(i, r"${0:.0f}^\circ$".format(i)) for i in locs]))
        cbar.set_label('reflectivity [dBZ]')
    plt.tight_layout()

.. plot::

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    # well, it's a wradlib example
    import wradlib
    from mpl_toolkits.axisartist.grid_finder import FixedLocator, DictFormatter
    # reading in data, range and theta arrays from special rhi hdf5 file
    file = wradlib.util.get_wradlib_data_file('hdf5/polar_rhi_dBZ_bonn.h5')
    data, meta = wradlib.io.from_hdf5(file, dataset='data')
    r, meta = wradlib.io.from_hdf5(file, dataset='range')
    th, meta = wradlib.io.from_hdf5(file, dataset='theta')
    # mask data array for better presentation
    mask_ind = np.where(data <= np.nanmin(data))
    data[mask_ind] = np.nan
    ma = np.ma.array(data, mask=np.isnan(data))
    gs = gridspec.GridSpec(3, 3)
    subplots = [gs[0, :], gs[1, :-1], gs[1:, -1], gs[-1, 0], gs[-1, -2]]
    cbarpad = [0.05, 0.075, 0.2, 0.2, 0.2]
    labelpad = [1.25, 1.25, 1.1, 1.25, 1.25]
    for i, sp in enumerate(subplots):
        cgax, caax, paax, pm = wradlib.vis.plot_cg_rhi(ma,
                                                       r, th, rf=1e3, autoext=True,
                                                       refrac=False, subplot=sp)
        t = plt.title('CG RHI #%(i)d' %locals())
        t.set_y(labelpad[i])
        cgax.set_ylim(0, 15)
        cbar = plt.gcf().colorbar(pm, pad=cbarpad[i])
        caax.set_xlabel('range [km]')
        caax.set_ylabel('height [km]')
        gh = cgax.get_grid_helper()
        # set theta to some nice values
        locs = [0., 5., 10., 15., 20., 30., 40., 60., 90.]
        gh.grid_finder.grid_locator1 = FixedLocator(locs)
        gh.grid_finder.tick_formatter1 = DictFormatter(dict([(i, r"${0:.0f}^\circ$".format(i)) for i in locs]))
        cbar.set_label('reflectivity [dBZ]')
    plt.tight_layout()
    plt.draw()
    plt.show()

Some padding has to be adjusted to get a nice plot.

.. seealso:: Anyway, there is further potential to customize your plots using the methods described in
 `Matplotlib GridSpec <http://matplotlib.org/users/gridspec.html>`_

.. _mp-axesdivider:

The AxesDivider Method
----------------------

Here the capabilities of `Matplotlib AxesGrid1 <http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#axes-grid1>`_ are used.

We make a **PPI** now, it matches much better. Just plot your **PPI** data and create an axes divider::

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullFormatter, FuncFormatter, MaxNLocator
    divider = make_axes_locatable(cgax)

Now you can easily append more axes to plot some other things, eg a maximum intensity projection::

    axMipX = divider.append_axes("top", size=1.2, pad=0.1, sharex=cgax))
    axMipY = divider.append_axes("right", size=1.2, pad=0.1, sharey=cgax))

We need to set some locators and formatters::

    # make some labels invisible
    cgax.axis["right"].major_ticklabels.set_visible(False)
    cgax.axis["top"].major_ticklabels.set_visible(False)
    axMipX.xaxis.set_major_formatter(NullFormatter())
    axMipX.yaxis.set_major_formatter(FuncFormatter(mip_formatter))
    axMipX.yaxis.set_major_locator(MaxNLocator(5))
    axMipY.yaxis.set_major_formatter(NullFormatter())
    axMipY.xaxis.set_major_formatter(FuncFormatter(mip_formatter))
    axMipY.xaxis.set_major_locator(MaxNLocator(5)

OK, the mip data is missing, we use the :func:`wradlib.util.maximum_intensity_projection`::

    # set angle of cut and scan elevation
    angle = 0.0
    elev = 0.0
    # first is for x-axis, second one is for y-axis
    xs, ys, mip1 = wradlib.util.maximum_intensity_projection(data, r=d1, az=d2, angle=angle, elev=elev)
    xs, ys, mip2 = wradlib.util.maximum_intensity_projection(data, r=d1, az=d2, angle=90+angle, elev=elev)

We may also need a new formatter::

    def mip_formatter(x, pos):
        x = x / 1000.
        fmt_str = '{:g}'.format(x)
        if np.abs(x) > 0 and np.abs(x) < 1:
            return fmt_str.replace("0", "", 1)
        else:
            return fmt_str

Now let's finalize the whole thing::

    ma = np.ma.array(mip1, mask=np.isnan(mip1))
    axMipX.pcolormesh(xs, ys, ma)
    axMipX.set_xlim(-np.max(d1),np.max(d1))
    axMipX.set_ylim(0, wradlib.georef.beam_height_n(d1[-2], elev))
    ma = np.ma.array(mip2, mask=np.isnan(mip2))
    axMipY.pcolormesh(ys.T, xs.T, ma.T)
    axMipY.set_ylim(-np.max(d1),np.max(d1))
    axMipY.set_xlim(0, wradlib.georef.beam_height_n(d1[-2], elev))
    axMipX.set_ylabel('height [km]')
    axMipY.set_xlabel('height [km]')
    axMipX.grid(True)
    axMipY.grid(True)
    t = plt.gcf().suptitle('AxesDivider CG-MIP Example')
    plt.draw()
    plt.show()

And this looks actually very nice:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    import wradlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import NullFormatter, FuncFormatter, MaxNLocator

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

    filename = wradlib.util.get_wradlib_data_file('misc/polar_dBZ_tur.gz')
    data = np.loadtxt(filename)
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
    t = plt.gcf().suptitle('AxesDivider CG-MIP Example')

    plt.draw()
    plt.show()


.. seealso:: `Matplotlib AxesGrid1 <http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#axes-grid1>`_
