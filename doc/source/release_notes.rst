Release Notes
=============

Please note that as long as wradlib releases are within the ``0.y.z`` series, the API cannot be considered stable. We will try to avoid sudden API breaks via deprecation warnings. All wradlib releases come without any warranty. Release notes might be incomplete. See `here <https://bitbucket.org/wradlib/wradlib/commits/all>`_ for a complete record of changes. 

You can install the latest wradlib release from PyPI via ``$ pip install wradlib`` or specific version via ``$ pip install wradlib==x.y.z``.


Bleeding Edge
-------------

*nothing new, yet*


Version 0.6.0
-------------

**Highlights**

wradlib functions concerned with georeferencing will only use projection information based on OSR objects. This version will help you to adapt your code base to this change before next minor release: Using deprecated PROJ.4 strings in call to the following functions/classes: ``vis.plot_ppi``,``vis.plot_ppi_crosshair``, ``georef.projected_bincoords_from_radarspec``, ``verify.PolarNeighbours``, ``vpr.volcoords_from_polar``, ``vpr.volcoords_from_polar_irregular``, ``vpr.make_3D_grid`` will generate a DeprecationWarning and try to correct old calling method at runtime.

Added ability to handle georeferenced raster and vector data

Port of PyRadarMet partial beamblockage calculations by DEM


**New features**

- In georef module a new helper function `create_osr` is added. This function helps to create an osr object for specific projections.
- Bugfix: add multiplication of grid resolution in ``wradlib.georef.get_radolan_grid`` merged from v0.5.1
- Several convenience functions for reading, transforming and projecting as well as plotting rasterfiles and shapefiles including example
- Calculation of partial/cumulative beamblockage with example
- The behaviour of wradlib.adjust.AdjustMFB has been changed. Control parameters are no longer passed via the ``__call__`` function, but, as for all other adjustment classes, via the initialisation of an adjustment object. Documentation has been revised to make the behaviour more transparent. The parameter 'biasby' has been replaced by a dictionary mfb_args that carries all parameters that control the behaviour of AdjustMFB.


**Deprecated features**

Function `georef.create_projstr` is deprecated.

**Removed functions**

*None.*


Version 0.5.1
-------------

**Highlights**

Bugfix: add multiplication of grid resolution in ``wradlib.georef.get_radolan_grid`` merged from v0.4.2


Version 0.5.0
-------------

**Highlights**

From version ``0.6.0`` on, wradlib functions concerned with georeferencing will only use projection information based on OSR objects. This version will help you to adapt your code base to this change before using version ``0.6.0``: Any use of proj4 strings will generate a deprecation warning with the following functions/classes: ``vis.plot_ppi``,``vis.plot_ppi_crosshair``, ``georef.projected_bincoords_from_radarspec``, ``verify.PolarNeighbours``, ``vpr.volcoords_from_polar``, ``vpr.volcoords_from_polar_irregular``, ``vpr.make_3D_grid``.

**New features**

- Two functions exposed documentation library section: :doc:`generated/wradlib.io.read_safnwc` and :doc:`generated/wradlib.vis.plot_max_plan_and_vert`
- New features, changes and deprecations will be addressed in the documentation in the future. This is by highlighting them as *New in Version X.Y.Z*, *Changed in Version X.Y.Z* or *Deprecated since Version X.Y.Z*

**Deprecated features**

*None.*

**Removed functions**

*None.*


Version 0.4.2
-------------

**Highlights**

Bugfix: add multiplication of grid resolution in ``wradlib.georef.get_radolan_grid``


Version 0.4.1
-------------

**Highlights**

From now on, wradlib will generate warnings if keyword parameters of wradlib functions will be or have been changed. This is achieved by using ``wradlib.util.apichange_kwargs`` as a decorator (see ``apichange_example.py`` for examples how these warnings might look like). Please take these warnings seriously and adapt your applications before stepping to a new wradlib release.


Version 0.4.0
-------------

**Highlights**

- Fixed a broken proj4 string for DWD radolan polarstereographic projection in function ``wradlib.georef.create_projstr``
- Added RADOLAN tutorial to the documentation. Together with that came new function to retrieve the RADOLAN composite grid.
- Adding section ``Release notes`` to the documentation.

**New features**

- comprehensive RADOLAN tutorial, examples, and example data: http://wradlib.bitbucket.org/tutorial_radolan_format.html
- enhanced :doc:`generated/wradlib.io.read_RADOLAN_composite` to read EX product
- :doc:`generated/wradlib.georef.get_radolan_grid`

**Deprecated features**

*None.*

**Removed functions**

*None.*


Version 0.3.0
-------------

**Highlights**

Visually most strikingly, we moved to a new style in our online documentation. 

However, the most important change introduced with this release was to remove a number of deprecated functions that will not be supported anymore (see list below). Users who want to use these functions need to fall back to ``0.2.0`` (not recommended). Accordingly, examples and documentation has been revised in order to remove all remaining usage of deprecated functions and also fix some documentation issues.

In addition, we removed three outdated tutorial (on clutter detection, convertion and rainfall accumulation) and replaced the two latter by a more concise tutorial "Converting reflectivity to rainfall" (see http://wradlib.bitbucket.org/tutorial_get_rainfall.html).

Finally, we use one "central" bibliography for literature cross-referencing now (see http://wradlib.bitbucket.org/zreferences.html). 

**New features**

- New style of online docs (http://wradlib.bitbucket.org), using sphinx_rtd_theme
- Added Tutorial http://wradlib.bitbucket.org/tutorial_get_rainfall.html
- New organisation of bibliography: http://wradlib.bitbucket.org/zreferences.html

**Deprecated features**

*None*

**Removed functions**

The following functions/classes that were marked as deprecated before have been removed with this release:

- ``wradlib.dp.fill_phidp``
- ``wradlib.dp.process_raw_phidp``
- ``wradlib.georef.polar2latlon``
- ``wradlib.georef.__pol2latlon``
- ``wradlib.georef.polar2latlonalt``
- ``wradlib.georef.polar2latlonalt_n``
- ``wradlib.georef.project``
- ``wradlib.vis.PolarPlot``
- ``wradlib.vis.polar_plot2``
- ``wradlib.vis.polar_plot``
- ``wradlib.vis.CartesianPlot``
- ``wradlib.vis.cartesian_plot``
- ``wradlib.vis.get_tick_vector``
- ``wradlib.vis.create_curvilinear_axes``
- ``wradlib.vis.rhi_plot``
- ``wradlib.vis.cg_plot``
- ``wradlib.vis.rhi_plot``


Version 0.2.0
-------------

``0.2.0`` is the first new release of wradlib after more than three years of ``0.0.1``. The changes that have accumulated during this time are too many to list them here. Let's just say that from here on, we will keep track of changes in wradlib releases more thoroughly.


Version 0.1.1
-------------

``0.1.1`` was the first experimental wradlib release. 
