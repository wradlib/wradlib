Release Notes
=============

Please note that as long as wradlib releases are within the ``0.y.z`` series, the API cannot be considered stable. We will try to avoid sudden API breaks via deprecation warnings. All wradlib releases come without any warranty. Release notes might be incomplete. See `here <https://bitbucket.org/wradlib/wradlib/commits/all>`_ for a complete record of changes. 

You can install the latest wradlib release from PyPI via ``$ pip install wradlib`` or specific version via ``$ pip install wradlib==x.y.z``.


Bleeding edge
-------------

**Highlights**

*Nothing new, yet.*

**New features**

*Nothing new, yet.*

**Deprecated features**

*None.*

**Removed functions**

*None.*


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
