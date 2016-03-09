******************************************************
A typical workflow for radar-based rainfall estimation
******************************************************

Raw, unprocessed reflectivity products can already provide useful visual information about the spatial distribution of rainfall fields. However, in order to use weather radar observations for quantitative studies (e.g. in hydrological modelling or for assimilation into Numerical Weather Prediction models), the data has to be carefully processed in order to account for typical errors sources such as ground echoes (clutter), attenuation of the radar signal, or uncertainties in the Z/R relationship.

Moreover, it might be necessary to transfer the data from polar coordinates to Cartesian grids, or to combine observations from different radar locations in overlapping areas on a common grid (composition). And in the end, you would typically like to visualise the spatial rainfall distribution on a map. Many users also need to quantify the potential error (uncertainty) of their data-based rainfall estimation.

These are just some steps that might be necessary in order to make radar data useful in a specific quantitative application environment. All steps together are typically referred to as a *"radar data processing chain"*. *wradlib* was designed to support you in establishing your own processing chain, suited to your specific requirements. In the following, we will provide an outline of a typical processing chain, step-by-step. You might not need all steps for your own workflow, or you might need steps which are not yet included here.

Consider this just as an example. We will not go into detail for each step in this section, but refer to more detailed tutorials (if available) or the corresponding entry in the library reference. Most of the steps have a corresponding *wradlib* module. In order to access the functions of wradlib, you have to import wradlib in your Python environment::

    import wradlib

If you have trouble with that import, please head back to the :doc:`gettingstarted` section.

.. note:: The code used in this tutorial can be found in the file ``wradib/examples/typical_workflow.py`` of the wradlib distribution. The corresponding example data is stored in ``wradlib/examples/data``.

.. warning:: Be aware that applying an algorithm for error correction does not guarantee that the error is totally removed. Error correction procedures are suceptible to errors, too. Not only might they fail to *remove* the error. They might also introduce *new* errors. The trade-off between costs (introduction of new errors) and benefits (error reduction) can turn out differently for different locations, different points in time, or different rainfall situations.


Reading the data
----------------
The binary encoding of many radar products is a major obstacle for many potential radar users. Often, decoder software is not easily available. *wradlib* supports a couple of formats such as the ODIM_H5 implementation, NetCDF, and some formats used by the German Weather Service. We seek to continuously enhance the range of supported formats.

The basic data type used in *wradlib* is a multi-dimensional array, the numpy.ndarray. Such an array might e.g. represent a polar or Cartesian grid, or a series of rain gage observations. Metadata are normally managed as Python dictionaries. In order to read the content of a data file into a numpy array, you would normally use the ``wradib.io`` module. In the following example, a local PPI from the German Weather Service, a DX file, is read::

    import pylab as pl
    data, metadata = wradlib.io.readDX("DX_sample_file")
    ax, pm = wradlib.vis.plot_ppi(data) # simple diagnostic plot
    cbar = pl.colorbar(pm, shrink=0.75)


The ``metadata`` object can be inspected via keywords. The ``data`` object contains the actual data, in this case a polar grid with 360 azimuth angles and 128 range bins.

.. seealso:: Get more info in the section :doc:`tutorial_supported_formats` and in the library reference section :doc:`io`.   


Clutter removal
---------------
Clutter are non-meteorological echos. They are caused by the radar beam hitting objects on the earth's surface (e.g. mountain or hill tops, houses, wind turbines) or in the air (e.g. airplanes, birds). These objects can potentially cause high reflectivities due large scattering cross sections. Static clutter, if not efficiently removed by Doppler filters, can cause permanent echos which could introduce severe bias in quantitative applications. Thus, an efficient identification and removal of clutter is mandatory e.g. for hydrological studies. Clutter removal can be based on static maps or dynamic filters. Normally, static clutter becomes visible more clearly in rainfall accumulation maps over periods of weeks or months. We recommend such accumulations to create static clutter maps which can in turn be used to remove the static clutter from an image and fill the resulting gaps by interpolation. In the following example, the clutter filter published by Gabella and Notarpietro :cite:`Gabella2002` is applied to the single radar sweep of the above example::

    clutter = wradlib.clutter.filter_gabella(data, tr1=12, n_p=6, tr2=1.1)
    ax, pm = wradlib.vis.plot_ppi(clutter, cmap=pl.cm.gray)
    pl.title('Clutter Map')
    cbar = pl.colorbar(pm, shrink=0.75)

The resulting Boolean array ``clutter`` indicates the position of clutter. It can be used to interpolate the values at those positons from non-clutter values, as shown in the following line::

    data_no_clutter = wradlib.ipol.interpolate_polar(data, clutter)

It is generally recommended to remove the clutter before e.g. gridding the data. Otherwise the clutter signal might be "smeared" over multiple grid cells, resulting into a decrease in detectability.

.. seealso:: Get more info in the library reference section :doc:`clutter`.


Attenuation correction
----------------------
Attenuation by wet radome and by heavy rainfall can cause serious underestimation of rainfall for `C-Band and X-Band <http://www.everythingweather.com/weather-radar/bands.shtml>`_ devices. For such radar devices, situations with heavy rainfall require a correction of attenuation effects. The general approach with single-polarized radars is to use a recursive gate-by-gate approach. See Kraemer and Verworn :cite:`Kraemer2008` for an introduction to this concept. Basically, the specific attenuation ``k`` of the first range gate is computed via a so-called ``k-Z`` relationship. Based on ``k``, the reflectivity of the second range gate is corrected and then used to compute the specific attenuation for the second range gate (and so on). The concept was first introduced by Hitchfeld and Bordan :cite:`Hitschfeld1954`. Its main drawback is its suceptibility to instable behaviour. *wradlib* provides different implementations which address this problem.

One example is the algorithm published by Kraemer and Verworn :cite:`Kraemer2008`::

    pia = wradlib.atten.correctAttenuationKraemer(data_no_clutter)
    data_attcorr = data_no_clutter + pia

The first line computes the path integrated attenuation ``pia`` for each radar bin. The second line uses ``pia`` to correct the reflectivity values. Let's inspect the effect of attenuation correction for an azimuth angle of 65 deg::

    import pylab as pl
    pl.plot(data_attcorr[65], label="attcorr")
    pl.plot(data_no_clutter[65], label="no attcorr")
    pl.xlabel("km")
    pl.ylabel("dBZ")
    pl.legend()

.. seealso:: Get more info in the library reference section :doc:`atten`. There you will learn to know the algorithms available for attenuation correction and how to manipulate their behaviour by using additonal keyword arguments.   


Vertical Profile of Reflectivity
--------------------------------
Precipitation is 3-dimensional in space. The vertical distribution of precipitation (and thus reflectivity) is typically non-uniform. As the height of the radar beam increases with the distance from the radar location (beam elevation, earth curvature), one sweep samples from different heights. The effects of the non-uniform VPR and the different sampling heights need to be accounted for if we are interested in the precipiation near the ground or in defined altitudes.

.. seealso:: Get more info in the library reference section :doc:`vpr`. There you will learn how to reference polar volume data, to create CAPPIs and Pseudo CAPPIs, to inspect vertical profiles of reflectivity (UNDER DEVELOPMENT), and to use these for correction (UNDER DEVELOPMENT).   



Conversion of reflectivity into rainfall
----------------------------------------
Reflectivity (Z) and precipitation rate (R) can be related in form of a power law ``R=a*Z**b``. The parameters ``a`` and ``b`` depend on the type of precipitation in terms of drop size distribution and water temperature. Before applying the Z-R relationship, we need to convert from dBZ to Z::

    R = wradlib.zr.z2r( wradlib.trafo.idecibel(data_attcorr) )

The above line uses the default parameters parameters ``a=200`` and ``b=1.6`` for the Z-R relationship. In order to compute a rainfall depth from rainfall intensity, we have to specify an integration interval in seconds. In this example, we chose five minutes (300 s), corresponding to the sweep return interval::

    depth = wradlib.trafo.r2depth(R, 300)

.. seealso:: Get more info in the section :doc:`tutorial_get_rainfall` and in the library reference sections :doc:`zr` and :doc:`trafo`. Here you will learn about the effects of the Z-R parameters ``a`` and ``b``.


Rainfall accumulation
---------------------
For many applications, accumulated rainfall depths over specific time intervals are required, e.g. hourly or daily accumulations. *wradlib* supports the corresponding datetime operations. In the following example, we will use a synthetic time series of 5 minute intervals. Just imagine we have repeated the above procedure for one day of five-minute sweeps and combined the arrays of rainfall depth in a 3-dimensional array of shape ``(number of time steps, number of azimuth angles, number of range gates)``.

Now we want to compute hourly accumulations::

    import numpy as np
    sweep_times  = wradlib.util.from_to("2012-10-26 00:00:00", "2012-10-27 00:00:00", 300)
    depths_5min  = np.random.uniform(size=(len(sweep_times)-1, 360, 128))
    hours        = wradlib.util.from_to("2012-10-26 00:00:00", "2012-10-27 00:00:00", 3600)
    depths_hourly= wradlib.util.aggregate_in_time(depths_5min, sweep_times, hours, func='sum')

Check the shape and values of your resulting array for plausibility:

>>> print depths_hourly.shape
(24, 360, 128)
>>> print depths_hourly.mean().round()
6.0

.. seealso:: Get more info in the library reference section :doc:`util`.


Georeferencing and projection
-----------------------------
In order to define the horizontal and vertical position of the radar bins, we need to retrieve the corresponding 3-dimensional coordinates in terms of longitude, latitude and altitude. This information is required e.g. if the positions should be plotted on a map. It is also required for constructing `CAPPIs <https://en.wikipedia.org/wiki/Constant_altitude_plan_position_indicator>`_. The position of a radar bin in 3-dimensional space depends on the position of the radar device, the elevation and azimuth angle of the radar beam, the range of the bin, and the assumed influence of atmospheric refraction on the beam propagation. For the sample data used above, the position of the radar device is the Feldberg in Germany (8.005, 47.8744, 1517)::

    import numpy as np
    radar_location = (8.005, 47.8744, 1517) # (lon, lat, alt) in decimal degree and meters
    elevation = 0.5 # in degree
    azimuths = np.arange(0,360) # in degrees
    ranges = np.arange(0, 128000., 1000.) # in meters
    polargrid = np.meshgrid(ranges, azimuths)
    lon, lat, alt = wradlib.georef.polar2lonlatalt_n(polargrid[0], polargrid[1],
                                                     elevation, radar_location)

*wradlib* supports the projection of geographical coordinates (lon/lat) to a Cartesian reference system. It uses GDAL/OSR Spatial References Objects as function parameters. Basically, you have to create the OSR-object by using GDAL-capabilities or one of the provided helper functions. We recommend the creation using `EPSG numbers <https://epsg.io/>`_::

    # Gauss Krueger Zone 3, EPSG-Number 31467
    gk3 = wradlib.georef.epsg_to_osr(31467)
    x, y = wradlib.georef.reproject(lon, lat, projection_target=gk3)

Second, you can provide a string which represents the projection - based on the `PROJ.4 library <https://trac.osgeo.org/proj/>`_. You can `look up projection strings <http://www.spatialreference.org/>`_, but for some projections, *wradlib* helps you to define a projection string. In the following example, the target projection is azimuthal equidistant::

    ae = wradlib.georef.create_osr("aeqd", lon_0=radar_location[0], lat_0=radar_location[1])
    x, y = wradlib.georef.reproject(lon, lat, projection_target=ae)

.. seealso:: Get more info in the library reference section :doc:`georef`.


Gridding
--------
Assume you would like to transfer the rainfall intensity from the above example (`Conversion of reflectivity into rainfall`_) from polar coordinates to a Cartesian grid, or to an arbitrary set of irregular points in space (e.g. centroids of sub-catchments). You already retrieved the Cartesian coordinates of the radar bins in the previous section (`Georeferencing and projection`_). Now you only need to define the target coordinates (e.g. a grid) and apply the ``togrid`` function of the ``wradlib.comp`` module. In this example, we want our grid only to represent the South-West sector of our radar circle on a 100 x 100 grid. First, we define the target grid coordinates (these must be an array of 100x100 rows with one coordinate pair each)::

    xgrid = np.linspace(x.min(), x.mean(), 100)
    ygrid = np.linspace(y.min(), y.mean(), 100)
    grid_xy = np.meshgrid(xgrid, ygrid)
    grid_xy = np.vstack((grid_xy[0].ravel(), grid_xy[1].ravel())).transpose()

Now we transfer the polar data to the grid and mask out invalid values for plotting (values outside the radar circle receive NaN)::

    xy=np.concatenate([x.ravel()[:,None],y.ravel()[:,None]], axis=1)
    gridded = wradlib.comp.togrid(xy, grid_xy, 128000., np.array([x.mean(), y.mean()]), data.ravel(), wradlib.ipol.Nearest)
    gridded = np.ma.masked_invalid(gridded).reshape((len(xgrid), len(ygrid)))

    fig = pl.figure(figsize=(10,8))
    ax = pl.subplot(111, aspect="equal")
    pm = pl.pcolormesh(xgrid, ygrid, gridded)
    pl.colorbar(pm, shrink=0.75)
    pl.xlabel("Easting (m)")
    pl.ylabel("Northing (m)")

.. seealso:: Get more info about the function :doc:`generated/wradlib.comp.togrid`.


Adjustment by rain gage observations
------------------------------------
Adjustment normally refers to using rain gage observations on the ground to correct for errors in the radar-based rainfall estimatin. Goudenhoofdt and Delobbe :cite:`Goudenhoofdt2009` provide an excellent overview of adjustment procedures. A typical approach is to quantify the error of the radar-based rainfall estimate *at* the rain gage locations, assuming the rain gage observation to be accurate. The error can be assumed to be additive, multiplicative, or a mixture of both. Most approaches assume the error to be heterogeneous in space. Hence, the error at the rain gage locations will be interpolated to the radar bin (or grid) locations and then used to adjust (correct) the raw radar rainfall estimates.

In the following example, we will use an illustrative one-dimensional example with synthetic data (just imagine radar rainfall estimates and rain gage observations along one radar beam). 

First, we create the synthetic "true" rainfall (``truth``)::

    import numpy as np
    radar_coords = np.arange(0,101)
    truth = np.abs(1.5+np.sin(0.075*radar_coords)) + np.random.uniform(-0.1,0.1,len(radar_coords))

The radar rainfall estimate ``radar`` is then computed by imprinting a multiplicative ``error`` on ``truth`` and adding some noise::

    error = 0.75 + 0.015*radar_coords
    radar = error * truth + np.random.uniform(-0.1,0.1,len(radar_coords))

Synthetic gage observations ``obs`` are then created by selecting arbitrary "true" values::

    obs_coords = np.array([5,10,15,20,30,45,65,70,77,90])
    obs = truth[obs_coords]

Now we adjust the ``radar`` rainfall estimate by using the gage observations. First, you create an "adjustment object" from the approach you
want to use for adjustment. After that, you can call the object with the actual data that is to be adjusted. Here, we use a multiplicative error model with spatially heterogenous error (see :doc:`generated/wradlib.adjust.AdjustMultiply`)::

    adjuster = wradlib.adjust.AdjustMultiply(obs_coords, radar_coords, nnear_raws=3)
    adjusted = adjuster(obs, radar)

Let's compare the ``truth``, the ``radar`` rainfall estimate and the ``adjusted`` product::

    import pylab as pl
    pl.plot(radar_coords, truth, 'k-', label="True rainfall", linewidth=2.)
    pl.xlabel("Distance (km)")
    pl.ylabel("Rainfall intensity (mm/h)")
    pl.plot(radar_coords, radar, 'k-', label="Raw radar rainfall", linewidth=2., linestyle="dashed")
    pl.plot(obs_coords, obs, 'o', label="Gage observation", markersize=10.0, markerfacecolor="grey")
    pl.plot(radar_coords, adjusted, '-', color="green", label="Multiplicative adjustment", linewidth=2., )
    pl.legend(prop={'size':12})

.. seealso:: Get more info in the library reference section :doc:`adjust`. There, you will also learn how to use the built-in *cross-validation* in order to evaluate the performance of the adjustment approach.


Verification and quality control
--------------------------------
Typically, radar-based precipitation estimation and the effectiveness of the underlying correction and adjustment methods are verified by comparing the results against rain gage observations on the ground. wradlib.verify provides procedures not only to extract the radar values at specific gauge locations, but also a set of error metrics which are computed from gage observations and the corresponding radar-based precipitation estimates (including standard metrics such as RMSE, mean error, Nash-Sutcliffe Efficiency). In the following, we will illustrate the usage of error metrics by comparing the "true" rainfall against the raw and adjusted radar rainfall estimates from the above example::

    raw_error  = wradlib.verify.ErrorMetrics(truth, radar)
    adj_error  = wradlib.verify.ErrorMetrics(truth, adjusted)

Error metrics can be reported e.g. as follows::

    raw_error.report()
    adj_error.report()

.. seealso:: Get more info in the library reference section :doc:`verify`.


Visualisation and mapping
-------------------------
In the above sections `Reading the data`_, `Clutter removal`_, and `Gridding`_ you already saw examples of the wradlib's plotting capabilities.

.. seealso:: Get more info in the library reference section :doc:`vis`.


Data export to other applications
---------------------------------
Once you created a dataset which meets your requirements, you might want to export it to other applications or archives. *wradlib* does not favour or support a specific output format. Basically, you have all the freedom of choice offered by Python and its packages in order to export your data. Arrays can be stored as text or binary files by using numpy functions. You can use the package `NetCDF4 <https://unidata.github.io/netcdf4-python/>`_ to write NetCDF files, and the packages `h5py <http://www.h5py.org/>`_ or `PyTables <http://www.pytables.org>`_ to write hdf5 files.
At a later stage of development, *wradlib* might support a standardized data export by using the OPERA's ODIM_H5 data model 
(see :doc:`tutorial_supported_formats`). Of course, you can also export data as images. See :doc:`vis` for some options.

Export your data array as a text file::

    np.savetxt("mydata.txt", data)

Or as a gzip-compressed text file::

    np.savetxt("mydata.gz", data)

Or as a NetCDF file::

    import netCDF4
    rootgrp = netCDF4.Dataset('test.nc', 'w', format='NETCDF4')
    sweep_xy = rootgrp.createGroup('sweep_xy')
    dim_azimuth = sweep_xy.createDimension('azimuth', None)
    dim_range = sweep_xy.createDimension('range', None)
    azimuths_var = sweep_xy.createVariable('azimuths','i4',('azimuth',))
    ranges_var = sweep_xy.createVariable('ranges','f4',('range',))
    dBZ_var = sweep_xy.createVariable('dBZ','f4',('azimuth','range',))
    azimuths_var[:] = np.arange(0,360)
    ranges_var[:] = np.arange(0, 128000., 1000.)
    dBZ_var[:] = data

You can easily add metadata to the NetCDF file on different group levels::

    rootgrp.bandwith = "C-Band"
    sweep_xy.datetime = "2012-11-02 10:15:00"
    rootgrp.close()

.. note:: An example for hdf5 export will follow.
