******************************************************
A typical workflow for radar-based rainfall estimation
******************************************************

Raw, unprocessed reflectivity products can already provide useful visual information about the spatial distribution of rainfall fields. However, in order to use weather radar observations for quantitative studies (e.g. in hydrological modelling or for assimilation into Numerical Weather Prediction models), the data has to be carefully processed in order to account for typical errors sources such as ground echoes (clutter), attenuation of the radar signal, or uncertainties in the Z/R relationship. Moreover, it might be necessary to transfer the data from polar coordinates to Cartesian grids, or to combine observations from different radar locations in overlapping areas on a common grid (composition). And in the end, you would typically like to visualise the spatial rainfall distribution on a map. Many users also need to quantify the potential error (uncertainty) of their data-based rainfall estimation.

These are just some steps that might be necessary in order to make radar data useful in a specific quantitative application environment. All steps together are typically referred to as a radar data processing chain. *wradlib* was designed to support you in establishing your own processing chain, suited to your specific requirements. In the following, we will provide an outline of a typical processing chain, step-by-step. You might not need all steps for your own workflow, or you might need steps which are not yet included here. Consider this just as an example. We will not go into detail for each step in this section, but refer to more detailed tutorials (if available) or the corresponding entry in the library reference. Most of the steps have a corresponding *wradlib* module. In order to access the functions of wradlib, you have to import wradlib in your Python console::

>>> import wradlib

If you have trouble with that import, please head back to the :doc:`gettingstarted` section.

.. note:: The code used in this tutorial can be found in the ``wradib/examples`` folder of the wradlib distribution. The corresponding example data is stored in ``wradlib/examples/data``. 

.. warning:: Be aware that applying an algorithm for error correction does not guarantee that the error is totally removed. Error correction procedures are suceptible to errors, too. Not only might they fail to *remove* the error. They might also introduce *new* errors. The trade-off between costs (introduction of new errors) and benefits (error reduction) can turn out differently for different locations, different points in time, or different rainfall situations.

Reading the data
----------------
The binary encoding of many radar products is a major obstacle for many potential radar users. Often, decoder software is not easily available. *wradlib* support a couple of formats such as the OPERA BUFR and hdf5 implementations, NetCDF, and some formats used by the Germany Weather Service. We seek to continuously enhance the range of supported formats.

The basic data type used in *wradlib* is a multi-dimensional array, the numpy.ndarray. Such an array might e.g. represent a polar or Cartesian grid, or a series of rain gage observations. Metadata are normally managed as Python dictionaries. In order to read the content of a data file into a numpy array, you would normally use the ``wradib.io`` module. In the following example, a local PPI from the German Weather Service, a DX file, is read::

>>> data, metadata = wradlib.io.readDX("DX_sample_file")

The ``metadata`` object can be inspected via keywords. The ``data`` object contains the actual data, in this case a polar grid with 360 azimuth angles and 128 range bins.

.. seealso:: Get more info in the section :doc:`tutorial_supported_formats` and in the library reference section :doc:`io`.   


Clutter removal
---------------
Clutter are non-precipitation echos. They are caused by the radar beam hitting objects on the earth's surface (e.g. mountain or hill tops, houses, wind turbines) or in the air (e.g. airplanes, birds). These objects can potentially cause high reflectivities due large scattering cross sections. Static clutter, if not efficiently removed by Doppler filters, can cause permanent echos which could introduce severe bias in quantitative applications. Thus, an efficient identification and removal of clutter is mandatory e.g. for hydrological studies. Clutter removal can be based on static maps or dynamic filters. Normally, static clutter becomes visible more clearly in rainfall accumulation maps over periods of weeks or months. We recommend such accumulations to create static clutter maps which can in turn be used to remove the static clutter from an image and fill the resulting gaps by interpolation. In the following example, the clutter filter published by Gabella and Notarpietro ([Gabella2002]_) is applied to the single radar sweep of the above example.  

>>> clutter = wradlib.clutter.filter_gabella(data, tr1=12, n_p=6, tr2=1.1)

The resulting Boolean array ``clutter`` indicates the position of clutter. It can be used to interpolate the values at those positons from non-clutter values, as shown in the following line:

>>> data_no_clutter = wradlib.ipol.interpolate_polar(data, clutter)

It is generally recommended to remove the clutter before e.g. gridding the data because this might smear the clutter signal over multiple grid cells, and result into a decrease in identifiability.

.. seealso:: Get more info in the section :doc:`tutorial_clutter_correction` and in the library reference section :doc:`clutter`.  


Attenuation correction
----------------------
Attenuation by wet radome and by heavy rainfall can cause serious underestimation of rainfall for `C-Band and X-Band <http://www.everythingweather.com/weather-radar/bands.shtml>`_ devices. For such radar devices, situations with heavy rainfall require a correction of attenuation effects. The general approach with single-polarized radars is to use a recursive gate-by-gate approach. See Kraemer and Verworn ([Kraemer2008]_) for an introduction to this concept. Basically, the specific attenuation ``k`` of the first range gate is computed via a so-called ``k-Z`` relationship. Based on ``k``, the reflectivity of the second range gate is corrected and then used to compute the specific attenuation for the second range gate (and so on). The concept was first introduced by Hitchfeld and Bordan ([Hitschfeld1954]_). Its main drawback is its suceptibility to instable behaviour. *wradlib* provides a different implementations which address this problem. One example is the algorithm published by Kraemer and Verworn ([Kraemer2008]_):

>>> pia = wradlib.atten.correctAttenuationKraemer(data_no_clutter)
>>> data_attcorr = data_no_clutter + pia

The first line computes the path integrated attenuation ``pia`` for each radar bin. The second line line uses ``pia`` to correct the reflectivity values. Let's inspect the effect of attenuation correction for an azimuth angle of 65°:

>>> import pylab as pl
>>> pl.plot(data_attcorr[65], label="attcorr")
>>> pl.plot(data_no_clutter[65], label="no attcorr")
>>> pl.xlabel("km")
>>> pl.ylabel("dBZ")
>>> pl.legend()
>>> pl.show()

.. seealso:: Get more info in the library reference section :doc:`atten`. Here you will learn to know the algorithms available for attenuation correction and how to manipulate their behaviour by using additonal keyword arguments.   


Vertical Profile of Reflectivity
--------------------------------
*Not yet available - implementation is ongoing.*


Conversion of reflectivity into rainfall
----------------------------------------
Reflectivity (Z) and precipitation rate (R) can be related in form of a power law R=a*Z**b. The parameters a and b depend on the type of precipitation in terms of drop size distribution and water temperature. Before applying the Z-R relationship, we need to convert from dBZ to Z:

>>> R = wradlib.zr.z2r( wradlib.trafo.idecibel(data_attcorr) )

The above line uses the default parameters parameters ``a=200`` and ``b=1.6`` for the Z-R relationship. In order to compute a rainfall depth from rainfall intensity, we have to specify an integration interval in seconds. In this example, we choos fove minutes (300 s), corresponding to the sweep return interval:

>>> depth = wradlib.trafo.r2depth(R, 300)

.. seealso:: Get more info in the section :doc:`tutorial_conversion` and in the library reference sections :doc:`zr` and :doc:`trafo`. Here you will learn about the effects of the Z-R parameters ``a`` and ``b``.


Rainfall accumulation
---------------------
For many applications, accumulated rainfall depths over specific time intervals are required, e.g. hourly or daily accumulations. *wradlib* supports the corresponding datetime operations. In the following example, we will use a synthetic time series of 5 minute intervals. Just imagine we have repeated the above procedure for one day of five-minute sweeps and combined the arrays of rainfall depth in a 3-dimensional array of shape ``(number of time steps, number of azimuth angles, number of range gates)``. Now we want to ocompute hourly accumulations:

>>> import numpy as np
>>> sweep_times  = wradlib.util.from_to("2012-10-26 00:00:00", "2012-10-27 00:00:00", 300)
>>> depths_5min  = np.random.uniform(size=(len(sweep_times)-1, 360, 128))
>>> hours        = wradlib.util.from_to("2012-10-26 00:00:00", "2012-10-27 00:00:00", 3600)
>>> depths_hourly= wradlib.util.aggregate_in_time(depths_5min, sweep_times, hours, func='sum')

Check the shape and values of your resulting array for plausibility:

>>> print depths_hourly.shape
(24, 360, 128)
>>> print depths_hourly.mean().round()
6.0

.. seealso:: Get more info in the library reference section :doc:`util`.


Georeferencing and projection
-----------------------------
In order to define the horizontal and vertical position of the radar bins, we need to retrieve the corresponding 3-dimensional coordinates in terms of latitude, longitude and altitude. This information is required e.g. if the positions should be plotted on a map. It is also required for constructing `CAPPIs <http://en.wikipedia.org/wiki/Constant_altitude_plan_position_indicator>`_. The position of a radar bin in 3-dimensional space depends on the position of the radar device, the elevation angle of the radar beam, as well as the azimuth angle and the range of a bin. For the sample data used above, the posiiton of the radar device is the Feldberg in Germany (47.8744, 8.005, 1517): 

>>> radar_location = (47.8744, 8.005, 1517) # (lat, lon, alt) in decimal degree and meters
>>> elevation = 0.5 # in degree
>>> azimuths = np.arange(0,360) # in degrees
>>> ranges = np.arange(0, 128000., 1000.) # in meters
>>> lat, lon, alt = wradlib.georef.polar2latlonalt(ranges, azimuths, elevation, radar_location)

*wradlib* supports the projection of geographical coordinates (lat/lon) to a Cartesian reference system. Basically, you have to provide a string which represents the projection - based on the `proj.4 library <http://trac.osgeo.org/proj/>`_. You can `look up projection strings <http://www.remotesensing.org/geotiff/proj_list>`_, but for some projections, *wradlib* helps you to define a projection string. In the following example, the target projection is Gauss-Krueger (zone 3): 

>>> gk3 = wradlib.georef.create_projstr("gk", zone=3)
>>> x, y = wradlib.georef.project(lat, lon, gk3)

.. seealso:: Get more info in the library reference section :doc:`georef`.


Gridding
--------
*No content, yet.*


Composition of different radar observations
-------------------------------------------
*No content, yet.*


Adjustment by rain gage observations
------------------------------------
*No content, yet.*

Verification and quality control
--------------------------------
*No content, yet.*

Visualisation and mapping
-------------------------
*No content, yet.*

Data export to other applications
---------------------------------
*No content, yet.*


References
----------
.. [Gabella2002] Gabella, M. & Notarpietro, R., 2002. Ground clutter characterization and elimination in mountainous terrain.
	In Proceedings of ERAD. Delft: Copernicus GmbH, pp. 305-311. URL: http://www.copernicus.org/erad/online/erad-305.pdf
	[Accessed Oct 25, 2012].

.. [Hitschfeld1954] Hitschfeld, W. & Bordan, J., 1954. Errors Inherent in the Radar Measurement of Rainfall at Attenuating
	Wavelengths. Journal of the Atmospheric Sciences, 11(1), p.58-67. DOI: 10.1175/1520-0469(1954)011<0058:EIITRM>2.0.CO;2

.. [Kraemer2008] Kraemer, S., H. R. Verworn, 2008: Improved C-band radar data processing for real time control of
    urban drainage systems. 11th International Conference on Urban Drainage, Edinburgh, Scotland, UK, 2008. URL: http://web.sbe.hw.ac.uk/staffprofiles/bdgsa/11th_International_Conference_on_Urban_Drainage_CD/ICUD08/pdfs/105.pdf [Accessed Oct 25, 2012].


