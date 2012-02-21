***********************************************
Reading, transform and visualize raw radar data
***********************************************

Encoding raw radar data as provided by German Weather Service (DX-data), transform them to dBZ values and visualize the results.


Reading DX-data
---------------
First thing we mostly have to do for weather radar data processing is to encode the data from their binary raw format. The German weather service provides its radar data as so called DX-data. The zeroes of this data have to be unpacked and the data transfered into an array of 128 (range values in resolution of 1 km) by 360 (angular values in resolution of 1 degree).

The naming convention for dx-data is: raa00-dx_<location-id>-<YYMMDDHHMM>-<location-abreviation>---bin or raa00-dx_<location-id>-<YYMMDDHHMM>-<location-abreviation>---bin. For example: raa00-dx_10908-200608281420-fbg---bin raw data from radar station Feldberg (fbg, 10908) from 28.8.2006 14:20.

The encoded data are stored as so called RVP6-units, which is choosen by the German Weather Service for storage optimization.

**Singular raw data**

Suppose we want to encode the pre-decided radar-scan::

   import wradlib as wrl
   singular_data = wrl.io.readDX('p:/progress/test/raa00-dx_10908-200608281420-fbg---bin')

**Multiple raw data**

First of all we should create an empty data array with the shape of the desired dimensions. In the example the dataset shall contain 2 timesteps of 360 by 128 (46080) values::

   import numpy as np
   multiple_data = np.repeat(np.empty(46080), 2).reshape(2,360,128)
   multiple_data[0] = wrl.io.readDX('p:/progress/test/raa00-dx_10908-200608180225-fbg---bin')
   multiple_data[1] = wrl.io.readDX('p:/progress/test/raa00-dx_10908-200608180230-fbg---bin')


Transforming the RVP6-units
---------------------------
The RVP6-units can reach values between 0 and 255 and can be linear transformed to dBZ-data with an dynamic range from -32.5 to 95 dBZ::

   singular_data=wrl.trafo.rvp2dBZ(singular_data)
   multiple_data=wrl.trafo.rvp2dBZ(multiple_data)


Visualizing dBZ values
----------------------
Now we want to see the results of encoding and transfering in an polar plot.

The quick solution

>>> wrl.vis.polar_plot(singular_data)
   
typical stratiform precipitation:
   
.. image:: images/first_dBZ_strat.png

with two shielding effects, the foothills of the Alps and unfortunately a spike caused by a television tower as seen in the google-map snapshot below.

.. image:: images/google.jpg

As another image visualization:

>>> wrl.vis.polar_plot(multiple_data[0])
   
typical convective precipitation cells:
   
.. image:: images/first_dBZ_con.png

We can add little adornment like a title, units and a spectral colormap for better identifiability of values:

>>> wrl.vis.polar_plot(multiple_data[0], title = 'Reflectivity: Radarscan Feldberg 18-08-2006 02:25', unit = 'dBZ', colormap = 'spectral')

.. image:: images/dBZ_con_title.png
   
Usually we talk about rain, when reflectivities exceed 20 dBZ (corresponding to a precipitation of 0.6 mm/h), thus we set the lower end of the colormap to an value of 20 dBZ for masking the wet noise:

>>> wrl.vis.polar_plot(multiple_data[0], title = 'Reflectivity: Radarscan Feldberg 18-08-2006 02:25', unit = 'dBZ', colormap = 'spectral', vmin = 20)

.. image:: images/dBZ_con_20.png
   
And for a better comparison of the stratiform and the convective precipitation we fix the upper end of the colormap to the maximum value of the datasets:

>>> max_data = max(singular_data.max(), multiple_data[0].max())
>>> wrl.vis.polar_plot(singular_data, unit = 'dBZ', colormap = 'spectral', vmin = 20, vmax = max_data)
>>> wrl.vis.polar_plot(multiple_data[0], unit = 'dBZ', colormap = 'spectral', vmin = 20, vmax = max_data)

.. image:: images/dBZ_comp.png



All raw data is provided by DWD