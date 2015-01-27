***********************************
Converting reflectivity to rainfall
***********************************

Reflectivity (Z) and precipitation rate (R) can be related in form of a power law Z=a*R**b. The parameters *a* and *b* depend on the type of precipitation (i.e. drop size distribution and water temperature). *wradlib* provides a couple of functions that could be useful in this context. The following example demonstrates the steps to convert from the common unit *dBZ* (decibel of the reflectivity factor *Z*) to rainfall intensity (in the unit of mm/h):

>>> import wradlib
>>> import numpy as np

This is an array of typical reflectivity values (**unit: dBZ**)

>>> dBZ = np.array([20., 30., 40., 45., 50., 55.])
>>> print dBZ
[ 20.  30.  40.  45.  50.  55.]

Convert to reflectivity factor Z (**unit: mm^6/m^3**)

>>> Z = wradlib.trafo.idecibel(dBZ)
>>> print Z
[  1.00000000e+02   1.00000000e+03   1.00000000e+04   3.16227766e+04
   1.00000000e+05   3.16227766e+05]

Convert to rainfall intensity (**unit: mm/h**) using the Marshall-Palmer Z(R) parameters

>>> R = wradlib.zr.z2rz2r(z, a=200., b=1.6)
>>> print np.round(R, 2)
[  0.65   2.73  11.53  23.68  48.62  99.85]

Convert to rainfall depth (**unit: mm**) assuming a rainfall duration of five minutes (i.e. 300 seconds)
 
>>> depth = wradlib.trafo.r2depth(R, 300)
>>> print np.round(depth, 2)
[ 0.05  0.23  0.96  1.97  4.05  8.32]


An example with real radar data
-------------------------------

The following example is based on observations of the DWD C-band radar on mount Feldberg (SW-Germany).
The figure shows a 15 minute accumulation of rainfall which was produced from three consecutive radar
scans at 5 minute intervals between 17:30 and 17:45 on June 8, 2008.

The radar data are read using :doc:`generated/wradlib.io.readDX` function which returns an array of dBZ values and a metadata dictionary
(see also :doc:`tutorial_reading_dx`). The conversion is carried out the same way as in the example above. The plot is produced using
the function :doc:`generated/wradlib.vis.plot_ppi`. See full source code below for download. 
 

.. plot::

	import matplotlib.pyplot as plt
	import numpy as np
	import wradlib
	def read_data(dtimes):
		"""Read raw data from multiple time steps <dtimes>
		"""
		data  = np.empty((len(dtimes),360,128))
		for i, dtime in enumerate(dtimes):
			f = "../../examples/data/raa00-dx_10908-%s-fbg---bin" % dtime
			data[i], attrs = wradlib.io.readDX(f)
		return data
	# read data from radar Feldberg for three consecutive 5 minute time steps
	dtimes = ["0806021735","0806021740","0806021745"]
	dBZ = read_data(dtimes)
	Z = wradlib.trafo.idecibel(dBZ)
	R = wradlib.zr.z2r(Z, a=200., b=1.6)
	depth = wradlib.trafo.r2depth(R, 300)
	# accumulate 15 minute rainfall depth over all three 5 minute intervals
	accum = np.sum(depth, axis=0)
	# plot PPI of 15 minute rainfall depth
	ax, cf = wradlib.vis.plot_ppi(accum, cmap="spectral")
	plt.xlabel("Easting from radar (km)")
	plt.ylabel("Northing from radar (km)")
	plt.title("Radar Feldberg\n15 min. rainfall depth, 2008-06-02 17:30-17:45 UTC")
	cb = plt.colorbar(cf, shrink=0.8)
	cb.set_label("mm")
	plt.xlim(-128,128)
	plt.ylim(-128,128)
	plt.grid(color="grey")

	plt.show()  