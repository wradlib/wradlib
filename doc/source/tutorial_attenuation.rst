**********************
Attenuation correction
**********************

Rainfall-induced attenuation is a major source of underestimation for radar-based precipitation estimation at C-band and X-band. Unconstrained forward gate-by-gate correction is known to be inherently unstable and thus not suited for unsupervised quality control procedures. Ideally, reference measurements (e.g. from microwave links) should be used to constrain gate-by-gate procedures. However, such attenuation references are ususally not available. *wradlib* provides a pragmatic approach to constrain gate-by-gate correction procedures, inspired by the work of [Kraemer2008]_. It turned out that these procedures can effectively reduce the error introduced by attenuation, and, at the same time, minimize instability issues [Jacobi2015]_.

The example event
=================

Let's have a look at the situation in South-West Germany on June 2nd, 2008, at 16:55 UTC, as observed by the DWD C-band radar on mount Feldberg.
The data can be read by the following lines and then visualized by ``wradlib.vis.plot_ppi`` (see image source code below)::

   filename = "../../examples/data/raa00-dx_10908-0806021655-fbg---bin"
   data, attrs = wradlib.io.readDX(filename)

.. plot::

   import matplotlib.pyplot as plt
   import wradlib
   filename = "../../examples/data/raa00-dx_10908-0806021655-fbg---bin"
   data, attrs = wradlib.io.readDX(filename)
   ax, cf = wradlib.vis.plot_ppi(data, cmap="spectral")
   plt.xlabel("Easting from radar (km)")
   plt.ylabel("Northing from radar (km)")
   plt.title("Radar Feldberg, 2008-06-02 16:55 UTC")
   cb = plt.colorbar(cf, shrink=0.8)
   cb.set_label("dBZ")
   plt.plot([0,105.6],[0,73.4],"-", color="white", lw=2)
   plt.xlim(-128,128)
   plt.ylim(-128,128)
   plt.grid(color="grey")

   plt.show()

We see a set of convective cells with high rainfall intensity in the NE-sector of the Feldberg radar. Let us examine the reflectivity profile along three beams which at azimuths 53-55 degree (as marked by the white line in the PPI above).

.. plot::

   import matplotlib.pyplot as plt
   import wradlib
   filename = "../../examples/data/raa00-dx_10908-0806021655-fbg---bin"
   data, attrs = wradlib.io.readDX(filename)
   mybeams = slice(53,56)
   labelsize=13
   fig = plt.figure(figsize=(10,3))
   ax = fig.add_subplot(111)
   plt.plot(data[53], label="53 deg")
   plt.plot(data[54], label="54 deg")
   plt.plot(data[55], label="55 deg")
   plt.grid()
   plt.text(0.99, 0.88, "Reflectivity along beams", horizontalalignment='right', transform = ax.transAxes, fontsize="large")
   plt.xlabel("range (km)", fontsize="large")
   plt.ylabel("Reflectivity (dBZ)", fontsize="large")
   plt.legend(loc="upper left")
   ax.tick_params(axis='x', labelsize=labelsize)
   ax.tick_params(axis='y', labelsize=labelsize)
   plt.xlim(0,128)
   plt.show()

We will now examine the behaviour of different attenuation correction procedures. In the end, we will show a `Comparison of all methods`_ 


Hitschfeld and Bordan
=====================

First, we examine the behaviour of the "classical" unconstrained forward correction which is typically referred to Hitchfeld and Bordan ([Hitschfeld1954]_), although Hitschfeld and Bordan themselves rejected this approach. The Path Integrated Attenuation (PIA) according to this approach can be obtained as follows::

   pia_hibo = wradlib.atten.correctAttenuationHB(data, coefficients = dict(a=8.e-5, b=0.731, l=1.0), mode="warn", thrs=59.)

In the coefficients dictionary, we can pass the power law parameters of the A(Z) relation as well as the gate length (in km). If we pass "warn" as the mode argument, we will obtain a warning log in case the corrected reflectivity exceeds the value of argument ``thrs`` (dBZ).

Plotting the result below the reflectivity profile, we obtain the following figure.  

.. plot::

	import matplotlib.pyplot as plt
	import wradlib
	filename = "../../examples/data/raa00-dx_10908-0806021655-fbg---bin"
	data, attrs = wradlib.io.readDX(filename)
	pia_hibo = wradlib.atten.correctAttenuationHB(data, coefficients = dict(a=8.e-5, b=0.731, l=1.0), mode="warn", thrs=59.)
	
	fig = plt.figure(figsize=(10,6))

	mybeams = slice(53,56)
	labelsize=13

	ax = fig.add_subplot(211)
	plt.plot(data[53], label="53 deg")
	plt.plot(data[54], label="54 deg")
	plt.plot(data[55], label="55 deg")
	plt.grid()
	plt.text(0.99, 0.88, "Reflectivity along beams", horizontalalignment='right', transform = ax.transAxes, fontsize="large")
	plt.ylabel("Reflectivity (dBZ)", fontsize="large")
	plt.legend(loc="upper left")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)

	ax = fig.add_subplot(212)
	plt.plot(pia_hibo[mybeams].T)
	plt.grid()
	plt.ylim(0,30)
	plt.xlabel("range (km)", fontsize="large")
	plt.ylabel("PIA (dB)", fontsize="large")
	plt.text(0.01, 0.88, "PIA according to Hitchfeld and Bordan", transform = ax.transAxes, fontsize="large")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)
	
	plt.show()

Apparently, slight differences in the reflectivity profile can cause a dramatic change in the behaviour. While at 54 and 55 degrees, the retrieval of PIA appears to be fairly stable, the profile of PIA for 53 degree demonstrates a case of instability.  


Harrison
========

Harrison et al. [Harrison2000]_ suggested to simply cap PIA in case it would cause a correction of rainfall intensity by more than a factor of two. Depending on the parameters of the Z(R) relationship, that would correpond to PIA values between 4 and 5 dB (4.8 dB if we assume exponent b=1.6). 

One way to implement this approach would be the following::

   pia_harrison = wradlib.atten.correctAttenuationHB(data, coefficients = dict(a=4.57e-5, b=0.731, l=1.0), mode="warn", thrs=59.)
   pia_harrison[pia_harrison4.8] = 4.8
   
And the results would look like this:

.. plot::

	import matplotlib.pyplot as plt
	import wradlib
	filename = "../../examples/data/raa00-dx_10908-0806021655-fbg---bin"
	data, attrs = wradlib.io.readDX(filename)
	pia_harrison = wradlib.atten.correctAttenuationHB(data, coefficients = dict(a=4.57e-5, b=0.731, l=1.0), mode="warn", thrs=59.)
	pia_harrison[pia_harrison>4.8] = 4.8
	
	fig = plt.figure(figsize=(10,6))

	mybeams = slice(53,56)
	labelsize=13

	ax = fig.add_subplot(211)
	plt.plot(data[53], label="53 deg")
	plt.plot(data[54], label="54 deg")
	plt.plot(data[55], label="55 deg")
	plt.grid()
	plt.text(0.99, 0.88, "Reflectivity along beams", horizontalalignment='right', transform = ax.transAxes, fontsize="large")
	plt.ylabel("Reflectivity (dBZ)", fontsize="large")
	plt.legend(loc="upper left")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)

	ax = fig.add_subplot(212)
	plt.plot(pia_harrison[mybeams].T)
	plt.grid()
	plt.ylim(0,30)
	plt.xlabel("range (km)", fontsize="large")
	plt.ylabel("PIA (dB)", fontsize="large")
	plt.text(0.01, 0.88, "PIA according to Harrison", transform = ax.transAxes, fontsize="large")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)
	
	plt.show()

	
Kraemer
=======

[Kraemer2008]_ suggested to iteratively determine the power law parameters of the A(Z). In particular, the power law coefficient is interatively decreased until the attenuation correction does not lead to reflectivity values above a given threshold (Kraemer suggested 59 dBZ). Using wradlib, this would be called by using the function :doc:`generated/wradlib.atten.correctAttenuationConstrained2` with a specific ``constraints`` argument::

   pia_kraemer = wradlib.atten.correctAttenuationConstrained2(
					data,
					a_max=1.67e-4, a_min=2.33e-5, n_a=100, 
					b_max=0.7, b_min=0.65, n_b=6, l=1.,
					constraints=[wrl.atten.constraint_dBZ], constraint_args=[[59.0]])
					
In brief, this call specifies ranges of the power parameters a and b of the A(Z) relation. Beginning from the maximum values (``a_max`` and ``b_max``), the function searches for values of ``a`` and ``b`` so that the corrected reflectivity will not exceed the dBZ constraint of 59 dBZ. Compared to the previous results, the corresponding profiles of PIA look like this:   

.. plot::

	import matplotlib.pyplot as plt
	import wradlib
	filename = "../../examples/data/raa00-dx_10908-0806021655-fbg---bin"
	data, attrs = wradlib.io.readDX(filename)
	pia_kraemer = wradlib.atten.correctAttenuationConstrained2(data, a_max=1.67e-4, a_min=2.33e-5, n_a=100, b_max=0.7, b_min=0.65, n_b=6, l=1.,  
	              constraints=[wradlib.atten.constraint_dBZ], constraint_args=[[59.0]])
	
	fig = plt.figure(figsize=(10,6))

	mybeams = slice(53,56)
	labelsize=13

	ax = fig.add_subplot(211)
	plt.plot(data[53], label="53 deg")
	plt.plot(data[54], label="54 deg")
	plt.plot(data[55], label="55 deg")
	plt.grid()
	plt.text(0.99, 0.88, "Reflectivity along beams", horizontalalignment='right', transform = ax.transAxes, fontsize="large")
	plt.ylabel("Reflectivity (dBZ)", fontsize="large")
	plt.legend(loc="upper left")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)

	ax = fig.add_subplot(212)
	plt.plot(pia_kraemer[mybeams].T)
	plt.grid()
	plt.ylim(0,30)
	plt.xlabel("range (km)", fontsize="large")
	plt.ylabel("PIA (dB)", fontsize="large")
	plt.text(0.01, 0.88, "PIA according to Kraemer", transform = ax.transAxes, fontsize="large")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)
	
	plt.show()


Modified Kraemer
================

The function :doc:`generated/wradlib.atten.correctAttenuationConstrained2` allows us to pass any kind of constraint function or lists of constraint functions via the argument ``constraints``. The arguments of these functions are passed via a nested list as argument ``constraint_args``. For example, [Jacobi2015]_ suggested to constrain *both* the corrected reflectivity (by a maximum of 59 dBZ) *and* the resulting path-intgrated attenuation PIA (by a maximun of 20 dB):: 

   pia_mKraemer = wradlib.atten.correctAttenuationConstrained2(
					data,
					a_max=1.67e-4, a_min=2.33e-5, n_a=100, 
					b_max=0.7, b_min=0.65, n_b=6, l=1.,
					constraints=[wradlib.atten.constraint_dBZ, wradlib.atten.constraint_pia], constraint_args=[[59.0],[20.0]])


.. plot::

	import matplotlib.pyplot as plt
	import wradlib
	filename = "../../examples/data/raa00-dx_10908-0806021655-fbg---bin"
	data, attrs = wradlib.io.readDX(filename)
	pia_mKraemer = wradlib.atten.correctAttenuationConstrained2(data, a_max=1.67e-4, a_min=2.33e-5, n_a=100, b_max=0.7, b_min=0.65, n_b=6, l=1.,  
	              constraints=[wradlib.atten.constraint_dBZ,wradlib.atten.constraint_pia], constraint_args=[[59.0],[20.0]])
	
	fig = plt.figure(figsize=(10,6))

	mybeams = slice(53,56)
	labelsize=13

	ax = fig.add_subplot(211)
	plt.plot(data[53], label="53 deg")
	plt.plot(data[54], label="54 deg")
	plt.plot(data[55], label="55 deg")
	plt.grid()
	plt.text(0.99, 0.88, "Reflectivity along beams", horizontalalignment='right', transform = ax.transAxes, fontsize="large")
	plt.ylabel("Reflectivity (dBZ)", fontsize="large")
	plt.legend(loc="upper left")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)

	ax = fig.add_subplot(212)
	plt.plot(pia_mKraemer[mybeams].T)
	plt.grid()
	plt.ylim(0,30)
	plt.xlabel("range (km)", fontsize="large")
	plt.ylabel("PIA (dB)", fontsize="large")
	plt.text(0.01, 0.88, "PIA according to modified Kraemer", transform = ax.transAxes, fontsize="large")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)
	
	plt.show()

	
Comparison of all methods
=========================

Plotting all of the above methods (`Hitschfeld and Bordan`_, `Harrison`_, `Kraemer`_, `Modified Kraemer`_) allows for a better comparison of their behaviour. Please refer to [Jacobi2015]_ for an in-depth discussion of this example.

.. plot::

	import matplotlib.pyplot as plt
	import wradlib
	filename = "../../examples/data/raa00-dx_10908-0806021655-fbg---bin"
	data, attrs = wradlib.io.readDX(filename)
	pia_hibo = wradlib.atten.correctAttenuationHB(data, coefficients = dict(a=8.e-5, b=0.731, l=1.0), mode="warn", thrs=59.)
	pia_harrison = wradlib.atten.correctAttenuationHB(data, coefficients = dict(a=4.57e-5, b=0.731, l=1.0), mode="warn", thrs=59.)
	pia_harrison[pia_harrison>4.8] = 4.8
	pia_kraemer = wradlib.atten.correctAttenuationConstrained2(data, a_max=1.67e-4, a_min=2.33e-5, n_a=100, b_max=0.7, b_min=0.65, n_b=6, l=1.,  
	              constraints=[wradlib.atten.constraint_dBZ], constraint_args=[[59.0]])
	pia_mKraemer = wradlib.atten.correctAttenuationConstrained2(data, a_max=1.67e-4, a_min=2.33e-5, n_a=100, b_max=0.7, b_min=0.65, n_b=6, l=1.,  
	              constraints=[wradlib.atten.constraint_dBZ,wradlib.atten.constraint_pia], constraint_args=[[59.0],[20.0]])
	
	fig = plt.figure(figsize=(10,12))

	mybeams = slice(53,56)
	labelsize=13

	ax = fig.add_subplot(511)
	plt.plot(data[53], label="53 deg")
	plt.plot(data[54], label="54 deg")
	plt.plot(data[55], label="55 deg")
	plt.grid()
	plt.text(0.99, 0.88, "Reflectivity along beams", horizontalalignment='right', transform = ax.transAxes, fontsize="large")
	plt.ylabel("Reflectivity (dBZ)", fontsize="large")
	plt.legend(loc="upper left")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)

	ax = fig.add_subplot(512)
	plt.plot(pia_hibo[mybeams].T)
	plt.grid()
	plt.ylim(0,30)
	plt.ylabel("PIA (dB)", fontsize="large")
	plt.text(0.01, 0.88, "PIA according to Hitschfeld and Bordan", transform = ax.transAxes, fontsize="large")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)
	
	ax = fig.add_subplot(513)
	plt.plot(pia_harrison[mybeams].T)
	plt.grid()
	plt.ylim(0,30)
	plt.ylabel("PIA (dB)", fontsize="large")
	plt.text(0.01, 0.88, "PIA according to Harrison", transform = ax.transAxes, fontsize="large")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)
	
	ax = fig.add_subplot(514)
	plt.plot(pia_kraemer[mybeams].T)
	plt.grid()
	plt.ylim(0,30)
	plt.ylabel("PIA (dB)", fontsize="large")
	plt.text(0.01, 0.88, "PIA according to Kraemer", transform = ax.transAxes, fontsize="large")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)

	ax = fig.add_subplot(515)
	plt.plot(pia_mKraemer[mybeams].T)
	plt.grid()
	plt.ylim(0,30)
	plt.xlabel("range (km)", fontsize="large")
	plt.ylabel("PIA (dB)", fontsize="large")
	plt.text(0.01, 0.88, "PIA according to modified Kraemer", transform = ax.transAxes, fontsize="large")
	ax.tick_params(axis='x', labelsize=labelsize)
	ax.tick_params(axis='y', labelsize=labelsize)
	plt.xlim(0,128)
	
	plt.show()




References
==========

.. [Harrison2000] Harrison, D. L., Driscoll, S. J., and Kitchen, M. (2000) Improving precipitation estimates from weather radar using quality control and correction techniques. Meteorol. Appl. 6:135-144.

.. [Hitschfeld1954] Hitschfeld, W. and Bordan, J. (1954) Errors inherent in the radar measurement of rainfall at attenuating wavelengths. J. Meteor. 11:58-67.

.. [Jacobi2015] Jacobi, S., and M. Heistermann, 2015: Benchmarking attenuation correction procedures for six years of single-polarised C-band weather radar observations in South-West Germany. Submitted to *Nat. Haz.*

.. [Kraemer2008] Kraemer, S., H. R. Verworn, 2008: Improved C-band radar data processing for real time control of
    urban drainage systems. 11th International Conference on Urban Drainage, Edinburgh, Scotland, UK, 2008. URL: http://web.sbe.hw.ac.uk/staffprofiles/bdgsa/11th_International_Conference_on_Urban_Drainage_CD/ICUD08/pdfs/105.pdf
	
