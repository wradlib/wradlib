.. wradlib documentation master file, created by
   sphinx-quickstart on Wed Oct 26 13:48:08 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

wradlib: An Open Source Library for Weather Radar Data Processing
=================================================================

The *wradlib* project has been initiated in order facilitate the use of weather radar data as well as to provide a common platform for research on new algorithms. *wradlib* is an open source library which is well documented and easy to use. It is written in the free programming language `Python <http://www.python.org>`_.

.. note:: Please cite wradlib as *Heistermann, M., Jacobi, S., and Pfaff, T.: Technical Note: An open source library for processing weather radar data (wradlib), Hydrol. Earth Syst. Sci. Discuss., 9, 12333-12356,* doi:`10.5194/hessd-9-12333-2012, 2012 <http://www.hydrol-earth-syst-sci-discuss.net/9/12333/2012/hessd-9-12333-2012.html>`_ 

.. image:: images/old_radarpic.png

Weather radar data is potentially useful in meteorology, hydrology and risk management. Its ability to provide information on precipitation 
with high spatio-temporal resolution over large areas makes it an invaluable tool for short term weather forecasting or flash flood forecasting.

*wradlib* is designed to assist you in the most important steps of processing weather radar data. These may include: reading common data formats, georeferencing, converting reflectivity to rainfall intensity, identifying and correcting typical error sources (such as clutter or attenuation) and visualising the data.

This documentation is under steady development. It provides a complete library reference as well as a set of tutorials which will get you started in working with *wradlib*. 

.. toctree::
   :maxdepth: 2
   
   gettingstarted   
   tutorials
   recipes
   reference
   dev_setup
   community

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

