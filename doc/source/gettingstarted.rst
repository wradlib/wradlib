Getting Started
===============

Installing Python
-----------------

In order to run *wradlib*, you need to have a Python interpreter installed on your local computer. *wradlib* will be guaranteed to work with a particular
Python version, however, we will not guarantee upward or downward compatibility at the moment. **The current version of wradlib is designed to be used with Python 2.6.6.**

*wradlib* was not designed to be a self-contained library. Besides extensive use of Numpy and Scipy, you *might* need to install additional libraries before you can use *wradlib*. See `Dependencies`_ for a full list of dependencies. Under **Linux**, the Python interpreter is usually pre-installed and installation of additional packages is supposed to be easy - but you will have to do it explicitely for each required package. 

Under **Windows** operating systems, we recommend to install a Python distribution such as Python(x,y) (http://code.google.com/p/pythonxy) which will contain most of the required packages. Go to http://code.google.com/p/pythonxy/wiki/Downloads, and select one of the Mirrors. Download **Python(x,y)-2.6.6.2.exe** and install it. We recommend to use the full installation mode. 

**When installing Python(x,y), make sure you choose "All Users" under the "Install for" menu component!**

Test the integrity of your Python installation by opening a console window and typing *python*. The console should show the Python version and a Python command prompt. Type
 
>>> exit()
 
in order to exit the Python environment. 


Installing wradlib
----------------

There is no installer available, yet. Just download the source from https://bitbucket.org/wradlib/wradlib and copy the *wradlib* directory into your Python installation directory under *./Lib/site-packages/wradlib*. 

Alternatively, copy the wradlib directory where you like and add that path to your environment variable PYTHONPATH. Under Windows, this usually implies selecting System properties > Advanced system settings > Environment Variables > Edit PYTHONPATH and then add the path to your wradlib directory (seperate it from other directories by a semicolon). 

Test the integrity of your wradlib installation by opening a console window and typing *python*. The Python prompt should appear. Then type

>>> import wradlib

If everything is ok, nothing will happen. If the *wradlib* directory is not found by the interpreter, you will get 

>>> import wradlib
ImportError: No module named wradlib

Check your PYTHONPATH, again, or check whether your Python directory contains the subdirectory *./Lib/sitepackages/wradlib/wradlib* and the corresponding source files. Depending on your local settings in Windows, you might need to set the PYTHONPATH not only under the *System variables*, but also under the *User variables*.

You may receive other ImportErrors if required packages are missing. Please make sure the required packages are installed (see `Dependencies`_).

*wradlib* is intended to support at least Windows and Linux platforms, but it was tested only on Windows, yet. If you discover any platform issues on Linux, please do not hesitate to raise an issue on https://bitbucket.org/wradlib/wradlib/issues.


Dependencies
------------

*wradlib* was not designed to be a self-contained library. Besides extensive use of Numpy and Scipy, *wradlib* uses additional libraries, which you will need to install before you can use *wradlib*. Note that all libraries marked with a (*) are *not* contained in the Python(x,y) distribution and thus have to be definitely installed manually.

For Windows users: If possible, we will link binary installer files for the libraries below. However, installers are not always available. In this case, you have to install from source. For pure Python packages, this is easy. Just extract the source, open a console window on the same level that conatins the ``setup.py`` file and execute::

   python setup.py install

- basemap (*): Download installer at http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/basemap-1.0.2/

- matplotlib

- netCDF4

- numpy

- numpydoc (*): Download source at http://pypi.python.org/pypi/numpydoc and install via *python setup.py install*

- pylab

- pyproj (*): Download installer at code.google.com/p/pyproj/downloads

- scipy

