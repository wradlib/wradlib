Getting Started
===============

Installing Python
-----------------

In order to run *wradlib*, you need to have a Python interpreter installed on your local computer. *wradlib* will be guaranteed to work with a particular Python version, however, we will not guarantee upward or downward compatibility at the moment. **The current version of wradlib is designed to be used with Python 2.7, but most features are also known to work under Python 2.6**.

*wradlib* was not designed to be a self-contained library. Besides extensive use of Numpy and Scipy, you *might* need to install additional libraries before you can use *wradlib*. See `Dependencies`_ for a full list of dependencies. Under **Linux**, the Python interpreter is usually pre-installed and installation of additional packages as well as dependency management is supposed to be easy. 

Under **Windows** operating systems, we strongly recommend to install a Python distribution such as Python(x,y) (http://code.google.com/p/pythonxy) which will contain most of the required packages. Go to http://code.google.com/p/pythonxy/wiki/Downloads, and select one of the Mirrors. Download the latest distribution (currently **Python(x,y)-2.7.2.3.exe** and install it. We recommend to use the full installation mode. 

**When installing Python(x,y), make sure you choose "All Users" under the "Install for" menu component!**

Test the integrity of your Python installation by opening a console window and typing *python*. The console should show the Python version and a Python command prompt. Type
 
>>> exit()
 
in order to exit the Python environment. 


Installing wradlib
------------------

Download the source from https://bitbucket.org/wradlib/wradlib via the ``get source`` button and extract it to any location on your computer. Inside the extracted folder, open a console window (on the same directory level as the setup.py file) and execute::

>>> python setup.py install

This way, the *wradlib* package will be installed under the Python site-packages directory and will thus be available for import.

Test the integrity of your wradlib installation by opening a console window and typing ``python``. The Python prompt should appear. Then type

>>> import wradlib

If everything is ok, nothing will happen. If the *wradlib* package is not found by the interpreter, you will get 

>>> import wradlib
ImportError: No module named wradlib

Check whether your Python installation directory contains the subdirectory */Lib/sitepackages/wradlib/wradlib* and the corresponding source files. If the wradlib directory is not there, execute ``python setup.py install``, again, and inspect the screen output if any helpful error messages are reported.

You may receive other ImportErrors if required packages are missing. Please make sure the required packages are installed (see `Dependencies`_).

**Attention:** In order to use *wradlib* for decoding BUFR files (see :doc:`tutorial_supported_formats`), the installation (via ``python setup.py install``) tries to compile and build the OPERA BUFR software (which is included in the wradlib source). This part of the installation has the potential to cause some trouble and was only tested on Windows 7 machines, yet. The process requires ``gcc`` and ``make``. Both are pre-installed on most Linux machines, and can be installed on Windows using the `MinGW compiler suite <http://www.mingw.org/wiki/Getting_Started>`_. **If you are using Python(x,y)**, ``gcc`` and ``mingw32-make`` should already be available on your machine! You can check this by opening a console window and typing ``gcc --version`` and ``mingw32-make --version``. For **Linux**, the required makefile is available and we hope that the installation process works. But we never tested it! Please give us your feedback how it works under Linux by sending an e-mail to wradlib-users@googlegroups.com or by `raising an issue <https://bitbucket.org/wradlib/wradlib/issues/new>`_.

Beyond this issue, *wradlib* is intended to support at least Windows and Linux platforms, but it was tested only on Windows, yet. If you discover any platform issues on Linux, please do not hesitate to `raise an issue <https://bitbucket.org/wradlib/wradlib/issues/new>`_.


Dependencies
------------

*wradlib* was not designed to be a self-contained library. Besides extensive use of Numpy and Scipy, *wradlib* uses additional libraries, which you will need to install before you can use *wradlib*. Note that all libraries marked with a (*) are *not* contained in the Python(x,y) distribution and thus have to be definitely installed manually.

For Windows users: If possible, we will link binary installer files for the libraries below. However, installers are not always available. In this case, you have to install from source. For pure Python packages listed in PyPI, this is easy. Just open a console window and type::

    easy_install *package_name* 

- basemap (*): Download installer at http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/basemap-1.0.2/

- h5py

- matplotlib

- netCDF4

- numpy

- numpydoc (*): Install via ``easy_install numpydoc``

- pylab

- pyproj (*): Download installer at http://code.google.com/p/pyproj/downloads

- scipy

- PyTables (tables)


Community
---------

*wradlib* is intended to be a community effort, and community needs communication. The key communication platform for *wradlib* is the  `wradlib-users <https://groups.google.com/forum/?fromgroups=#!forum/wradlib-users>`_ mailing list and forum. Through this forum, you can help to improve wradlib by reporting bugs, proposing enhancements, or by contributing code snippets (in any programming language) and documentation of algorithms. You can also ask other users and developers for help, or use your own knowledge and experience to help other users. We strongly encourage you to `subscribe <https://groups.google.com/group/wradlib-users/subscribe>`_ to this list. Check it out! 

Learn more about wradlib as a community effort :doc:`here <community>`!