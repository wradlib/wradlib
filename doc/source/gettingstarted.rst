Getting Started
===============

Installation under Windows
--------------------------

In order to run *wradlib*, you need to have a Python interpreter installed on your local computer, as well as a number of Python packages (`Dependencies`_). We strongly recommend to install Python(x,y). It installs Python, a number of required packages, and other useful tools (e.g. MinGW).

Go to http://code.google.com/p/pythonxy/wiki/Downloads, and select one of the Mirrors. Download the latest distribution (currently **Python(x,y)-2.7.6.0.exe** and install it. We recommend to use the full installation mode. Make sure you choose "All Users" under the "Install for" menu component! Make sure you choose "All Users" under the "Install for" menu component (in case you have administrative privileges)!

If you Python installation is working, the following command (in a DOS console) should work:

``> python --version``

Now you can install wradlib. Open a DOS console window and type:

``> easy_install wradlib``

Test the integrity of your wradlib installation by opening a console window and typing ``>python``. The Python prompt should appear. Then type

>>> import wradlib

If everything is ok, nothing will happen. If the *wradlib* package is not found by the interpreter, you will get 

>>> import wradlib
ImportError: No module named wradlib

**Gotchas under Windows**

If you do not install Python(x,y), you will need to make sure by yourself that everything is there: The Python interpreter (version 2.7) and all `Dependencies`_. You can test the availability of the required tools by opening a DOS console and requesting the versions of the required software. For Python, it is important that the version is 2.7.x, for the other tools, the version number is not so important, but rather checks whether the software is available at all:

``> python --version``

Ideally, all installations should be done with administrative privileges. In case you do not have administrative privileges, you can instll wradib via:

``> easy_install wradlib --user``  


Installation under Linux
------------------------

Installing Manually
^^^^^^^^^^^^^^^^^^^^

This way no dependency checking is done and you need to make sure yourself that all dependencies (see below) are met.

`Download the source <http://bitbucket.org/wradlib/wradlib/get/default.zip>`_, unpack, go to the directory, where you unpacked the files and run:

``$ python setup.py install``

This will install wradlib to your site-packages or dist-packages folder.

Installing via easy_install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open a terminal and run:

``$ easy_install wradlib``

Depending on your system you might need to be root (or sudo the above command) for this to work.
easy_install will then fetch the source distribution from the Python Package Index and run the installation.

Afterwards it will check for any dependencies not met, yet.

Be aware that with easy_install we can only look for python-module dependencies.
For example the pyproj module itself depends on some other libraries, which need to be present in order for the module to compile properly after being downloaded by easy_install. We have no control over these dependencies and it is rather hard to give a complete overview.

Therefore we recommend trying to satisfy the dependencies using your favorite package management system.
As an example we give all packages necessary to successfully install wradlib on a Ubuntu Linux 12.04 (thanks to Diego Civera from the wradlib-users mailing list for pioneering this).

Satisfying wradlib dependencies using apt-get on Ubuntu 12.04 LTS (precise pangolin)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to get started very quickly and are not afraid to use third party repositories, then the following lines should get you going within minutes (the '$' prompt means that these commands should be issued on a terminal)

| ``$ sudo add-apt-repository ppa:adrian-m-benson/ppa``
| ``$ sudo apt-get update``
| ``$ sudo apt-get install python-setuptools python-numpy python-scipy python-matplotlib python-tables python-h5py python-netcdf4 python-pyproj``
| ``$ sudo easy_install wradlib``


We will break up the installation into several steps, in case something goes wrong along the way, or you would like to know better, what is happening.

1. Getting easy_install
"""""""""""""""""""""""
As there is no wradlib Debian package yet, the python-setuptools package gives you access to easy_install.

``$ sudo apt-get install python-setuptools``

2. NumPy, SciPy and Matplotlib
""""""""""""""""""""""""""""""
NumPy and SciPy are needed for many algorithms and Matplotlib is used for visualization.

``$ sudo apt-get install python-numpy python-scipy python-matplotlib``

3. Data Formats: HDF5 and NetCDF4
"""""""""""""""""""""""""""""""""
The two major packages for working with HDF5 files are h5py and PyTables

``$ sudo apt-get install python-h5py python-tables``

For NetCDF4, there are no official Debian Packages for Ubuntu 12.04

| You have two options here.  
| **EITHER** You can try a user repository:

| ``$ sudo add-apt-repository ppa:adrian-m-benson/ppa``
| ``$ sudo apt-get update``
| ``$ sudo apt-get install python-netcdf4``

**OR** you satisfy the source dependencies of python-netcdf4 and let easy_install do the rest:

| ``$ sudo apt-get install libnetcdf-dev libhdf5-dev``
| ``$ sudo easy_install netCDF4``

4. georeferencing
"""""""""""""""""
Projections, which are the main part of georeferencing are handled by the pyproj package

``$ sudo apt-get install python-pyproj``

5. wradlib
""""""""""
Finally you can install wradlib. 

``$ sudo easy_install wradlib``

This will install wradlib and all missing dependencies.

**Gotchas under Linux**

As of Ubuntu 12.04 numpydoc, which is needed to build the source documentation of wradlib also does not have a Debian package. This will change in more current versions. As numpydoc seems to be purely written in python, easy_install should be able to install it without problems. However, numpydoc is not needed in order to use wradlib.

.. _ref-dependencies:

Dependencies
------------

*wradlib* was not designed to be a self-contained library. Besides extensive use of Numpy and Scipy, *wradlib* uses additional libraries, which you will need to install before you can use *wradlib*. Note that all libraries marked with a (*) are *not* contained in the Python(x,y) distribution. Installers for the remaining libraries can be downloaded at http://code.google.com/p/pythonxy/wiki/StandardPlugins.

- numpy, version >= 1.6.1

- matplotlib, version >= 1.1.0

- scipy, version >= 0.9

- h5py, version >= 2.0.1

- netCDF4, version >= 1.0

- numpydoc (*), version >= 0.3, install via ``easy_install numpydoc``

- pyproj (*), version >= 1.8.9, install via ``easy_install pyproj``

You can check whether the required `Dependencies`_ are available on your computer by opening a Python console and enter:

>>> import <package_name>
ImportError: No module named <package_name>
 
This will be the response in case the package is not available. 

In case the import is successful, you should also check the version number:

>>> package_name.__version__
some version number

The version number should be consistent with the above `Dependencies`_.


Optional Dependencies
---------------------

Apart from the obligatory `Dependencies`_, some dependencies in wradlib are optional. This is because the installation of these dependencies can be somewhat tedious while many wradlib users will not need them anyway. In case users use a wradlib function that requires an optional dependency, and this dependency is not satisfied in the local environment, wradlib will raise an exception.

As for now, the following dependencies are defined as optional:

**BUFR software**

The *bufr* module requires the installation of the OPERA BUFR software.

*TODO:* Insert instructions for installing the BUFR software.

**The speedup module**

The speedup module is intended as a collection of Fortran code in order to speed up specific wradlib function that are critical for performance.
In order to build the speedup module as a shared library, you need to use f2py (http://cens.ioc.ee/projects/f2py2e/). f2py usually ships with numpy and should be available via the command line. To test whether f2py is available on your system, execute ``f2py`` on the system console. Or, alternatively, ``f2py.py``. If it is available, you should get a bunch of help instructions. Now change to the wradlib module directory and execute on the system console:

``f2py.py -c -m speedup speedup.f``

Now the speedup module should be availble.
 

Community
---------

*wradlib* is intended to be a community effort, and community needs communication. The key communication platform for *wradlib* is the  `wradlib-users <https://groups.google.com/forum/?fromgroups=#!forum/wradlib-users>`_ mailing list and forum. Through this forum, you can help to improve wradlib by reporting bugs, proposing enhancements, or by contributing code snippets (in any programming language) and documentation of algorithms. You can also ask other users and developers for help, or use your own knowledge and experience to help other users. We strongly encourage you to `subscribe <https://groups.google.com/group/wradlib-users/subscribe>`_ to this list. Check it out! 

Learn more about wradlib as a community effort :doc:`here <community>`!
