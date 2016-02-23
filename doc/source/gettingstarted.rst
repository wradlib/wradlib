Getting Started
===============

Bleeding edge code
------------------

.. warning:: The wradlib version on `PyPI <https://pypi.python.org/pypi/wradlib>`_ might lag behind the actual developments. You can use the bleeding edge code from the `wradlib repository <https://bitbucket.org/wradlib/wradlib>`_. Note, however, that you need to make sure yourself that all `Dependencies`_ are met (see below).

`Download <http://bitbucket.org/wradlib/wradlib/get/default.zip>`_ the source, unzip, and run:

``$ python setup.py install``

Alternatively, you can add the wradlib directory to your environment variable ``PYTHONPATH``.
  

Installation under Windows
--------------------------

In order to run *wradlib*, you need to have a Python interpreter installed on your local computer, as well as a number of Python packages (`Dependencies`_). We recommend to install `Python(x,y) <http://code.google.com/p/pythonxy>`_ as it installs Python, a number of required packages, and other useful tools (e.g. MinGW).

Download and install the latest Python(x,y) distribution from http://code.google.com/p/pythonxy. 
The latest distribution tested by us is **Python(x,y)-2.7.9.0.exe**. 
We recommend to use the full installation mode, because this will install all packages. 
Otherwise you have to make sure that all prerequisites mentioned below will be installed, which may not be the case with a standard installation. 
If you have administrative privileges, make sure you choose "All Users" under the "Install for" menu component.
Installation for the current user only, may work, but may also produce all kinds of unexpected behavior.

If your Python installation is working, the following command (in a DOS console) should work:

``> python --version``

Now you can install wradlib. Open a DOS console window and type:

``> easy_install wradlib``

Alternatively, you can install the `Bleeding edge code`_.

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
^^^^^^^^^^^^^^^^^^^

This way no dependency checking is done and you need to make sure yourself that all dependencies (see below) are met.

`Download the source <http://bitbucket.org/wradlib/wradlib/get/default.zip>`_, unpack, go to the directory, where you unpacked the files and run:

``$ python setup.py install``

This will install wradlib to your site-packages or dist-packages folder.

Installing via easy_install
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open a terminal and run:

``$ easy_install wradlib``

Depending on your system you might need to be root (or sudo the above command) for this to work.
easy_install will then fetch the source distribution from the Python Package Index and run the installation.

Afterwards it will check for any dependencies not met, yet.

Be aware that with easy_install we can only look for python-module dependencies.
For example the numpy module itself depends on some other libraries, which need to be present in order for the module to compile properly after being downloaded by easy_install. We have no control over these dependencies and it is rather hard to give a complete overview.

Therefore we recommend trying to satisfy the dependencies using your favorite package management system.

Since python environments as Enthought Python or Anaconda Python have ermerged and matured in the past, we also recommend to use these environments.

As an example we give all packages necessary to successfully install wradlib on a Ubuntu Linux 12.04 (thanks to Diego Civera from the wradlib-users mailing list for pioneering this).

Satisfying wradlib dependencies using apt-get on Ubuntu 12.04 LTS (precise pangolin)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to get started very quickly and are not afraid to use third party repositories, then the following lines should get you going within minutes (the '$' prompt means that these commands should be issued on a terminal)

| ``$ sudo add-apt-repository ppa:adrian-m-benson/ppa``
| ``$ sudo apt-get update``
| ``$ sudo apt-get install python-setuptools python-numpy python-scipy python-matplotlib python-tables python-h5py python-netcdf4 python-gdal``
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
Projections, which are the main part of georeferencing are handled by the gdal package

``$ sudo apt-get install python-gdal``

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

- gdal, version >= 1.9

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

**The speedup module**

The speedup module is intended as a collection of Fortran code in order to speed up specific wradlib function that are critical for performance.
In order to build the speedup module as a shared library, you need to use f2py (http://cens.ioc.ee/projects/f2py2e/). f2py usually ships with numpy and should be available via the command line. To test whether f2py is available on your system, execute ``f2py`` on the system console. Or, alternatively, ``f2py.py``. If it is available, you should get a bunch of help instructions. Now change to the wradlib module directory and execute on the system console:

``f2py.py -c -m speedup speedup.f``

Now the speedup module should be availble.

**xmltodict**

We use xmltodict to convert the Rainbow Data Files (which have an metadata XML header) to an ordered dict.

.. _ref-knownissues:

Known Issues
------------

Depending on your OS and installation method you may encounter different problems. Here are some guidelines for attacking them.

Generally it is a good idea to use your systems package manager to install the dependencies. This will also take account for other needed bindings, libs etc. Windows user should install one of the (scientific) python packages to resolve the problems there.

You may install the requirements via pip for all requirements::

    pip install -r requirements.txt

or for any requirement itself::

    pip install 'numpy>=1.7.1'

If you are installing wradlib and the missing dependencies via pip or setup.py there may be missing some libraries and/or include ('header') files. The only solutions to this is to install the missing libraries via packet manager or compile them from scratch (windows user using the python packages should not encounter such problems).

If you are installing wradlib and the missing dependencies via pip or setup.py there also may be version conflicts between the packages, some libraries and/or include ('header') files. If, for instance, the newest available gdal-devel libraries which come with your system are version 1.10.0, but gdal version downloaded from PyPI is 1.11.0, then this may have an error at compile time as a result. Solution is to explicitely declare the gdal version::

    pip install 'gdal==1.10.0'

This may also be an issue with other dependencies which are relying on libraries.

If you are in need to install everything from scratch, or if you are setting up a clean virtual environment, etc., you may encounter some other strange problems. Especially in virtual environments you may have to export some PATH variables so that libraries and includes can be found.

If all this doesn't help, check on your favorite search engine or create an issue `here <https://bitbucket.org/wradlib/wradlib/issues?status=new&status=open>`_ with details on the problem or send an email on the `wradlib-users <https://groups.google.com/forum/?fromgroups=#!forum/wradlib-users>`_ mailing list.


Community
---------

*wradlib* is intended to be a community effort, and community needs communication. The key communication platform for *wradlib* is the  `wradlib-users <https://groups.google.com/forum/?fromgroups=#!forum/wradlib-users>`_ mailing list and forum. Through this forum, you can help to improve wradlib by reporting bugs, proposing enhancements, or by contributing code snippets (in any programming language) and documentation of algorithms. You can also ask other users and developers for help, or use your own knowledge and experience to help other users. We strongly encourage you to `subscribe <https://groups.google.com/group/wradlib-users/subscribe>`_ to this list. Check it out! 

Learn more about wradlib as a community effort :doc:`here <community>`!
