Getting Started
===============

.. _ref-installation:

Installation
------------

In order to run :math:`\omega radlib`, you need to have a Python interpreter installed on your local computer, as well as a number of Python packages (`Dependencies`_). We recommend to install `Anaconda <https://www.continuum.io/why-anaconda/>`_ as it installs Python, a number of required packages, and other useful tools (e.g. spyder).

Using Anaconda the installation process is harmonised across platforms. Download and install the latest Anaconda distribution from https://www.continuum.io/downloads for your specific OS.
We are constantly performing tests with these distributions (for python2.7 and python3 respectively).

If your Anaconda Python installation is working, the following command (in a console) should work::

    $ python --version
    Python 3.5.1 :: Continuum Analytics, Inc.

Now you can use the ``conda`` package and environment manager (`conda documentation <http://conda.pydata.org/docs/#>`_) to setup your :math:`\omega radlib` installation.

#. Clone the root environment or create one from scratch::

    $ conda create --name wradlib --clone root
    or
    $ conda create --name wradlib python=2.7

#. Add the conda-forge channel, where :math:`\omega radlib` and its dependencies are located. Read more about the community effort `conda-forge <https://conda-forge.github.io/>`_::

    $ conda config --add channels conda-forge

#. Activate the :math:`\omega radlib` environment

    Linux::

        $ source activate wradlib

    Windows::

        > activate wradlib

#. Install :math:`\omega radlib` and its dependencies::

    (wradlib) $ conda install wradlib

#. Make sure the GDAL_DATA environment variable (needed for georeferencing) is set within your environment. If not, you can set it like this:

    Linux/OSX::

        (wradlib) $ export GDAL_DATA=/path/to/anaconda/envs/wradlib/share/gdal

    Windows CMD.exe::

        [wradlib] > setx GDAL_DATA C:\path\to\anaconda\envs\wradlib\Library\share\gdal

Now you have a ``conda`` environment with a working :math:`\omega radlib` installation.

Test the integrity of your :math:`\omega radlib` installation by opening a console window and typing calling the python interpreter::

    $ python
    Python 3.5.1 |Continuum Analytics, Inc.| (default, Dec  7 2015, 11:16:01)
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
    Type "help", "copyright", "credits" or "license" for more information.

The Python prompt should appear. Then type::

    >>> import wradlib
    >>> wradlib.__version__
    '0.8.0'

If everything is ok, this will show the running :math:`\omega radlib` version. If the :math:`\omega radlib` package is not found by the interpreter, you will get::

    >>> import wradlib
    ImportError: No module named wradlib

Alternatively, you can install the `Bleeding edge code`_, but you have to keep track of :math:`\omega radlib's` dependencies yourself.


Bleeding edge code
------------------

.. warning:: The :math:`\omega radlib` version on `PyPI <https://pypi.python.org/pypi/wradlib>`_ might lag behind the actual developments. You can use the bleeding edge code from the :math:`\omega radlib` `repository <https://github.com/wradlib/wradlib>`_. Note, however, that you need to make sure yourself that all `Dependencies`_ are met (see below).

`Download <https://github.com/wradlib/wradlib/archive/master.zip>`_ the source, unzip, and run::

    $ python setup.py install

Alternatively, you can add the :math:`\omega radlib` directory to your environment variable ``PYTHONPATH``.


Installing via easy_install
---------------------------

Although we recommend using the Anaconda Python Environment you can install :math:`\omega radlib` from PyPi via easy_install.

Open a terminal and run::

    $ easy_install wradlib

Depending on your system you might need to be root (or sudo the above command) for this to work.
easy_install will then fetch the source distribution from the Python Package Index and run the installation.

Afterwards it will check for any dependencies not met, yet.

Be aware that with easy_install we can only look for python-module dependencies.
For example the numpy module itself depends on some other libraries, which need to be present in order for the module to compile properly after being downloaded by easy_install. We have no control over these dependencies and it is rather hard to give a complete overview.

Therefore we recommend trying to satisfy the dependencies using your favorite package management system.


.. _ref-dependencies:

Dependencies
------------

:math:`\omega radlib` was not designed to be a self-contained library. Besides extensive use of Numpy and Scipy, :math:`\omega radlib` uses additional libraries, which you will need to install before you can use :math:`\omega radlib`.

.. tabularcolumns:: |L|L|L|]

+------------+-----------+-------------+
| Package    |    min    | recommended |
+============+===========+=============+
| numpy      | >= 1.6.1  | >= 1.10.1   |
+------------+-----------+-------------+
| matplotlib | >= 1.1.0  | >= 1.5.1    |
+------------+-----------+-------------+
| scipy      | >= 0.9    | >= 0.17.0   |
+------------+-----------+-------------+
| h5py       | >= 2.0.1  | >= 2.5.0    |
+------------+-----------+-------------+
| netCDF4    | >= 1.0    | >= 1.2.2    |
+------------+-----------+-------------+
| gdal       | >= 1.9    | >= 2.1.0    |
+------------+-----------+-------------+

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

Apart from the obligatory `Dependencies`_, some dependencies in :math:`\omega radlib` are optional. This is because the installation of these dependencies can be somewhat tedious while many :math:`\omega radlib` users will not need them anyway. In case users use a :math:`\omega radlib` function that requires an optional dependency, and this dependency is not satisfied in the local environment, :math:`\omega radlib` will raise an exception.

As for now, the following dependencies are defined as optional:

**The speedup module**

The speedup module is intended as a collection of Fortran code in order to speed up specific :math:`\omega radlib` function that are critical for performance.
In order to build the speedup module as a shared library, you need to use f2py (https://sysbio.ioc.ee/projects/f2py2e/). f2py usually ships with numpy and should be available via the command line. To test whether f2py is available on your system, execute ``f2py`` on the system console. Or, alternatively, ``f2py.py``. If it is available, you should get a bunch of help instructions. Now change to the :math:`\omega radlib` module directory and execute on the system console::

    $ f2py.py -c -m speedup speedup.f

Now the speedup module should be available.

**xmltodict**

We use xmltodict to convert the Rainbow Data Files (which have a metadata XML header) to an ordered dict. It is easily installed with `pip`::

    $ pip install xmltodict


.. _ref-knownissues:

Known Issues
------------

Depending on your OS and installation method you may encounter different problems. Here are some guidelines for attacking them.

Generally it is a good idea to use your systems package manager to install dependencies. This will also take account for other needed bindings, libs etc. Windows user should install one of the (scientific) python packages to resolve the problems there.

We recommend using the Anaconda conda package and environment manager (see `Installation`_).

If you encounter problems installing :math:`\omega radlib`, check on your favorite search engine or create an issue `here <https://github.com/wradlib/wradlib/issues>`_ with details on the problem or send an email on the `wradlib-users <https://groups.google.com/forum/?fromgroups=#!forum/wradlib-users>`_ mailing list.


Community
---------

:math:`\omega radlib` is intended to be a community effort, and community needs communication.

The key communication platform for :math:`\omega radlib` is the  `wradlib-users <https://groups.google.com/forum/?fromgroups=#!forum/wradlib-users>`_ mailing list and forum. Through this forum, you can help to improve :math:`\omega radlib` by reporting bugs, proposing enhancements, or by contributing code snippets (in any programming language) and documentation of algorithms.

You can also ask other users and developers for help, or use your own knowledge and experience to help other users. We strongly encourage you to `subscribe <https://groups.google.com/forum/#!forum/wradlib-users/join>`_ to this list. Check it out!

Learn more about :math:`\omega radlib` as a community effort :doc:`here <community>`!
