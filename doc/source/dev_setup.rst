Development Setup
=================

In the future, each release of wradlib will be guaranteed to work with a certain version of python(x,y) under MS Windows. The current version is 2.7.2.3. On other operating systems, the library should work, if similar versions of the required packages are installed.

However, at the moment you will need to install the following packages beyond python(x,y):

- pyproj: Performs cartographic transformations and geodetic computations. Visit http://code.google.com/p/pyproj/ and download the package installer.

- basemap: The matplotlib basemap toolkit is a library for plotting 2D data on maps. Find documentation on http://matplotlib.org/basemap and `download the latest basemap installer <http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits>`_.


Documentation Setup
-------------------

There are a few things that need to be done in order for the documentation to be built properly.

The documentation tool is `Sphinx <http://sphinx.pocoo.org/>`_. We oriented ourselves at the Numpy/Scipy documentation concerning docstrings. This implies using `numpydoc <http://pypi.python.org/pypi/numpydoc>`_ to enable Sphinx to understand the formatting of those docstrings. If you installed Python(x,y) Enthought Python, Sphinx is already installed. In order to install numpydoc, simply open a console window and type ``easy_install numpydoc``.

Now you can open a console window in the folder wradlib/doc and execute ``make html``. This will give you the latest documentation under the wradlib\doc\build\html directory. Simply open the index.html file to view the documentation.

