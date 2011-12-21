Development Setup
=================

In the future, each release of wradlib will be guaranteed to work with a 
certain version of python(x,y) under MS Windows. The current version is 2.6.6.2. 
On other operating systems, the library should work, if similar versions of the 
required packages are installed.

However, at the moment you will need to install the following packages beyond python(x,y):

- pyproj: Performs cartographic transformations and geodetic computations. 
  Visit http://code.google.com/p/pyproj/ and download the package installer.

- basemap: The matplotlib basemap toolkit is a library for plotting 2D data on maps. Find
  documentation on http://matplotlib.github.com/basemap and download the package installer
  at http://sourceforge.net/projects/matplotlib/files/matplotlib-toolkits/basemap-1.0.2/



Documentation Setup
-------------------

There are a few things that need to be done in order for the documentation to be 
generated properly.

The documentation tool is Sphinx (**todo add link to webpage**).

We oriented ourselves at the Numpy/Scipy documentation concerning docstrings. 
This implies using numpydoc to enable Sphinx to understand the formatting of 
those docstrings.

So before you can run ``make html`` in the doc-folder you need to install numpydoc

We plan to use jsmath for typesetting mathematical formula. As the jsmath 
package is large, it is usually not distributed but needs to be installed 
individually. So in order to have the maths properly set, you need to put the 
jsmath files in doc/source/static/jsMath
