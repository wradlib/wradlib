Development Setup
=================

As a convention, each release of wradlib will be guaranteed to work with a 
certain version of python(x,y) under MS Windows. The current version is 2.6.6.2. 
On other operating systems, the library should work, if similar versions of the 
required packages are installed.

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
