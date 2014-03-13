# distribute_setup module will automatically download a matching version
#   of setuptools from PyPI, if it isn't present on the target system.
#import distribute_setup
#distribute_setup.use_setuptools()


import os
import sys
import site

### BEFORE importing distutils, remove MANIFEST. distutils doesn't
### properly update it when the contents of directories change.
##if os.path.exists('MANIFEST'): os.remove('MANIFEST')

from distutils.core import setup
#from setuptools import setup
#from distutils.sysconfig import get_python_lib

#import subprocess as sub

# get current version from file
with open("version") as f:
    VERSION = f.read()
    VERSION = VERSION.strip()

##MAJOR               = 0
##MINOR               = 1
##MICRO               = 1
##VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


if __name__ == '__main__':

    setup(name='wradlib',
          version=VERSION,
          description='Open Source Library for Weather Radar Data Processing',
          long_description = """\
wradlib - An Open Source Library for Weather Radar Data Processing
==================================================================

wradlib is designed to assist you in the most important steps of
processing weather radar data. These may include: reading common data
formats, georeferencing, converting reflectivity to rainfall
intensity, identifying and correcting typical error sources (such as
clutter or attenuation) and visualising the data.

""",
          license='BSD',
          url='http://wradlib.bitbucket.org/',
          download_url='https://bitbucket.org/wradlib/wradlib',
          packages=['wradlib'],
          include_package_data=True, # see MAINFEST.in
          classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: BSD License',
          'Environment :: Console',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          ],
          install_requires=["numpydoc >= 0.3", "pyproj >= 1.8",
                            "netCDF4 >= 1.0", "h5py >= 2.0.1",
                            "matplotlib >= 1.1.0", "scipy >= 0.9", "numpy >= 1.7.0"]
          )

