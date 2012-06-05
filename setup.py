import os

# BEFORE importing distutils, remove MANIFEST. distutils doesn't
# properly update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

from distutils.core import setup

MAJOR               = 0
MINOR               = 1
MICRO               = 1
VERSION             = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

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
          classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: BSD License',
          'Environment :: Console',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          ],
          )

