#!/usr/bin/env python
"""wradlib - An Open Source Library for Weather Radar Data Processing

wradlib is designed to assist you in the most important steps of
processing weather radar data. These may include: reading common data
formats, georeferencing, converting reflectivity to rainfall
intensity, identifying and correcting typical error sources (such as
clutter or attenuation) and visualising the data.
"""

import os
import sys
from subprocess import check_output, CalledProcessError

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: Education
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Atmospheric Science
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
"""

NAME = 'wradlib'
MAINTAINER = "wradlib developers"
MAINTAINER_EMAIL = "wradlib@wradlib.org"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
URL = "http://wradlib.org"
DOWNLOAD_URL = "https://github.com/wradlib/wradlib"
LICENSE = 'MIT'
CLASSIFIERS = filter(None, CLASSIFIERS.split('\n'))
PLATFORMS = ["Linux", "Mac OS-X", "Unix", "Windows"]
MAJOR = 1
MINOR = 0
PATCH = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, PATCH)


# Return the git revision as a string
def git_version():
    try:
        rev = check_output(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = rev.strip().decode('ascii')
    except (CalledProcessError, OSError):
        GIT_REVISION = 'unknown'
    return GIT_REVISION


# This is a bit hackish: we are setting a global variable so that the main
# wradlib __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.
builtins.__WRADLIB_SETUP__ = True


def write_version_py(filename='wradlib/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM WRADLIB SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of wradlib.version messes up the build under
    # Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('wradlib/version.py'):
        # must be a source distribution, use existing version file
        try:
            from wradlib.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "wradlib/version.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev+' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


def setup_package():

    # rewrite version file
    write_version_py()

    from setuptools import setup, find_packages

    with open('requirements.txt', 'r') as f:
        INSTALL_REQUIRES = [rq for rq in f.read().split('\n') if rq != '']

    with open('requirements_devel.txt', 'r') as f:
        DEVEL_REQUIRES = [rq for rq in f.read().split('\n') if rq != '']

    metadata = dict(
        name=NAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        classifiers=CLASSIFIERS,
        platforms=PLATFORMS,
        install_requires=INSTALL_REQUIRES,
        extras_require={'dev': DEVEL_REQUIRES},
        packages=find_packages(),
        entry_points={
            'console_scripts': [
                'testrunner=scripts.testrunner:main'
            ]
        },
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
