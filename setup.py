#!/usr/bin/env python
# Copyright (c) 2011-2021, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""wradlib - An Open Source Library for Weather Radar Data Processing

wradlib is designed to assist you in the most important steps of
processing weather radar data. These may include: reading common data
formats, georeferencing, converting reflectivity to rainfall
intensity, identifying and correcting typical error sources (such as
clutter or attenuation) and visualising the data.
"""

DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: Education
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Atmospheric Science
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
"""

NAME = "wradlib"
MAINTAINER = "wradlib developers"
MAINTAINER_EMAIL = "wradlib@wradlib.org"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])
URL = "https://wradlib.org"
DOWNLOAD_URL = "https://github.com/wradlib/wradlib"
LICENSE = "MIT"
CLASSIFIERS = list(filter(None, CLASSIFIERS.split("\n")))
PLATFORMS = ["Linux", "Mac OS-X", "Unix", "Windows"]


def setup_package():

    from setuptools import find_packages, setup

    with open("requirements.txt", "r") as f:
        INSTALL_REQUIRES = [rq for rq in f.read().split("\n") if rq != ""]

    with open("requirements_optional.txt", "r") as f:
        OPTIONAL_REQUIRES = [rq for rq in f.read().split("\n") if rq != ""]

    with open("requirements_devel.txt", "r") as f:
        DEVEL_REQUIRES = [rq for rq in f.read().split("\n") if rq != ""]

    INSTALL_REQUIRES += OPTIONAL_REQUIRES

    metadata = dict(
        name=NAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        url=URL,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        classifiers=CLASSIFIERS,
        platforms=PLATFORMS,
        setup_requires=["setuptools_scm"],
        install_requires=INSTALL_REQUIRES,
        extras_require=dict(
            dev=DEVEL_REQUIRES,
        ),
        packages=find_packages(),
        entry_points={
            "xarray.backends": [
                "cfradial1 = wradlib.io.backends:CfRadial1BackendEntrypoint",
                "cfradial2 = wradlib.io.backends:CfRadial2BackendEntrypoint",
                "furuno = wradlib.io.backends:FurunoBackendEntrypoint",
                "gamic = wradlib.io.backends:GamicBackendEntrypoint",
                "odim = wradlib.io.backends:OdimBackendEntrypoint",
                "radolan = wradlib.io.backends:RadolanBackendEntrypoint",
                "iris = wradlib.io.backends:IrisBackendEntrypoint",
                "rainbow = wradlib.io.backends:RainbowBackendEntrypoint",
            ]
        },
    )

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
