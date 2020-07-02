# README #

[![Join the chat at https://gitter.im/wradlib/wradlib](https://badges.gitter.im/wradlib/wradlib.svg)](https://gitter.im/wradlib/wradlib?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

**To anyone who wants to use a bleeding edge version of wradlib from this repository:**

wradlib uses the Cloud Services [Travis CI](https://travis-ci.com/), [Appveyor](https://www.appveyor.com/) and [Azure Pipelines](https://azure.microsoft.com/de-de/services/devops/pipelines/) for Continuous Integration. This means that once new code is pushed to the wradlib repository, a suite of tests and checks are run. In case the tests run without failure, a new documentation will be built on readthedocs and is available at the [wradlib documentation website](https://docs.wradlib.org).

Below you see the status of wradlib. A green status indicates that the current revision of wradlib did pass the tests successfully. You should be aware of that a green status is not a guarantee that the latest revision is bug free. It just means that our tests did not find any bugs. Also the status of code coverage, of ReadTheDocs documentation builds and of availability of wradlib conda package is displayed.

## Package Status ##

| Item  | Status  |
|:---:|:---:|
| Travis CI | [![Build Status](https://travis-ci.com/wradlib/wradlib.svg?branch=master)](https://travis-ci.org/wradlib/wradlib) |
| Appveyor | [![Build status](https://ci.appveyor.com/api/projects/status/7x7xs1t6hg9m3h2b?svg=true)](https://ci.appveyor.com/project/wradlib/wradlib) |
| Azure | [![Build Status](https://dev.azure.com/wradlib/wradlib/_apis/build/status/wradlib.wradlib?branchName=master)](https://dev.azure.com/wradlib/wradlib/_build/latest?definitionId=1&branchName=master) |
| Codecov | [![codecov](https://codecov.io/gh/wradlib/wradlib/branch/master/graph/badge.svg)](https://codecov.io/gh/wradlib/wradlib) |
| RTD Latest | [![ReadTheDocs Latest](https://readthedocs.org/projects/wradlib-docs/badge/?version=latest)](https://docs.wradlib.org/en/latest/) |
| RTD Stable | [![ReadTheDocs Stable](https://readthedocs.org/projects/wradlib-docs/badge/?version=stable)](https://docs.wradlib.org/en/stable/) |
| Anaconda Latest | [![Anaconda Latest](https://anaconda.org/conda-forge/wradlib/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/wradlib/) |
| Anaconda Version | [![Anaconda Status](https://anaconda.org/conda-forge/wradlib/badges/version.svg)](https://anaconda.org/conda-forge/wradlib/) |

### Documentation ###

For further information on wradlib (getting started, library reference, development setup, ...), please see our homepage:

https://wradlib.org

Specific information for developers is available via the [wradlib-wiki](https://github.com/wradlib/wradlib/wiki).

### Citing wradlib

You can cite wradlib using the DOI below, or select the fitting zenodo DOI if you want to cite a distinct wradlib version by following the link.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1209843.svg)](https://doi.org/10.5281/zenodo.1209843)

### User forum ###

In order to stay tuned, we recommend that you register to the [wradlib user forum and mailing list](https://groups.google.com/forum/?fromgroups#!forum/wradlib-users). 

### Found any bugs or want to add wishes ###

Create an issue [here](https://github.com/wradlib/wradlib/issues).
