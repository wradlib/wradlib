#!/usr/bin/env bash
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

set -e

# print the travis CI vars
echo "TRAVIS_PULL_REQUEST " $TRAVIS_PULL_REQUEST
echo "TRAVIS_SECURE_ENV_VARS " $TRAVIS_SECURE_ENV_VARS
echo "TRAVIS_TAG " $TRAVIS_TAG ${TRAVIS_TAG:1}

# get python major version
PY_MAJOR="${PYTHON_VERSION%.*}"

# download and install latest micromamba
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba --strip-components=1
./micromamba shell init -s bash -p ~/micromamba
cat ~/.bashrc
export MAMBA_ROOT_PREFIX=~/micromamba
export MAMBA_EXE=$(pwd)/micromamba
. ${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh
echo "micromamba version $(micromamba --version)"

WRADLIB_ENV="travis_wradlib"
WRADLIB_PYTHON=$PYTHON_VERSION

echo "PATH:" $PATH
echo "WRADLIB_ENV:" $WRADLIB_ENV
echo "WRADLIB_PYTHON:" $WRADLIB_PYTHON
echo "GDAL_VERSION:" $GDAL_VERSION

# setup wradlib dependencies
WRADLIB_DEPS="gdal=$GDAL_VERSION numpy scipy matplotlib netcdf4 h5py h5netcdf xarray dask cartopy deprecation xmltodict semver"
NOTEBOOK_DEPS="notebook nbconvert psutil tqdm wetterdienst"
MISC_DEPS="coverage codecov pytest pytest-cov pytest-xdist pytest-sugar"

# Install twine for pypi upload
if [[ "$DEPLOY" == "true" ]]; then
    MISC_DEPS="$MISC_DEPS twine"
fi

# Install wradlib-data if not set
if [ -z "${WRADLIB_DATA+x}" ]; then
    git clone https://github.com/wradlib/wradlib-data.git $WRADLIB_BUILD_DIR/wradlib-data
    export WRADLIB_DATA=$WRADLIB_BUILD_DIR/wradlib-data
fi

# Install wradlib-notebooks if $WRADLIB_NOTEBOOKTEST is set
if [ -n "${WRADLIB_NOTEBOOKTEST+x}" ]; then
    git clone --depth=1 https://github.com/wradlib/wradlib-notebooks.git $WRADLIB_BUILD_DIR/wradlib-notebooks
    export WRADLIB_NOTEBOOKS=$WRADLIB_BUILD_DIR/wradlib-notebooks
    WRADLIB_DEPS="$WRADLIB_DEPS $NOTEBOOK_DEPS"
fi

# Create environment with the correct Python version and the needed dependencies
echo $WRADLIB_DEPS
echo $MISC_DEPS
micromamba create --yes --strict-channel-priority --name $WRADLIB_ENV python=$WRADLIB_PYTHON pip $WRADLIB_DEPS $MISC_DEPS --channel conda-forge
micromamba activate $WRADLIB_ENV

# create sdist (needed to create version.py)
python setup.py sdist
# install wradlib
python -m pip install . --no-deps --ignore-installed --no-cache-dir

# print some stuff
python --version
pip --version

python -c "import numpy; print(numpy.__version__)"
python -c "import numpy; print(numpy.__path__)"
