#!/bin/bash
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

set -e

# print the vars
echo "TRAVIS_PULL_REQUEST " $TRAVIS_PULL_REQUEST
echo "TRAVIS_SECURE_ENV_VARS " $TRAVIS_SECURE_ENV_VARS
echo "TRAVIS_TAG " $TRAVIS_TAG ${TRAVIS_TAG:1}

# get python major version
PY_MAJOR="${PYTHON_VERSION%.*}"

# download and install latest MinicondaX
wget http://repo.continuum.io/miniconda/Miniconda$PY_MAJOR-latest-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH

# Add conda-forge channel
conda config --add channels conda-forge
conda update --yes conda

# Create environment with the correct Python version
conda create -n wradlib --yes pip python=$PYTHON_VERSION
source activate wradlib

# Install wradlib dependencies
conda install --yes gdal numpy scipy matplotlib netcdf4 h5py deprecation xmltodict coverage codecov nose
conda list

# Install wradlib-data
git clone https://github.com/wradlib/wradlib-data.git $HOME/wradlib-data

# Install twine for pypi upload
if [[ "$DEPLOY" == "true" ]]; then
    conda install --yes twine
fi

# Install wradlib
python setup.py install

# print some stuff
python --version
pip --version

python -c "import numpy; print(numpy.__version__)"
python -c "import numpy; print(numpy.__path__)"
