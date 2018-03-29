#!/bin/bash
# Copyright (c) 2011-2018, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

set -e

# test if `conda` is available
if ! [ -x "$(command -v conda)" ]; then
    # if conda is not available
    # test if we are in CI
    if [ -z ${TRAVIS+x} ]; then
        # if local print error and return
        echo "Error: conda is not available in PATH."
        return
    else
        # if in CI install conda
        # print the travis CI vars
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
        WRADLIB_ENV="travis_wradlib"
        WRADLIB_PYTHON=$PYTHON_VERSION
    fi
else
    # check if envname parameter is available
    if [ -z ${@+x} ]; then
        # if not available, print error and return
        echo "Error: Please specify a new conda environment as parameter."
        echo "  usage: . scripts/install.sh newenv"
        return
    else
        # export env, if available
        WRADLIB_ENV=$@
    fi
    # export build dir
    export WRADLIB_BUILD_DIR="$( pwd )"
    # export current python version
    WRADLIB_PYTHON=`python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";`
fi

echo $PATH
echo $WRADLIB_ENV
echo $WRADLIB_PYTHON
echo "conda create -n $WRADLIB_ENV --yes pip python=$WRADLIB_PYTHON"

# Add conda-forge channel
conda config --add channels conda-forge
if [ ! -z ${CONDA_DEFAULT_ENV+x} ]; then
    source deactivate
fi
conda update --yes conda

# Create environment with the correct Python version
conda create -n $WRADLIB_ENV --yes pip python=$WRADLIB_PYTHON
source activate $WRADLIB_ENV

# Install wradlib dependencies
conda install --yes gdal numpy scipy matplotlib netcdf4 h5py deprecation xmltodict coverage codecov nose nose-timer
conda list

# Install wradlib-data if not set
if [ -z ${WRADLIB_DATA+x} ]; then
    git clone https://github.com/wradlib/wradlib-data.git $WRADLIB_BUILD_DIR/wradlib-data
    export WRADLIB_DATA=$WRADLIB_BUILD_DIR/wradlib-data
fi

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
