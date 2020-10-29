#!/usr/bin/env bash
# Copyright (c) 2011-2020, wradlib developers.
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

        # download latest micromamba
        wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba --strip-components=1
        ./micromamba shell init -s bash -p ~/micromamba
        cat ~/.bashrc
        export MAMBA_ROOT_PREFIX=~/micromamba
        export MAMBA_EXE=$(pwd)/micromamba
        . ${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh
        echo "micromamba version $(micromamba --version)"
        #micromamba create --yes --name MAMBA mamba
        #micromamba install -y mamba -c conda-forge

#        # download and install latest MinicondaX
#        wget http://repo.continuum.io/miniconda/Miniconda$PY_MAJOR-latest-Linux-x86_64.sh \
#            -O miniconda.sh
#        chmod +x miniconda.sh
#        bash miniconda.sh -b -p $HOME/miniconda
#        export PATH=$HOME/miniconda/bin:$PATH
#        source $HOME/miniconda/etc/profile.d/conda.sh

        WRADLIB_ENV="travis_wradlib"
        WRADLIB_PYTHON=$PYTHON_VERSION

        micromamba create --yes --name $WRADLIB_ENV python=$WRADLIB_PYTHON mamba --channel conda-forge
        micromamba activate $WRADLIB_ENV

        # special packages directory for caching in travis-ci
        # remove temprorarily, it seems it's faster without caching
        # conda config --add pkgs_dirs $HOME/condacache/pkgs
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
echo $GDAL_VERSION
#echo "mamba create -n $WRADLIB_ENV --yes pip python=$WRADLIB_PYTHON"

#if [ ! -z ${CONDA_DEFAULT_ENV+x} ]; then
#    mamba deactivate
#fi

# Add conda-forge channel
#conda config --add channels conda-forge
# Set strict channel priority
#conda config --set channel_priority strict

#mamba update --yes conda

# activate conda for current bash
#export PATH=$HOME/miniconda/bin:$PATH
#source $HOME/miniconda/etc/profile.d/conda.sh

# Install wradlib dependencies
WRADLIB_DEPS="gdal=$GDAL_VERSION numpy scipy matplotlib netcdf4 h5py h5netcdf xarray dask cartopy deprecation xmltodict semver"
NOTEBOOK_DEPS="notebook nbconvert psutil tqdm"
MISC_DEPS="coverage codecov pytest pytest-cov pytest-xdist pytest-sugar"

# Install twine for pypi upload
if [[ "$DEPLOY" == "true" ]]; then
    MISC_DEPS="$MISC_DEPS twine"
fi

# Install wradlib-data if not set
if [ -z ${WRADLIB_DATA+x} ]; then
    git clone https://github.com/wradlib/wradlib-data.git $WRADLIB_BUILD_DIR/wradlib-data
    export WRADLIB_DATA=$WRADLIB_BUILD_DIR/wradlib-data
fi

# Install WRADLIB_NOTEBOOKTEST is set
if [ ! -z ${WRADLIB_NOTEBOOKTEST+x} ]; then
    git clone --depth=1 https://github.com/wradlib/wradlib-notebooks.git $WRADLIB_BUILD_DIR/wradlib-notebooks
    export WRADLIB_NOTEBOOKS=$WRADLIB_BUILD_DIR/wradlib-notebooks
    WRADLIB_DEPS="$WRADLIB_DEPS $NOTEBOOK_DEPS"
fi

# Create environment with the correct Python version and the needed dependencies
echo $WRADLIB_DEPS
echo $MISC_DEPS
#mamba create -n $WRADLIB_ENV --yes pip python=$WRADLIB_PYTHON $WRADLIB_DEPS $MISC_DEPS
mamba install --yes --strict-channel-priority --channel conda-forge pip $WRADLIB_DEPS $MISC_DEPS
#conda activate $WRADLIB_ENV

# Install wradlib
python setup.py sdist
python -m pip install . --no-deps --ignore-installed --no-cache-dir

# print some stuff
python --version
pip --version

python -c "import numpy; print(numpy.__version__)"
python -c "import numpy; print(numpy.__path__)"
