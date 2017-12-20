#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.


# print the vars
echo "TRAVIS_PULL_REQUEST " $TRAVIS_PULL_REQUEST
echo "TRAVIS_SECURE_ENV_VARS " $TRAVIS_SECURE_ENV_VARS
echo "TRAVIS_TAG " $TRAVIS_TAG ${TRAVIS_TAG:1}

wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH

# Add conda-forge channel
conda config --add channels conda-forge

conda update --yes conda
conda update --yes conda

# Create a testenv with the correct Python version
conda create -n wradlib --yes pip python=$PYTHON_VERSION
source activate wradlib

# Install wradlib dependencies
conda install --yes gdal numpy scipy matplotlib netcdf4 h5py
conda list
ls -lart $HOME/miniconda/envs/wradlib/share/gdal

# Install optional wradlib dependencies
conda install --yes xmltodict

# Install wradlib-data
git clone https://github.com/wradlib/wradlib-data.git $HOME/wradlib-data
echo $PWD
ls -lart $HOME
ls -lart $HOME/wradlib-data

# Install nbconvert
conda install --yes notebook nbconvert

# Install wradlib docu dependencies
if [[ "$DOC_BUILD" == "true" ]]; then
    conda install --yes sphinx numpydoc
    conda install --yes sphinx_rtd_theme
    pip install sphinxcontrib-bibtex
    # install notebook dependencies
    conda install --yes runipy pandoc
    # install nbsphinx
    conda install --yes nbsphinx
fi

# Install flake8 PEP checker
conda install --yes flake8

# Install coverage modules
if [[ "$COVERAGE" == "true" ]]; then
    conda install --yes coverage
    pip install codecov
fi

python setup.py install

# print some stuff
python --version
pip --version

python -c "import numpy; print(numpy.__version__)"
python -c "import numpy; print(numpy.__path__)"
