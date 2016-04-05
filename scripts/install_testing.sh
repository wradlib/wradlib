#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/miniconda2/bin:$PATH
conda update --yes conda
conda update --yes conda

# Create a testenv with the correct Python version
conda create -n wradlib --yes pip python=$PYTHON_VERSION
source activate wradlib

# Install wradlib dependencies
conda install -c https://conda.anaconda.org/anaconda --yes numpy scipy matplotlib netcdf4 proj4
conda install -c https://conda.anaconda.org/anaconda --yes gdal h5py geos

# install coverage modules
pip install coverage
if [[ "$COVERALLS" == "true" ]]; then
    pip install python-coveralls
fi

python setup.py install
