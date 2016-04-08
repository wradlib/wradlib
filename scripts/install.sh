#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:
export PATH="$HOME/miniconda/bin:$PATH"
#export PATH=/home/travis/miniconda2/bin:$PATH
#export PATH=/home/k.muehlbauer/miniconda2/bin:$PATH
#conda update --yes conda
conda update --yes conda

conda info -a

# Create a testenv with the correct Python version
conda create -n wradlib --yes pip python=$PYTHON_VERSION
#conda create -n wradlib --yes pip python=2.7
source activate wradlib

# add conda-forge channel
conda config --add channels conda-forge

# Install wradlib dependencies
conda install --yes gdal numpy scipy matplotlib netcdf4 h5py==2.5.0
ls -lart $HOME/miniconda/envs/wradlib/share/gdal

# install wradlib docu dependencies
conda install --yes sphinx numpydoc
conda install --yes sphinx_rtd_theme
pip install sphinxcontrib-bibtex
# install notebook dependencies
conda install --yes notebook runipy pandoc
pip install nbsphinx

# install optional wradlib dependencies
pip install xmltodict

# install flake8 PEP checker
pip install flake8

# install coverage modules
pip install coverage
if [[ "$COVERALLS" == "true" ]]; then
    pip install python-coveralls
fi

python setup.py install
