#!/bin/bash

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

# add conda-forge channel
conda config --add channels conda-forge

# Install wradlib dependencies
conda install --yes gdal numpy scipy matplotlib netcdf4 h5py==2.5.0
ls -lart /home/travis/miniconda2/envs/wradlib/share/gdal

# install wradlib docu dependencies
conda install --yes sphinx numpydoc
conda install --yes sphinx_rtd_theme
pip install sphinxcontrib-bibtex

# install optional wradlib dependencies
pip install xmltodict

# install fkale8 PEP checker
pip install flake8

# install coverage modules
pip install coverage
if [[ "$COVERALLS" == "true" ]]; then
    pip install python-coveralls
fi

python setup.py install
