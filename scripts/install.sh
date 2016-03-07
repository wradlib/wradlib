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

# Install wradlib dependencies
conda install -c https://conda.anaconda.org/anaconda --yes numpy scipy matplotlib netcdf4 proj4

conda install -c https://conda.anaconda.org/anaconda --yes sphinx numpydoc h5py geos krb5
conda install -c https://conda.anaconda.org/osgeo --yes gdal kealib
ls -lart /home/travis/miniconda2/envs/wradlib/share/gdal
conda install --yes sphinx_rtd_theme
pip install sphinxcontrib-bibtex
pip install xmltodict

# install coverage modules
pip install coverage
if [[ "$COVERALLS" == "true" ]]; then
    pip install python-coveralls
fi

python setup.py install
