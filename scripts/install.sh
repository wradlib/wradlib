#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH
conda update --yes conda
conda update --yes conda

# Create a testenv with the correct Python version
conda create -n wradlib --yes pip python=$PYTHON_VERSION
source activate wradlib

# add conda-forge channel
conda config --add channels conda-forge

# Install wradlib dependencies
conda install --yes gdal numpy scipy matplotlib netcdf4 h5py
ls -lart $HOME/miniconda/envs/wradlib/share/gdal

# install wradlib-data
git clone https://github.com/wradlib/wradlib-data.git $HOME/wradlib-data
echo $PWD
ls -lart $HOME
ls -lart $HOME/wradlib-data

# install wradlib docu dependencies
conda install --yes sphinx numpydoc
conda install --yes sphinx_rtd_theme
pip install sphinxcontrib-bibtex
# install notebook dependencies
conda install --yes notebook runipy pandoc

# install nbsphinx
git clone https://github.com/spatialaudio/nbsphinx.git $HOME/nbsphinx
pip install -e $HOME/nbsphinx

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

# print some stuff
python --version
pip --version

python -c "import numpy; print(numpy.__version__)"
python -c "import numpy; print(numpy.__path__)"

python -c "import runipy; print(runipy.__version__)"
python -c "import runipy; print(runipy.__path__)"

