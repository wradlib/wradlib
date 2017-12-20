#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

set -e

# print the vars
echo "TRAVIS_PULL_REQUEST " $TRAVIS_PULL_REQUEST
echo "TRAVIS_SECURE_ENV_VARS " $TRAVIS_SECURE_ENV_VARS
echo "TRAVIS_TAG " $TRAVIS_TAG ${TRAVIS_TAG:1}

export PING_SLEEP=30s
export WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BUILD_OUTPUT=$WORKDIR/build.out

touch $BUILD_OUTPUT

dump_output() {
   echo Tailing the last 500 lines of output:
   tail -500 $BUILD_OUTPUT
}
error_handler() {
  echo ERROR: An error was encountered with the build.
  dump_output
  exit 1
}

# If an error occurs, run our error handler to output a tail of the build.
trap 'error_handler' ERR

wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
    -O miniconda.sh
chmod +x miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH

# Add conda-forge channel
conda config --add channels conda-forge

conda update --yes conda >> $BUILD_OUTPUT
conda update --yes conda >> $BUILD_OUTPUT

# Create a testenv with the correct Python version
conda create -n wradlib --yes pip python=$PYTHON_VERSION >> $BUILD_OUTPUT
source activate wradlib

# Install wradlib dependencies
conda install --yes gdal numpy scipy matplotlib netcdf4 h5py >> $BUILD_OUTPUT
conda list
ls -lart $HOME/miniconda/envs/wradlib/share/gdal

# Install optional wradlib dependencies
conda install --yes xmltodict >> $BUILD_OUTPUT

# Install wradlib-data
git clone https://github.com/wradlib/wradlib-data.git $HOME/wradlib-data >> $BUILD_OUTPUT
echo $PWD
ls -lart $HOME
ls -lart $HOME/wradlib-data

# Install nbconvert
conda install --yes notebook nbconvert >> $BUILD_OUTPUT

# Install wradlib docu dependencies
if [[ "$DOC_BUILD" == "true" ]]; then
    conda install --yes sphinx numpydoc sphinx_rtd_theme runipy pandoc nbsphinx >> $BUILD_OUTPUT
    pip install sphinxcontrib-bibtex >> $BUILD_OUTPUT
fi

# Install flake8 PEP checker
conda install --yes flake8 >> $BUILD_OUTPUT

# Install coverage modules
if [[ "$COVERAGE" == "true" ]]; then
    conda install --yes coverage >> $BUILD_OUTPUT
    pip install codecov >> $BUILD_OUTPUT
fi

python setup.py install

# print some stuff
python --version
pip --version

python -c "import numpy; print(numpy.__version__)"
python -c "import numpy; print(numpy.__path__)"

# The build finished without returning an error so dump a tail of the output.
dump_output

# Nicely terminate the ping output loop.
kill $PING_LOOP_PID
rm -rf $BUILD_OUTPUT

exit 0
