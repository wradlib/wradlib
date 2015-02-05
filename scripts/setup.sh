# echo $CI_BRANCH

# install prerequisites
# numpy and gdal apparently don't install using the requirements file
# so for now we do it explicitly - this should be fixed some time
pip install numpy
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL==1.9.0
# install all other requirements
pip install -r requirements.txt
#hg update -r $CI_BRANCH
#hg update -r testing
# numpydoc for building docs
pip install numpydoc

# install wradlib into virtualenv
python setup.py install 2>&1 | tee setup_install.txt

# copy all result text files to the wiki working copy
cp *.txt ~/src/bitbucket.org/wradlibcodeship/wiki
