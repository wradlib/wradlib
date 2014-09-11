# build documentation
sphinx-build -b html doc/source ~/src/bitbucket.org/wradlibcodeship/wradlib.bitbucket.org 2>&1 | tee sphinx.txt
cp *.txt ~/src/bitbucket.org/wradlibcodeship/wiki
# change to documentation repo
cd ~/src/bitbucket.org/wradlibcodeship/wradlib.bitbucket.org
# remove .doctrees folder from
hg addremove -X .doctrees/
