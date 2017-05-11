#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

cd $WRADLIB_NOTEBOOKS
# get notebooks list
notebooks=`find notebooks -path notebooks/*.ipynb_checkpoints -prune -o -name *.ipynb -print`
echo $notebooks

# copy notebooks to doc/sources
for nb in $notebooks; do
    cp --parents $nb $TRAVIS_BUILD_DIR/doc/source/
done

# copy images to docs too
images=`find notebooks -path notebooks/*.ipynb_checkpoints -prune -o -name *.png -print`
echo $images
for im in $images; do
    cp --parents $im $TRAVIS_BUILD_DIR/doc/source/
done