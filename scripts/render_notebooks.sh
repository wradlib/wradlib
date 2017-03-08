#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

# duplicate folder
mkdir notebooks-render
cp -R notebooks notebooks-render/

cd notebooks-render
# get notebooks list
notebooks=`find notebooks -path notebooks/*.ipynb_checkpoints -prune -o -name *.ipynb -print`
echo $notebooks

# render notebooks to doc/sources
for nb in $notebooks; do
    echo "runipy --quiet --overwrite --matplotlib $nb"
    runipy --quiet --overwrite --matplotlib $nb
    cp --parents $nb ../doc/source/
done

# copy images to docs too
images=`find notebooks -path notebooks/*.ipynb_checkpoints -prune -o -name *.png -print`
echo $images
for im in $images; do
    cp --parents $im ../doc/source/
done