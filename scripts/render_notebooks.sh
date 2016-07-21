#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

# get notebooks list
notebooks=`find notebooks -path notebooks/.ipynb_checkpoints -prune -o -name *.ipynb -print`
echo $notebooks

# render notebooks to doc/sources
for nb in $notebooks; do
    cp --parents $nb doc/source/
    echo "runipy --quiet --overwrite --matplotlib --matplotlib doc/source/$nb"
    runipy --quiet --overwrite --matplotlib doc/source/$nb
done
