#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

# get notebooks list
notebooks=`find notebooks -path notebooks/.ipynb_checkpoints -prune -o -name *.ipynb -print`
echo $notebooks

# copy notebooks to doc/sources
for nb in $notebooks; do
    echo "cp --parents $nb doc/source/"
    cp --parents $nb doc/source/
done
