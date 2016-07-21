#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

# get notebooks list
notebooks=`find notebooks -path notebooks/.ipynb_checkpoints -prune -o -name *.ipynb -print`
echo $notebooks

# convert notebooks to python scripts
for nb in $notebooks; do
    jupyter nbconvert --to script $nb
    # this is for testrunner.py finding the notebook- tests
    touch "${nb%/*}/__init__.py"
done
