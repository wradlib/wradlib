#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

# render notebooks to doc/sources
notebooks=`ls notebooks/*.ipynb`
echo $notebooks

for nb in $notebooks; do
    base=${nb##*/}
    echo "runipy --quiet --matplotlib $nb doc/source/$base"
    runipy --quiet --matplotlib $nb doc/source/$base
done



