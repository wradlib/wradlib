#!/bin/bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

# convert notebooks to python scripts
cd notebooks
jupyter nbconvert --to script *.ipynb
cd ..



