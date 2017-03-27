#!/bin/bash
# Copyright (c) 2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

Xvfb :99 &
export DISPLAY=:99

if [[ "$COVERAGE" == "true" ]]; then
    coverage run --source wradlib testrunner.py -u
    coverage run -a --source wradlib testrunner.py -d
    coverage run -a --source wradlib testrunner.py -e
    coverage run -a --source wradlib testrunner.py -n
else
    python testrunner.py -u
    python testrunner.py -d
    python testrunner.py -e
    python testrunner.py -n
fi