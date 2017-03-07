#!/bin/bash
# Copyright (c) 2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.


if [[ "$COVERALLS" == "true" ]]; then
    xvfb-run -a -w 5 coverage run --source wradlib testrunner.py -u
    xvfb-run -a -w 5 coverage run -a --source wradlib testrunner.py -d
    xvfb-run -a -w 5 coverage run -a --source wradlib testrunner.py -e
    xvfb-run -a -w 5 coverage run -a --source wradlib testrunner.py -n
else
    xvfb-run -a -w 5 python testrunner.py -u
    xvfb-run -a -w 5 python testrunner.py -d
    xvfb-run -a -w 5 python testrunner.py -e
    xvfb-run -a -w 5 python testrunner.py -n
fi