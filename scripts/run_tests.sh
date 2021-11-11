#!/bin/bash
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

set -e

if [[ "$TRAVIS" == "true" ]]; then
    # export location of .coveragerc
    export COVERAGE_PROCESS_START=$WRADLIB_BUILD_DIR/.coveragerc
    coverage run -m pytest --verbose --doctest-modules --doctest-plus --durations=15 --pyargs $1
    coverage combine
    coverage xml
else
    pytest --verbose --doctest-modules --doctest-plus --durations=15 --pyargs $1
fi
