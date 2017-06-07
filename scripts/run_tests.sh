#!/bin/bash
# Copyright (c) 2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

exit_status=0

# export location of .coveragerc
if [[ "$COVERAGE" == "true" ]]; then
    export COVERAGE_PROCESS_START=$WRADLIB_BUILD_DIR/.coveragerc
    # run tests, retrieve exit status
    ./testrunner.py -u -c -s
    (( exit_status = ($? || $exit_status) ))
    ./testrunner.py -d -c -s
    (( exit_status = ($? || $exit_status) ))
    ./testrunner.py -e -c -s
    (( exit_status = ($? || $exit_status) ))
    ./testrunner.py -n -c -s
    (( exit_status = ($? || $exit_status) ))

    # combine coverage
    coverage combine

else
    # run tests, retrieve exit status
    ./testrunner.py -u -s
    (( exit_status = ($? || $exit_status) ))
    ./testrunner.py -d -s
    (( exit_status = ($? || $exit_status) ))
    ./testrunner.py -e -s
    (( exit_status = ($? || $exit_status) ))
    ./testrunner.py -n -s
    (( exit_status = ($? || $exit_status) ))
fi

exit $exit_status
