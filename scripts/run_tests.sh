#!/bin/bash
# Copyright (c) 2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

exit_status=0

if [[ "$COVERAGE" == "true" ]]; then
    #export COVERAGE_PROCESS_START=".coveragerc"
    coverage run --source wradlib testrunner.py -u -s
    (( exit_status = ($? || $exit_status) ))
    coverage run --source wradlib testrunner.py -d -s
    (( exit_status = ($? || $exit_status) ))
    coverage run --source wradlib testrunner.py -e -s
    (( exit_status = ($? || $exit_status) ))
    coverage run --source wradlib testrunner.py -n -s
    (( exit_status = ($? || $exit_status) ))
    coverage combine

else
    python testrunner.py -u -s
    python testrunner.py -d -s
    python testrunner.py -e -s
    python testrunner.py -n -s
fi

exit $exit_status
