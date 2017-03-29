#!/bin/bash
# Copyright (c) 2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

exit_status=0

if [[ "$COVERAGE" == "true" ]]; then
    #export COVERAGE_PROCESS_START=".coveragerc"
    coverage run --source wradlib testrunner.py -u
    (( exit_status = ($? || $exit_status) ))
    coverage run --source wradlib testrunner.py -d
    (( exit_status = ($? || $exit_status) ))
    coverage run --source wradlib testrunner.py -e
    (( exit_status = ($? || $exit_status) ))
    coverage run --source wradlib testrunner.py -n
    (( exit_status = ($? || $exit_status) ))
    coverage combine

else
    python testrunner.py -u
    python testrunner.py -d
    python testrunner.py -e
    python testrunner.py -n
fi

exit $exit_status
