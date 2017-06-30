#!/bin/bash
# Copyright (c) 2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

# start headless rendering with xvfb
export DISPLAY=:99
Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp &
xvfb=$!
echo "Xvfb PID:" $xvfb

exit_status=0

# export location of .coveragerc
if [[ "$COVERAGE" == "true" ]]; then
    export COVERAGE_PROCESS_START=$WRADLIB_BUILD_DIR/.coveragerc
    # run tests, retrieve exit status
    ./testrunner.py -u -c -s
    (( exit_status = ($? || $exit_status) ))
    coverage combine
    coverage xml -o coverage-unittests.xml
    ./testrunner.py -d -c -s
    (( exit_status = ($? || $exit_status) ))
    coverage combine
    coverage xml -o coverage-doctests.xml
    ./testrunner.py -e -c -s
    (( exit_status = ($? || $exit_status) ))
    coverage combine
    coverage xml -o coverage-exampletests.xml
    ./testrunner.py -n -c -s
    (( exit_status = ($? || $exit_status) ))
    coverage combine
    coverage xml -o coverage-notebooktests.xml

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

# shutdown xvfb
kill -15 $xvfb
wait $xvfb

exit $exit_status
