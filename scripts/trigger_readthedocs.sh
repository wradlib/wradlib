#!/bin/bash
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

set -e

if [ "$TRAVIS_PYTHON_VERSION" = "3.7" ] && [ "$TRAVIS_PULL_REQUEST" = "false" ] ; then
    curl -X POST -d "token=$RTD_TOKEN" $RTD_URL
fi
