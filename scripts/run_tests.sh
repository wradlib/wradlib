#!/usr/bin/env bash
# Copyright (c) 2011-2023, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

set -e
pytest -n auto --dist loadfile --verbose --doctest-modules --doctest-plus --durations=15 --pyargs $1
