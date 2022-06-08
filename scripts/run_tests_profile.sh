#!/bin/bash
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.
# Example:run_tests_profile.sh "wradlib.tests.test_io" "XarrayTests().test_read_odim()"

set -e

module=$1
method=$2

python -c "import $module ; import cProfile ; cProfile.run('${module}.${method}','restats')"
python -c "import pstats; pstats.Stats('restats').sort_stats('cumtime').print_stats(50)"
rm restats
