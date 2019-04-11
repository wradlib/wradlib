#!/bin/bash
# Copyright (c) 2011-2019, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

set -e

module=$1
method=$2

python -c "import $module ; import cProfile ; cProfile.run('${module}.${method}','restats')"
python -c "import pstats; pstats.Stats('restats').sort_stats('cumtime').print_stats(50)"
rm restats
