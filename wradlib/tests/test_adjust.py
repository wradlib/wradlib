#!/usr/bin/env python
# -------------------------------------------------------------------------------
# Name:        test_adjust.py
# Purpose:     testing file for the wradlib.adjust module
#
# Authors:     wradlib developers
#
# Created:     26.02.2016
# Copyright:   (c) wradlib developers
# Licence:     The MIT License
# -------------------------------------------------------------------------------

import unittest

import wradlib.adjust as adjust
import numpy as np

class AdjustBaseTest(unittest.TestCase):
    def test___init__(self):
        pass

    def test__checkip(self):
        pass

    def test__check_shape(self):
        pass

    def test___call__(self):
        pass

    def test__get_valid_pairs(self):
        pass

    def test_xvalidate(self):
        pass


class AdjustAddTest(unittest.TestCase):
    def test___call__(self):
        pass


class AdjustMultiplyTest(unittest.TestCase):
    def test___call__(self):
        pass


class AdjustMixedTest(unittest.TestCase):
    def test___call__(self):
        pass


class AdjustMFBTest(unittest.TestCase):
    def test___call__(self):
        pass


class AdjustNoneTest(unittest.TestCase):
    def test___call__(self):
        pass


class GageOnlyTest(unittest.TestCase):
    def test___call__(self):
        pass



class AdjustHelperTest(unittest.TestCase):
    def test__get_neighbours_ix(self):
        pass

    def test__get_neighbours(self):
        pass

    def test__get_statfunc(self):
        pass

    def test_best(self):
        pass


if __name__ == '__main__':
    unittest.main()
