#------------------------------------------------------------------------------
# Name:        runtest.py
# Purpose:     testrunner for unittest, doctest and examples test
#
# Author:      Kai Muehlbauer
#
# Created:     03.03.2014
# Copyright:   (c) Kai Muehlbauer 2014
# Licence:     The MIT License
#------------------------------------------------------------------------------

import unittest
import doctest
import os
import glob

testSuite = []

# examples test
root_dir = 'examples/'
files = []
skip = ['__init__.py', 'bufr.py', 'test_']
for root, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename in skip or filename[-3:] != '.py':
            continue
        if 'examples/data' in root:
            continue
        f = os.path.join(root, filename)
        f = f.replace('/', '.')
        f = f[:-3]
        files.append(f)
import inspect
suite = unittest.TestSuite()
for module in files:
    module1, func = module.split('.')
    module = __import__(module)
    func = getattr(module, func)
    funcs = inspect.getmembers(func, inspect.isfunction)
    [suite.addTest(unittest.FunctionTestCase(v)) for k,v in funcs if k.startswith("ex_") ]

testSuite.append(unittest.TestSuite(suite))

# doctest
root_dir = 'wradlib/'
files = []
skip = ['__init__.py', 'bufr.py', 'test_']

for root, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename in skip or filename[-3:] != '.py':
            continue
        if 'wradlib/tests' in root:
            continue
        f = os.path.join(root, filename)
        f = f.replace('/', '.')
        f = f[:-3]
        files.append(f)

suite = unittest.TestSuite()
for module in files:
    print(module, type(module))
    suite.addTest(doctest.DocTestSuite(module))
testSuite.append(unittest.TestSuite(suite))

# unittest
files = []
skip = ['__init__.py']
root_dir = 'wradlib/tests/'
for root, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename in skip or filename[-3:] != '.py':
            continue
        f = os.path.join(root, filename)
        f = f.replace('/', '.')
        f = f[:-3]
        files.append(f)
suite = [unittest.defaultTestLoader.loadTestsFromName(str) for str in files]
testSuite.append(unittest.TestSuite(suite))

for ts in testSuite:
    unittest.TextTestRunner(verbosity=2).run(ts)




