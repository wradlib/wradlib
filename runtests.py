__author__ = 'k.muehlbauer'

import unittest
import doctest
import os
import glob



# files = glob.glob(root_dir + 'test_*.py')
# module_strings = [str[0:len(str)-3].replace('/','.') for str in files]
# #module_strings.replace('/', '.')
# print(module_strings)
# suites = [unittest.defaultTestLoader.loadTestsFromName(str) for str in module_strings]
# testSuite = unittest.TestSuite(suites)

testSuite = []


# Doctest
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




