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

import sys
import os
import getopt
import unittest
import doctest

import glob
import inspect

# This import has to be done in order to return correct exit codes
# see http://stackoverflow.com/questions/14354496/python-returns-wrong-exit-code
#    for details on this weird fix
#import scipy.weave

VERBOSE = 2

def create_examples_testsuite():
    # gather information on examples
    # all functions inside the examples starting with 'ex_' or 'recipe_' are considered as tests
    # find example files in examples directory
    root_dir = 'examples/'
    files = []
    skip = ['__init__.py']
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
    # create empty testsuite
    suite = unittest.TestSuite()
    # find matching functions in
    for idx, module in enumerate(files):
        module1, func = module.split('.')
        module = __import__(module)
        func = getattr(module, func)
        funcs = inspect.getmembers(func, inspect.isfunction)
        [suite.addTest(unittest.FunctionTestCase(v)) for k,v in funcs if k.startswith(("ex_", "recipe_")) ]
    return suite

#testSuite.append(unittest.TestSuite(suite))

def create_doctest_testsuite():
    # gather information on doctests, search in only wradlib folder
    root_dir = 'wradlib/'
    files = []
    skip = ['__init__.py', 'version.py', 'bufr.py', 'test_']
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

    # put modules in doctest suite
    suite = unittest.TestSuite()
    for module in files:
        suite.addTest(doctest.DocTestSuite(module))
    return suite

#testSuite.append(unittest.TestSuite(suite))

def create_unittest_testsuite():
    # gather information on tests (unittest etc)
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
            print(f)
    suite = [unittest.defaultTestLoader.loadTestsFromName(str) for str in files]
    return suite

def main(args):
    usage_message = """Usage: python testrunner.py options

    If run without options, testrunner displays the usage message. If all tests
    suites should be run,, use the -a option.

    options:

        -a
        --all
            Run all tests (examples, test, doctest)

        -m
            Run all tests within a single testsuite [default]

        -M
            Run each suite as separate instance

        -e
        --examples
            Run only examples tests

        -d
        --doc
            Run only doctests

        -u
        --unit
            Run only unit test

        -v level
            Set the level of verbosity.

            0 - Silent
            1 - Quiet (produces a dot for each succesful test)
            2 - Verbose (default - produces a line of output for each test)

        -h
            Display usage information.

    """

    test_all = 0
    test_examples = 0
    test_docs = 0
    test_units = 0
    verbosity = VERBOSE


    try:
        options, arg = getopt.getopt(args, 'aeduhv:', ['all','examples', 'docs', 'units', 'help'])
    except getopt.GetoptError as e:
        err_exit(e.msg)

    if not options:
        err_exit(usage_message)
    for name, value in options:
        if name in ('-a', '--all'):
            test_all = 1
        elif name in ('-e', '--examples'):
            test_examples = 1
        elif name in ('-d', '--docs'):
            test_docs = 1
        elif name in ('-u', '--units'):
            test_units = 1
        elif name in ('-h', '--help'):
            err_exit(usage_message, 0)
        elif name == '-v':
            verbosity = int(value)
        else:
            err_exit(usage_message)

    if not(test_all or test_examples or test_docs or test_units):
        err_exit('must specify one of: -a -e -d -u')

    # change to main package path, where testrunner.py lives
    path = os.path.dirname(__file__)
    if path:
        os.chdir(path)

    testSuite = []

    if test_all:
        testSuite.append(unittest.TestSuite(create_examples_testsuite()))
        testSuite.append(unittest.TestSuite(create_doctest_testsuite()))
        testSuite.append(unittest.TestSuite(create_unittest_testsuite()))
    elif test_examples:
        testSuite.append(unittest.TestSuite(create_examples_testsuite()))
    elif test_docs:
        testSuite.append(unittest.TestSuite(create_doctest_testsuite()))
    elif test_units:
        testSuite.append(unittest.TestSuite(create_unittest_testsuite()))

    all_success = 1
    for ts in testSuite:
        result = unittest.TextTestRunner(verbosity=verbosity).run(ts)
        # if any test suite was not successful, all_success should be 0 in the end
        all_success = all_success & result.wasSuccessful()

    if all_success:
        sys.exit(0)
    else:
        # This will retrun exit code 1
        sys.exit("At least one test hase failed. Please see test report for details.")

def err_exit(message, rc=2):
    sys.stderr.write("\n%s\n" % message)
    sys.exit(rc)

if __name__ == '__main__':
    main(sys.argv[1:])


