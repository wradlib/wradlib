#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2020, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import doctest
import getopt
import inspect
import io
import os
import sys
import unittest
from multiprocessing import Process, Queue

import coverage
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

VERBOSE = 2


def create_examples_testsuite():
    # gather information on examples
    # all functions inside the examples starting with 'ex_' or 'recipe_'
    # are considered as tests
    # find example files in examples directory
    root_dir = "examples/"
    files = []
    skip = ["__init__.py"]
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in skip or filename[-3:] != ".py":
                continue
            if "examples/data" in root:
                continue
            f = os.path.join(root, filename)
            f = f.replace("/", ".")
            f = f[:-3]
            files.append(f)
    # create empty testsuite
    suite = unittest.TestSuite()
    # find matching functions in
    for idx, module in enumerate(files):
        module1, func = module.split(".")
        module = __import__(module)
        func = getattr(module, func)
        funcs = inspect.getmembers(func, inspect.isfunction)
        [
            suite.addTest(unittest.FunctionTestCase(v))
            for k, v in funcs
            if k.startswith(("ex_", "recipe_"))
        ]

    return suite


class NotebookTest(unittest.TestCase):
    def __init__(self, nbfile, cov):
        setattr(self.__class__, nbfile, staticmethod(self._runTest))
        super(NotebookTest, self).__init__(nbfile)
        self.nbfile = nbfile
        self.cov = cov

    def _runTest(self):
        kernel = "python%d" % sys.version_info[0]
        cur_dir = os.path.dirname(self.nbfile)

        with open(self.nbfile) as f:
            nb = nbformat.read(f, as_version=4)
            if self.cov:
                covdict = {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {"collapsed": True},
                    "outputs": [],
                    "nbsphinx": "hidden",
                    "source": "import coverage\n"
                    "coverage.process_startup()\n"
                    "import sys\n"
                    'sys.path.append("{0}")\n'.format(cur_dir),
                }
                nb["cells"].insert(0, nbformat.from_dict(covdict))

            exproc = ExecutePreprocessor(kernel_name=kernel, timeout=600)

            try:
                run_dir = os.getenv("WRADLIB_BUILD_DIR", cur_dir)
                exproc.preprocess(nb, {"metadata": {"path": run_dir}})
            except CellExecutionError as e:
                raise e

        if self.cov:
            nb["cells"].pop(0)

        with io.open(self.nbfile, "wt") as f:
            nbformat.write(nb, f)

        self.assertTrue(True)


def create_notebooks_testsuite(**kwargs):
    # gather information on notebooks
    # all notebooks in the notebooks folder
    # are considered as tests
    # find notebook files in notebooks directory
    cov = kwargs.pop("cov")
    root_dir = os.getenv("WRADLIB_NOTEBOOKS", "notebooks")
    files = []
    skip = []
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in skip or filename[-6:] != ".ipynb":
                continue
            # skip checkpoints
            if "/." in root:
                continue
            f = os.path.join(root, filename)
            files.append(f)

    # create one TestSuite per Notebook to treat testrunners
    # memory overconsumption on travis-ci
    suites = []
    for file in files:
        suite = unittest.TestSuite()
        suite.addTest(NotebookTest(file, cov))
        suites.append(suite)

    return suites


def create_doctest_testsuite():
    # gather information on doctests, search in only wradlib folder
    root_dir = "wradlib/"
    files = []
    skip = ["__init__.py", "version.py", "bufr.py", "test_"]
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in skip or filename[-3:] != ".py":
                continue
            if "wradlib/tests" in root:
                continue
            f = os.path.join(root, filename)
            f = f.replace("/", ".")
            f = f[:-3]
            files.append(f)

    # put modules in doctest suite
    suite = unittest.TestSuite()
    for module in files:
        suite.addTest(doctest.DocTestSuite(module))
    return suite


def create_unittest_testsuite():
    # gather information on tests (unittest etc)
    root_dir = "wradlib/tests/"
    return unittest.defaultTestLoader.discover(root_dir)


def single_suite_process(queue, test, verbosity, **kwargs):
    test_cov = kwargs.pop("coverage", 0)
    test_nb = kwargs.pop("notebooks", 0)
    if test_cov and not test_nb:
        cov = coverage.coverage()
        cov.start()
    all_success = 1
    for ts in test:
        if ts.countTestCases() != 0:
            res = unittest.TextTestRunner(verbosity=verbosity).run(ts)
            all_success = all_success & res.wasSuccessful()
    if test_cov and not test_nb:
        cov.stop()
        cov.save()
    queue.put(all_success)


def keep_tests(suite, arg):
    newsuite = unittest.TestSuite()
    try:
        for tc in suite:
            try:
                if tc.id().find(arg) != -1:
                    newsuite.addTest(tc)
            except AttributeError:
                new = keep_tests(tc, arg)
                if new.countTestCases() != 0:
                    newsuite.addTest(new)
    except TypeError:
        pass
    return newsuite


def main():

    args = sys.argv[1:]

    usage_message = """Usage: python testrunner.py options arg

    If run without options, testrunner displays the usage message.
    If all tests suites should be run, use the -a option.
    If arg is given, only tests containing arg are run.

    options:

        -a
        --all
            Run all tests (examples, test, doctest, notebooks)

        -m
            Run all tests within a single testsuite [default]

        -M
            Run each suite as separate instance

        -e
        --example
            Run only examples tests

        -d
        --doc
            Run only doctests

        -u
        --unit
            Run only unit test

        -n
        --notebook
            Run only notebook test

        -s
        --use-subprocess
            Run every testsuite in a subprocess.

        -c
        --coverage
            Run notebook tests with code coverage

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
    test_notebooks = 0
    test_units = 0
    test_subprocess = 0
    test_cov = 0
    verbosity = VERBOSE

    try:
        options, arg = getopt.getopt(
            args,
            "aednuschv:",
            [
                "all",
                "example",
                "doc",
                "notebook",
                "unit",
                "use-subprocess",
                "coverage",
                "help",
            ],
        )
    except getopt.GetoptError as e:
        err_exit(e.msg)

    if not options:
        err_exit(usage_message)
    for name, value in options:
        if name in ("-a", "--all"):
            test_all = 1
        elif name in ("-e", "--example"):
            test_examples = 1
        elif name in ("-d", "--doc"):
            test_docs = 1
        elif name in ("-n", "--notebook"):
            test_notebooks = 1
        elif name in ("-u", "--unit"):
            test_units = 1
        elif name in ("-s", "--use-subprocess"):
            test_subprocess = 1
        elif name in ("-c", "--coverage"):
            test_cov = 1
        elif name in ("-h", "--help"):
            err_exit(usage_message, 0)
        elif name == "-v":
            verbosity = int(value)
        else:
            err_exit(usage_message)

    if not (test_all or test_examples or test_docs or test_notebooks or test_units):
        err_exit("must specify one of: -a -e -d -n -u")

    testSuite = []

    if test_all:
        testSuite.append(create_examples_testsuite())
        testSuite.append(create_notebooks_testsuite(cov=test_cov))
        testSuite.append(create_doctest_testsuite())
        testSuite.append(create_unittest_testsuite())
    elif test_examples:
        testSuite.append(create_examples_testsuite())
    elif test_notebooks:
        testSuite.append(create_notebooks_testsuite(cov=test_cov))
    elif test_docs:
        testSuite.append(unittest.TestSuite(create_doctest_testsuite()))
    elif test_units:
        testSuite.append(create_unittest_testsuite())

    all_success = 1
    if test_subprocess:
        for test in testSuite:
            if arg:
                test = keep_tests(test, arg[0])
            queue = Queue()
            keywords = {"coverage": test_cov, "notebooks": test_notebooks}
            proc = Process(
                target=single_suite_process,
                args=(queue, test, verbosity),
                kwargs=keywords,
            )
            proc.start()
            result = queue.get()
            proc.join()
            # all_success should be 0 in the end
            all_success = all_success & result
    else:
        if test_cov and not test_notebooks:
            cov = coverage.coverage()
            cov.start()
        for ts in testSuite:
            if arg:
                ts = keep_tests(ts, arg[0])
            for test in ts:
                if test.countTestCases() != 0:
                    result = unittest.TextTestRunner(verbosity=verbosity).run(test)
                    # all_success should be 0 in the end
                    all_success = all_success & result.wasSuccessful()
        if test_cov and not test_notebooks:
            cov.stop()
            cov.save()

    if all_success:
        sys.exit(0)
    else:
        # This will return exit code 1
        sys.exit("At least one test has failed. " "Please see test report for details.")


def err_exit(message, rc=2):
    sys.stderr.write("\n%s\n" % message)
    sys.exit(rc)


if __name__ == "__main__":
    main()
