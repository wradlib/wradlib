import os
import sys

# if setuptools not present bootstrap it
try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup

# import pkg_resources for version checking
from pkg_resources import get_distribution, parse_version


def query_yes_quit(question, default="quit"):
    """Ask a yes/quit question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" or "quit" (the default).

    The "answer" return value is one of "yes" or "quit".
    """
    valid = {"yes":"yes",   "y":"yes",    "ye":"yes",
             "quit":"quit", "qui":"quit", "qu":"quit", "q":"quit"}
    if default == None:
        prompt = " [y/q] "
    elif default == "yes":
        prompt = " [Y/q] "
    elif default == "quit":
        prompt = " [y/Q] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while 1:
        sys.stdout.write(question + prompt)
        sys.stdout.flush()
        rinput = vars(__builtins__).get('raw_input',input)
        choice = rinput().lower()
        if default is not None and choice == '':
            return default
        elif choice in valid.keys():
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'quit'.\n")
            sys.stdout.flush()

# package requires (dependencies)
with open('requirements.txt') as f:
    requires = f.read().splitlines()
missing = []

for sample in requires:
    samplesplit = sample.split()
    modulestr = samplesplit[0]
    # check if sphinxcontrib
    if 'sphinxcontrib' in modulestr:
        modulestr = modulestr.replace("-",".")
    try:
        if modulestr=="gdal":
            # We need to try two ways because the import command for gdal has changed
            try:
                module = __import__("gdal")
            except:
                module = __import__("osgeo")
        else:
            module = __import__(modulestr)
        # if dependency has version constraints
        if len(samplesplit) >= 3:
            ver = samplesplit[2]
            mver = get_distribution(modulestr).version
            if parse_version(mver) < parse_version(ver):
                print("Dependency %s version %s installed, but %s needed! " % (modulestr, mver, ver))
                sys.stdout.flush()
                missing.append(sample)
    except ImportError as e:
        print("Dependency %s not installed." % modulestr)
        print("The following exception occured: %s" % e)
        sys.stdout.flush()
        missing.append(sample)

# just run if missing/version mismatch dependencies are detected
if missing:
    question = ("\n\nDependencies %s are missing or version mismatch!\n\n" 
    "See http://wradlib.bitbucket.org on how to install dependencies!\n\n"
    "Type `yes` if setup.py should try to install/update missing packages\n"
    "(NOT RECOMMENDED!!!)\n"    
    "Type `quit` if you want to install via package manager or any other means!\n") % missing
    answer = query_yes_quit(question)
    if answer == 'quit':
        sys.exit('User quit setup.py')

# get current version from file
with open("version") as f:
    VERSION = f.read()
    VERSION = VERSION.strip()


if __name__ == '__main__':

    setup(name='wradlib',
          version=VERSION,
          description='Open Source Library for Weather Radar Data Processing',
          long_description = """\
wradlib - An Open Source Library for Weather Radar Data Processing
==================================================================

wradlib is designed to assist you in the most important steps of
processing weather radar data. These may include: reading common data
formats, georeferencing, converting reflectivity to rainfall
intensity, identifying and correcting typical error sources (such as
clutter or attenuation) and visualising the data.

""",
          license='BSD',
          url='http://wradlib.bitbucket.org/',
          download_url='https://bitbucket.org/wradlib/wradlib',
          packages=['wradlib'],
          include_package_data=True, # see MAINFEST.in
          classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: BSD License',
          'Environment :: Console',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          ],
          install_requires=missing
          )

