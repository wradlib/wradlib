#!/usr/bin/env bash
# Copyright (c) 2016, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.
# Adapted from the continuous_integration/build_docs.sh file from the pyart project
# https://github.com/ARM-DOE/pyart/


set -e

export PING_SLEEP=30s
export WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BUILD_OUTPUT=$WORKDIR/build.out

touch $BUILD_OUTPUT

dump_output() {
   echo Tailing the last 500 lines of output:
   tail -500 $BUILD_OUTPUT
}
error_handler() {
  echo ERROR: An error was encountered with the build.
  dump_output
  exit 1
}

# If an error occurs, run our error handler to output a tail of the build.
trap 'error_handler' ERR

# Set up a repeating loop to send some output to Travis.
bash -c "while true; do echo \$(date) - building ...; sleep $PING_SLEEP; done" &
PING_LOOP_PID=$!

cd "$WRADLIB_BUILD_DIR"

# print the vars
echo "TRAVIS_PULL_REQUEST " $TRAVIS_PULL_REQUEST
echo "TRAVIS_SECURE_ENV_VARS " $TRAVIS_SECURE_ENV_VARS
echo "TRAVIS_TAG " $TRAVIS_TAG ${TRAVIS_TAG:1}

# remove possible build residues
rm -rf doc-build
rm -rf wradlib-docs

# create doc build directory
mkdir doc-build

# create docs and upload to wradlib-docs repo if this is not a pull request and
# secure token is available.
# else build local docs
if [ "$TRAVIS_PULL_REQUEST" == "false" ] && [ $TRAVIS_SECURE_ENV_VARS == 'true' ]; then

    # clone wradlib-docs
    echo "Cloning wradlib-docs repo"
    git clone https://github.com/wradlib/wradlib-docs.git

    # get tagged versions from cloned repo
    cd wradlib-docs
    TAGGED_VERSIONS=`for f in [0-9]*; do echo "$f"; done `
    cd ..
    # export variable, used in conf.py
    echo "TAGGED_VERSIONS " $TAGGED_VERSIONS
    export TAGGED_VERSIONS=$TAGGED_VERSIONS

    # move index.html to build directory
    mv wradlib-docs/index.html doc-build/.

    # copy tagged doc folders to build directory
    for folder in $TAGGED_VERSIONS; do
        mv wradlib-docs/$folder doc-build/.
    done

    # check travis_tag
    # if is tagged version
    if [ -n "$TRAVIS_TAG" ]; then

        echo "Building Tagged Docs"

        TAG=${TRAVIS_TAG:1}
        # export variable, used in conf.py
        export TAG=$TAG

        # we need to link tag to latest
        mkdir doc-build/$TAG
        ln -s $TAG doc-build/latest

        # need to replace /latest/ in notebooks to $TAG
        find doc/source/ -name *.ipynb -type f -exec sed -i 's/latest/$TAG/g' {} \;

    # if is devel version
    else
        echo "Building Devel Docs"
        TAG='latest'
        export TAG=$TAG
    fi

    # build docs
    sphinx-build -b html doc/source doc-build/$TAG >> $BUILD_OUTPUT

    echo "Pushing Docs"
    cd doc-build
    git config --global user.email "wradlib-docs@wradlib.org"
    git config --global user.name "wradlib-docs-bot"
    git init
    touch README
    git add README
    git commit -m "Initial commit" --allow-empty
    git branch gh-pages
    git checkout gh-pages
    touch .nojekyll
    git add --all .
    git commit -m "Version" --allow-empty
    git remote add origin https://$GH_TOKEN@github.com/wradlib/wradlib-docs.git >> $BUILD_OUTPUT
    git push origin gh-pages -fq >> $BUILD_OUTPUT

else

    echo "Building Local Docs"
    sphinx-build -b html doc/source doc-build/latest >> $BUILD_OUTPUT
    echo "Not Pushing Docs"

fi

# The build finished without returning an error so dump a tail of the output.
dump_output

# Nicely terminate the ping output loop.
kill $PING_LOOP_PID
rm -rf $BUILD_OUTPUT
