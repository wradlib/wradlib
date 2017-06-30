#!/bin/bash
# Copyright (c) 2017, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

# print env vars
echo "TRAVIS_PULL_REQUEST " $TRAVIS_PULL_REQUEST
echo "TRAVIS_SECURE_ENV_VARS " $TRAVIS_SECURE_ENV_VARS
echo "TRAVIS_TAG " $TRAVIS_TAG ${TRAVIS_TAG:1}
echo "PYTHON_VERSION" $PYTHON_VERSION
echo "COVERAGE" $COVERAGE
echo "DOC_BUILD" $DOC_BUILD
echo "WRADLIB_DOCKER_TAG" $WRADLIB_DOCKER_TAG

# run docker container
docker run -d -ti \
            --name wradlib_build \
            -v "${WRADLIB_BUILD_DIR}":/home/build \
            -e LOCAL_USER_ID=$UID \
            -e WRADLIB_BUILD_DIR=/home/build \
            -e WRADLIB_DATA=/home/build/wradlib-data \
            -e WRADLIB_NOTEBOOKS=/home/build/notebooks-render \
            -e PYTHON_VERSION=$PYTHON_VERSION \
            -e COVERAGE=$COVERAGE \
            -e DOC_BUILD=$DOC_BUILD \
            -e CI \
            -e TRAVIS \
            -e TRAVIS_PULL_REQUEST \
            -e TRAVIS_SECURE_ENV_VARS \
            -e TRAVIS_BRANCH \
            -e TRAVIS_TAG \
            -e TRAVIS_JOB_NUMBER \
            -e TRAVIS_JOB_ID \
            -e TRAVIS_REPO_SLUG \
            -e TRAVIS_COMMIT \
            -e TRAVIS_BUILD_DIR=/home/build/wradlib \
            -e TRAVIS_OS_NAME \
            -e TRAVIS_PYTHON_VERSION=$PYTHON_VERSION \
            wradlib/wradlib-docker:$WRADLIB_DOCKER_TAG /bin/bash