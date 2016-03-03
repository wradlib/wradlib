#!/usr/bin/env bash
# Adapted from the continuous_integration/build_docs.sh file from the pyart project
# https://github.com/ARM-DOE/pyart/
set -e

cd "$TRAVIS_BUILD_DIR"

echo "Building Docs"

#mv "$TRAVIS_BUILD_DIR"/doc /tmp
sphinx-build -b html doc/source doc-build

# upload to wradlib-docs repo if this is not a pull request and
# secure token is available.
if [ "$TRAVIS_PULL_REQUEST" == "false" ] && [ $TRAVIS_SECURE_ENV_VARS == 'true' ]; then
    echo "Pushing Docs"
    cd doc-build
    git config --global user.email "wradlib-docs@example.com"
    git config --global user.name "wradlib-docs"

    git init
    touch README
    git add README
    git commit -m "Initial commit" --allow-empty
    git branch gh-pages
    git checkout gh-pages
    touch .nojekyll
    git add --all .
    git commit -m "Version" --allow-empty
    git remote add origin https://$GH_TOKEN@github.com/wradlib/wradlib-docs.git &> /dev/null
    git push origin gh-pages -fq &> /dev/null
else
    echo "Not Pushing Docs"
fi

exit 0
