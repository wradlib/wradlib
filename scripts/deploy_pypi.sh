# deploy to pypi if new version
if "$(CI_BRANCH)"=='release'
then
    python setup.py sdist bdist_wheel bdist_wininst upload
else
    python setup.py sdist bdist_wheel bdist_wininst
fi
