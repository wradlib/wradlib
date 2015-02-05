# clone documentation repository
hg clone ssh://hg@bitbucket.org/wradlib/wradlib.bitbucket.org ~/src/bitbucket.org/wradlibcodeship/wradlib.bitbucket.org 2>&1 | tee clone_docu.txt
# clone wiki for later upload of build results
hg clone ssh://hg@bitbucket.org/wradlib/wradlib.bitbucket.org/wiki ~/src/bitbucket.org/wradlibcodeship/wiki 2>&1 | tee clone_wiki.txt
