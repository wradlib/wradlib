# commit and push
hg commit -u "Codeship CI <wradlib@codeship.io>" -m "DOC: upload newly built doc" 2>&1 | tee commit_doc.txt
hg push ssh://hg@bitbucket.org/wradlib/wradlib.bitbucket.org 2>&1 | tee push_doc.txt
# and go back
cd -
cp *.txt ~/src/bitbucket.org/wradlibcodeship/wiki
# change to documentation repo wiki
cd ~/src/bitbucket.org/wradlibcodeship/wiki
# remove .doctrees folder from
ls -lart
hg add
# commit and push
hg commit -u "Codeship CI <wradlib@codeship.io>" -m "WIKI: wiki upload after build-test"
hg push ssh://hg@bitbucket.org/wradlib/wradlib.bitbucket.org/wiki
