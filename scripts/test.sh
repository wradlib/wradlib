# try this to see if the pipe gets testrunner's exit status
#set -o pipefail
#python testrunner.py --all 2>&1 | tee testrunner.txt
python testrunner.py --all 2>&1 > testrunner.txt
# echo exit code of testrunner call
echo $?
