cmd=$1

if [ ${cmd} == "git" ]; then
    git add . && git commit -m ""$(date +%Y-%m-%d)"" && git push
elif [ ${cmd} == "RFEtrain" ]; then
    nohup python3 -u RFEtrain.py >./log/result1.log 2>&1 &
elif [ ${cmd} == "test" ]; then
    nohup python3 -u test.py >./log/test1.log 2>&1 &
else
    echo "No selection"
fi
