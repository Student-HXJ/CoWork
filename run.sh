cmd=$1

if [ ${cmd} == "git" ]; then
    git add . && git commit -m ""$(date +%Y-%m-%d)"" && git push
elif [ ${cmd} == "RFEtrain" ]; then
    nohup python3 -u RFEtrain.py >./log/test.log 2>&1 &
else
    echo "No selection"
fi
