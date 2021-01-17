cmd=$1

if [ ${cmd} == "git" ]; then
    git add . && git commit -m ""$(date +%Y-%m-%d)"" && git push
elif [ ${cmd} == "rfe" ]; then
    nohup python3 -u RFEtrain.py >./log/rferesult4.log 2>&1 &
elif [ ${cmd} == "knn" ]; then
    nohup python3 -u KNNtrain.py >./log/knnresult.log 2>&1 &
elif [ ${cmd} == "svm" ]; then
    nohup python3 -u SVMtrain.py >./log/svmresult.log 2>&1 &
elif [ ${cmd} == "dst" ]; then
    nohup python3 -u DSTtrain.py >./log/dstresult.log 2>&1 &
else
    echo "No selection"
fi
