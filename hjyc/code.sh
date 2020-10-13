nohup python train.py > ot.out 2>&1 &
python valid.py --checkpoint=./logdir/train/2020-05-08T20-14-27/model_epoch200_step144.pth