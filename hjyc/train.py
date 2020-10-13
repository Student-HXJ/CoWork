#-*-coding:utf-8-*-
import argparse
import os
from datetime import datetime

import math
from PIL import Image

import numpy as np
import sys
import torch
import torch.nn as nn
from model import Custom_Model, TIFDataset

LEARNING_RATE = 0.0001
EPOCH_NUM = 200
CUDA_DEVICE = 0
DATA_DIRECTORY = './China_PM25'
LOGDIR_ROOT = './logdir'
STARTED_DATESTRING = datetime.now()


def get_arguments():
    parser = argparse.ArgumentParser(description='s2s network')
    parser.add_argument('--num_epoch', type=int, default=EPOCH_NUM, help='Number of epochs.')
    parser.add_argument(
        '--logdir_root',
        type=str,
        default=None,
        help='Root directory to place the logging '
        'output and generated model. These are stored ')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate for training.')
    parser.add_argument('--cuda_device', type=int, default=CUDA_DEVICE, help='cuda device last, -1 mean use cpu.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY, help='The directory of dataset.')
    parser.add_argument('--batch_size', type=int, default=1, help='The directory of dataset.')
    parser.add_argument('--step', type=int, default=12, help='The directory of dataset.')
    parser.add_argument('--img_size', type=int, default=50, help='The directory of dataset.')
    return parser.parse_args()


def save(path, net, optimizer, epoch, scaler, step='Done'):
    model_name = 'model_epoch{}_step{}.pth'.format(epoch, step)
    if not os.path.exists(path):
        os.makedirs(path)
    savepath = os.path.join(path, model_name)
    sys.stdout.flush()
    print('Storing checkpoint to {} ...'.format(savepath))

    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'scaler': scaler}
    torch.save(state, savepath)
    os.chmod(savepath, 777)


def validate_directories(args):
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT
    logdir_root = os.path.join(logdir_root, 'train', "{0:%Y-%m-%dT%H-%M-%S}".format(STARTED_DATESTRING))
    return logdir_root


def main():
    print('start time is ', "{0:%Y-%m-%dT%H-%M-%S}".format(STARTED_DATESTRING))
    args = get_arguments()
    if args.cuda_device != -1:
        cuda_device = args.cuda_device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    # get root logdir
    logdir = validate_directories(args)
    # get model
    model = Custom_Model()
    print(model)
    # set opt and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    # use GPU
    if cuda_device != -1:
        model = model.cuda()
        loss_function = loss_function.cuda()
    # dataset
    dataset = TIFDataset(os.path.abspath(args.data_dir), step=args.step, img_size=args.img_size)
    # pytorch dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
        # ,pin_memory=True
    )
    train_len = len(train_loader)
    for epoch in range(args.num_epoch):
        for i, (vp, vc, vf) in enumerate(train_loader):
            vp = vp.contiguous().view(args.step, 1, args.img_size, args.img_size)
            vc = vc.contiguous().view(args.step, 1, args.img_size, args.img_size)
            vf = vf.contiguous().view(args.step, args.img_size, args.img_size)
            if cuda_device != -1:
                vp, vc, vf = vp.cuda(), vc.cuda(), vf.cuda()
            pred = model(vp, vc)
            loss = loss_function(pred, vf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('epoch %d/%d, step %d/%d, loss %f' % (epoch, args.num_epoch, i, train_len, loss.item()))
    save(logdir, model, optimizer, args.num_epoch, dataset.scaler, train_len)
    end_time = datetime.now()
    print('end time is ', "{0:%Y-%m-%dT%H-%M-%S}".format(end_time))
    print('time used is ', end_time - STARTED_DATESTRING)


if __name__ == '__main__':
    main()
