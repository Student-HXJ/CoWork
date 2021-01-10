#-*-coding:utf-8-*-
import argparse
import os
from datetime import datetime

import math
from PIL import Image

import sys
import torch
import torch.nn as nn
from model import Custom_Model, TIFDataset, AccuracyCaculate
CUDA_DEVICE = 0
DATA_DIRECTORY = './China_PM25'
TEST_DIRECTORY = './China_PM25_T'
LOGDIR_ROOT = './logdir'
STARTED_DATESTRING = datetime.now()
STEP = 12


def get_arguments():
    parser = argparse.ArgumentParser(description='s2s network predict')
    parser.add_argument('--checkpoint',
                        type=str,
                        help='Which model checkpoint to predict')
    parser.add_argument('--data_dir',
                        type=str,
                        default=DATA_DIRECTORY,
                        help='The directory of dataset.')
    parser.add_argument('--test_dir',
                        type=str,
                        default=TEST_DIRECTORY,
                        help='The directory of dataset.')
    parser.add_argument('--cuda_device',
                        type=int,
                        default=CUDA_DEVICE,
                        help='cuda device, -1 mean use cpu.')
    parser.add_argument('--step',
                        type=int,
                        default=STEP,
                        help='The directory of dataset.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='The directory of dataset.')
    parser.add_argument('--img_size',
                        type=int,
                        default=50,
                        help='The directory of dataset.')
    parser.add_argument('--logdir_root',
                        type=str,
                        default=LOGDIR_ROOT,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    return parser.parse_args()


def load(path):
    print("Trying to restore saved checkpoints from {} ...".format(path))
    checkpoint = torch.load(path)
    return checkpoint


def validate_directories(args):
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT
    logdir_root = os.path.join(
        logdir_root, 'out', "{0:%Y-%m-%dT%H-%M-%S}".format(STARTED_DATESTRING))
    return logdir_root


def img_save(path, img, type_, step):
    if not os.path.exists(path):
        os.makedirs(path)
    model_name = type_ + '_' + str(step) + '.tif'
    savepath = os.path.join(path, model_name)
    sys.stdout.flush()
    img_ = Image.fromarray(img)
    img_.save(savepath)


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

    # restore
    checkpoint = load(args.checkpoint)
    # dataset
    dataset = TIFDataset(
        os.path.abspath(args.data_dir),
        testpath=os.path.abspath(args.test_dir),
        step=args.step,
        # scaler=checkpoint['scaler'],
        img_size=args.img_size,
        isTrain=False)
    # pytorch dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # ,pin_memory=True
    )

    model.load_state_dict(checkpoint['net'])
    print('restore done...........')
    model.eval()
    # validate model
    acc = AccuracyCaculate(cuda_device, dataset.scaler)
    # use GPU
    if cuda_device != -1:
        model, acc = model.cuda(), acc.cuda()
    train_len = len(train_loader)
    for i, (vp, vc, vf) in enumerate(train_loader):
        vp = vp.contiguous().view(args.step, 1, args.img_size, args.img_size)
        vc = vc.contiguous().view(args.step, 1, args.img_size, args.img_size)
        vf = vf.contiguous().view(args.step, args.img_size, args.img_size)
        if cuda_device != -1:
            vp, vc = vp.cuda(), vc.cuda()
        pred = model(vp, vc)
        if cuda_device != -1:
            pred = pred.cpu()
        pred, vf = pred.detach().numpy(), vf.numpy()
        acc(pred, vf)
        for n_step in range(args.step):
            array_predict = pred[n_step, :, :]
            array_actual = vf[n_step, :, :]
            img_save(logdir, array_predict, 'predict', n_step)
            img_save(logdir, array_actual, 'label', n_step)
            print('step {}'.format(n_step))
            acc(array_predict, array_actual, pred_op=False)
        break
    end_time = datetime.now()
    print('end time is ', "{0:%Y-%m-%dT%H-%M-%S}".format(end_time))
    print('time used is ', end_time - STARTED_DATESTRING)


if __name__ == '__main__':
    main()
