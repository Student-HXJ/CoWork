#-*-coding:utf-8-*-
from torch.nn import Module
import torch


def inv_transform(scaler, data):
    shape = data.shape
    return scaler.inverse_transform(data.reshape(-1, 1)).reshape(shape)


class AccuracyCaculate(Module):

    def __init__(self, cuda_device, scaler, step=12):
        super(AccuracyCaculate, self).__init__()
        self.cuda_device = cuda_device
        self.scaler = scaler
        self.step = step

    def forward(self, pred, label, pred_op=True):
        if pred_op:
            pred = inv_transform(self.scaler, pred).reshape(self.step, 50, 50)
            pred[pred < 0] = 0
            label = inv_transform(self.scaler, label).reshape(self.step, 50, 50)
        pred, label = torch.from_numpy(pred), torch.from_numpy(label)
        if self.cuda_device != -1:
            pred, label = pred.cuda(), label.cuda()
        pred, label = torch.flatten(pred), torch.flatten(label)
        mse = torch.mean((pred - label).pow(2))
        rmse = mse.pow(0.5)
        mae = torch.mean(torch.abs(pred - label))
        r2 = 1 - mse / torch.var(label)
        print('MSE=%.3f\tRMSE=%.3f\tMAE=%.3f\tR2=%.3f' % (mse, rmse, mae, r2))
        return
