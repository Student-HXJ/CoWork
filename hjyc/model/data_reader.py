import torch.utils.data as data
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from PIL import Image


def create_scaler(folder):
    file_name = os.listdir(folder)[0]
    img = Image.open(os.path.join(folder, file_name))
    img = np.array(img)
    img[img < 0] = 0
    scaler = StandardScaler()
    scaler.fit(img.reshape(-1, 1))
    return scaler


def transform(scaler, data):
    shape = data.shape
    return scaler.transform(data.reshape(-1, 1)).reshape(shape)


class TIFDataset(data.Dataset):

    def __init__(self, filepath, testpath=None, step=12, scaler=None, isTrain=True, img_size=50):
        if scaler is None:
            self.scaler = create_scaler(filepath)
        else:
            self.scaler = scaler
        self.step = step
        self.isTrain = isTrain
        self.img_size = img_size
        self.folder = filepath
        self.test_folder = testpath
        self.files = []
        self.pos_w = 140
        self.pos_h = 52
        for e in os.listdir(filepath):
            self.files.append(e)
        self.files = sorted(self.files)

    def __len__(self):
        lens = len(self.files)
        if self.isTrain:
            file_len = lens - self.step * 3
        else:
            file_len = lens - self.step * 2
        self.file_len = file_len
        return file_len

    def __getitem__(self, index):
        vp, vc, vf = [], [], []
        if not self.isTrain:
            index = self.file_len
        # get vp
        for n in range(self.step):
            img = Image.open(os.path.join(self.folder, self.files[index + n]))
            img = img.crop((self.pos_w, self.pos_h, self.pos_w + self.img_size, self.pos_h + self.img_size))
            img = np.array(img)
            img[img < 0] = 0
            vp.append(img)
        # get vc
        for n in range(self.step, 2 * self.step):
            img = Image.open(os.path.join(self.folder, self.files[index + n]))
            img = img.crop((self.pos_w, self.pos_h, self.pos_w + self.img_size, self.pos_h + self.img_size))
            img = np.array(img)
            img[img < 0] = 0
            vc.append(img)

        vp = torch.from_numpy(transform(self.scaler, np.asarray(vp))).float()
        vc = torch.from_numpy(transform(self.scaler, np.asarray(vc))).float()
        # get vf
        if self.isTrain:
            for n in range(2 * self.step, 3 * self.step):
                img = Image.open(os.path.join(self.folder, self.files[index + n]))
                img = img.crop((self.pos_w, self.pos_h, self.pos_w + self.img_size, self.pos_h + self.img_size))
                img = np.array(img)
                img[img < 0] = 0
                vf.append(img)
        else:
            t_files = []
            for e in os.listdir(self.test_folder):
                t_files.append(e)
            t_files = sorted(t_files)
            for n in range(self.step):
                img = Image.open(os.path.join(self.test_folder, t_files[n]))
                img = img.crop((self.pos_w, self.pos_h, self.pos_w + self.img_size, self.pos_h + self.img_size))
                img = np.array(img)
                img[img < 0] = 0
                vf.append(img)
        vf = torch.from_numpy(transform(self.scaler, np.asarray(vf))).float()
        return vp, vc, vf
