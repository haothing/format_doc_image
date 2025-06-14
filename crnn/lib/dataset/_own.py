from __future__ import print_function, absolute_import
import torch.utils.data as data
from torchvision.datasets.vision import VisionDataset
import os
import numpy as np
import cv2
from PIL import Image

class _OWN(VisionDataset):
    def __init__(self, config, is_train=True, transform=None):

        self.root = config.DATASET.ROOT
        super().__init__(self.root, transform=transform)

        self.is_train = is_train
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        txt_files = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']
        if type(txt_files) is not list:
            txt_files = [txt_files]

        # convert name:indices to name:string
        self.labels = []
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as file:
                self.labels += [{c.split(' ')[0]: c.split(' ')[-1][:-1]} for c in file.readlines()]

        #print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img_name = list(self.labels[idx].keys())[0]
        img = Image.open(os.path.join(self.root, img_name))       
        if self.transform is not None:
            img = self.transform(img)
        
        img = img.resize((self.inp_w, self.inp_h), Image.BICUBIC)
        img = np.array(img)
        
        # img_h, img_w = img.shape
        # img = cv2.imread(os.path.join(self.root, img_name))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (0,0), fx=self.inp_w / img_w, fy=self.inp_h / img_h, interpolation=cv2.INTER_CUBIC)

        img = np.reshape(img, (self.inp_h, self.inp_w, 1))
        img = img.astype(np.float32) / 255.
        # TODO comment out get standard score use setting
        # img = (img - self.mean) / self.std
        img = (img - img.mean()) / img.std()
        img = img.transpose([2, 0, 1])

        return img, idx








