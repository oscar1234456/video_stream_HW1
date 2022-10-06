# Author: 310551076 Oscar Chen
# Course: NYCU Video Streaming and Tracking
# Title: HW1 - Classification
# Date: 2022 / 10 / 10
# Email: oscarchen.cs10@nycu.edu.tw
import logging

import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import random

logger = logging.getLogger(__name__)


# For Training & Val
class ImageLoader(Dataset):
    def __init__(self, root, mode, config):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.config = config
        logger.info(f">Found {len(self.img_name)} images ({mode})...")

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        data_transform = {
            "train": transforms.Compose(
                # [
                #     # Try Different Transform
                #     # transforms.RandomRotation(degrees=(0,360)),  #first priority
                #     # transforms.RandomResizedCrop(224),
                #     # transforms.Resize(260),
                #     # transforms.CenterCrop(224),
                #     # transforms.RandomHorizontalFlip(),
                #     # tra
                #     nsforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                #     # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                #     # transforms.Normalize([0.4693, 0.3225, 0.2287], [0.1974, 0.1399, 0.1014])
                #     # transforms.Resize((224, 224)),
                #     # transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                # ]
                self.config.train_data_transform
            ),
            "test": transforms.Compose(
                # [
                #     transforms.Resize((224, 224)),
                #     transforms.ToTensor(),
                # ]
                self.config.test_data_transform
            ),
        }
        img_path = self.root + self.mode+ "/" +self.img_name[index]
        image = Image.open(img_path).convert('RGB')
        label = self.label[index]
        imageConvert = data_transform[self.mode](image)
        return imageConvert, label


def getData(mode):
    # training / validation
    if mode == "train":
        train_list = pd.read_csv("/mnt/2ndHDD/oscarchencs10/video_streaming/train.csv")
        print(train_list)
        img = train_list["names"]
        label = train_list["label"]
    else:
        val_list = pd.read_csv("/mnt/2ndHDD/oscarchencs10/video_streaming/val.csv")
        img = val_list["names"]
        label = val_list["label"]

    #TODO: add testing dataset loader

    return np.squeeze(img.values), np.squeeze(label.values)


# For Imbalanced Data Approach Testing(Count Weight with each class):
def normalWeightGetter():
    labelData = pd.read_csv("./csv/y_train.csv")
    # labelDF = pd.DataFrame(labelData)
    labelCount = labelData.value_counts()
    normalWeight = 1 - (labelCount / labelCount.sum())
    # normalWeight = len(labelData)/(5*labelCount)
    return torch.FloatTensor(normalWeight)


if __name__ == '__main__':
    # Test For DataLoader
    test_data = ImageLoader("/mnt/2ndHDD/oscarchencs10/video_streaming/val", 'test')
    train_data = ImageLoader("/mnt/2ndHDD/oscarchencs10/video_streaming/train", 'train')

    img, label = train_data[24]
    plt.figure()
    # transforms.Compose(
    #     [
    #         transforms.Resize((224, 224)),
    #     ])(img)
    img_tran = img.numpy().transpose((1, 2, 0)).squeeze()  # [C,H,W]->[H,W,C]
    # img_tran = img.numpy().transpose((1, 2, 0))
    plt.imshow((img_tran * 255).astype(np.uint8), cmap=plt.cm.gray_r)
    # plt.imshow(img_tran)
    plt.show()

    # trainLoader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    # for batch, (inputs, labels) in enumerate(trainLoader):
    #     # size = len(dataloaders[phase])
    #     print("in training")
    #     inputs = inputs
    #     labels = labels
