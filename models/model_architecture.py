import torch.nn as nn
import torch.nn.functional as F
import torch


class VGGNet_19(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(inplace=True),
            # torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            # torch.nn.BatchNorm2d(256),
            # torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(),
            # torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            # torch.nn.BatchNorm2d(512),
            # torch.nn.LeakyReLU(),
            # 191634前拔掉
            # torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            # torch.nn.BatchNorm2d(512),
            # torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(inplace=True),
            # torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            # torch.nn.BatchNorm2d(512),
            # torch.nn.LeakyReLU(inplace=True),
            # 191634前拔掉
            # torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            # torch.nn.BatchNorm2d(512),
            # torch.nn.LeakyReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.flatten = nn.Flatten()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=512*3*3, out_features=200),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=200, out_features=25),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_features=25, out_features=10)
        )

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)
        return x
