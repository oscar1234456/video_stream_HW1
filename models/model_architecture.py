import torch.nn as nn
import torch.nn.functional as F
import torch


class VGGNet_19(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3)),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.flatten = nn.Flatten()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=512*7*7, out_features=4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=4096, out_features=4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=4096, out_features=10)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
