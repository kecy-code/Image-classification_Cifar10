import torch
import torch.nn as nn
import torch.nn.functional as F


class SeBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SeVGGbase(nn.Module):
    def __init__(self):
        super(SeVGGbase, self).__init__()

        # 3 * 28 * 28 (crop-->32, 28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SeBlock(64)
        )
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 14 * 14
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SeBlock(128)
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SeBlock(128)
        )

        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 7 * 7
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SeBlock(256)
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            SeBlock(256)
        )

        self.max_pooling3 = nn.MaxPool2d(kernel_size=2,
                                         stride=2,
                                         padding=1)

        # 4 * 4
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SeBlock(512)
        )

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SeBlock(512)
        )
        self.max_pooling4 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)

        # batchsize * 512 * 2 *2 --> batchsize * (512 * 4)
        self.fc = nn.Linear(512 * 4, 10)

    def forward(self, x):
        batchsize = x.size(0)
        out = self.conv1(x)
        out = self.max_pooling1(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.max_pooling2(out)

        #
        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.max_pooling3(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.max_pooling4(out)
        #
        out = out.view(batchsize, -1)
        # batchsize * c * h * w --> batchsize * n

        out = self.fc(out)  # batchsize * 11
        out = F.log_softmax(out, dim=1)

        return out


def Se_VGGNet():
    return SeVGGbase()
