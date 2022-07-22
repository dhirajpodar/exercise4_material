import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.nn import functional as F
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # For downsampling
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        output = self.conv1(x)
        output = F.relu(self.bn1(output))

        output = self.conv2(output)
        output = self.bn2(output)

        if self.in_channels != self.out_channels:
            # For downsampling
            res = self.conv3(x)
            res = self.bn3(res)

            output += res
        else:
            output += x

        output = F.relu(output)

        return output


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)

        self.block1 = ResBlock(64, 64, 1)
        self.block2 = ResBlock(64, 128, 2)
        self.block3 = ResBlock(128, 256, 2)
        self.block4 = ResBlock(256, 512, 2)

        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pooling(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x


'''class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.model = nn.Sequential(
            *list(models.alexnet(pretrained=True).features.children())[:-1]
        )
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

        self.new_layer = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        x = self.model(x)
        x = self.new_layer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return torch.sigmoid(x)
'''