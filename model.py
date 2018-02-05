import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, dropout_rate=0.2, activation=F.relu):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size,  out_size, kernel_size, padding=1)
        self.norm1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.norm2 = nn.BatchNorm2d(out_size)
        self.activation = activation
        self.drop = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.norm1(x)
        x = self.activation(self.conv2(x))
        x = self.norm2(x)
        x = self.drop(x)
        return x

class ConvUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, dropout_rate=0.2, activation=F.relu):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.norm1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.norm2 = nn.BatchNorm2d(out_size)
        self.activation = activation
        self.drop = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, bridge):
        x = self.up(x)
        x = torch.cat([x, bridge], 1)
        x = self.activation(self.conv1(x))
        x = self.norm1(x)
        x = self.activation(self.conv2(x))
        x = self.norm2(x)
        x = self.drop(x)
        return x
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(4, 16)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(16, 32)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = ConvBlock(32, 64)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = ConvBlock(64, 128)
        self.p4 = nn.MaxPool2d(2)
        self.c5 = ConvBlock(128, 256)
        self.u6 = ConvUpBlock(256, 128)
        self.u7 = ConvUpBlock(128, 64)
        self.u8 = ConvUpBlock(64, 32)
        self.u9 = ConvUpBlock(32, 16)
        self.ce = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        c1 = x = self.c1(x)
        x = self.p1(x)
        c2 = x = self.c2(x)
        x = self.p2(x)
        c3 = x = self.c3(x)
        x = self.p3(x)
        c4 = x = self.c4(x)
        x = self.p4(x)
        c5 = x = self.c5(x)
        x = self.u6(x, c4)
        x = self.u7(x, c3)
        x = self.u8(x, c2)
        x = self.u9(x, c1)
        x = self.ce(x)
        x = F.sigmoid(x)
        return x


# Transfer Learning VGG16_BatchNorm as Encoder part of UNet
class DeConvBlk(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_ratio=0.1):
        super().__init__()
        self.upscaling = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch//2 + out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(p=dropout_ratio),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(p=dropout_ratio)
        )

    def forward(self, x1, x2):
        x1 = self.upscaling(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x1, x2], dim=1)
        x = self.convs(x)
        return x


class outConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNetVgg16(nn.Module):
    def __init__(self, n_channels, n_classes, fixed_vgg = False):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.vgg16 = models.vgg16_bn(pretrained=True)

        self.conv1a = self.vgg16.features[0]   #64
        self.bn1a = self.vgg16.features[1]
        self.conv1b = self.vgg16.features[3]   #64
        self.bn1b = self.vgg16.features[4]

        self.conv2a = self.vgg16.features[7]  #128
        self.bn2a = self.vgg16.features[8]
        self.conv2b = self.vgg16.features[10]  #128
        self.bn2b = self.vgg16.features[11]

        self.conv3a = self.vgg16.features[14]  #256
        self.bn3a = self.vgg16.features[15]
        self.conv3b = self.vgg16.features[17]  #256
        self.bn3b = self.vgg16.features[18]
        self.conv3c = self.vgg16.features[20]  #256
        self.bn3c = self.vgg16.features[21]

        self.conv4a = self.vgg16.features[24]  #512
        self.bn4a = self.vgg16.features[25]
        self.conv4b = self.vgg16.features[27]  #512
        self.bn4b = self.vgg16.features[28]
        self.conv4c = self.vgg16.features[30]  #512
        self.bn4c = self.vgg16.features[31]

        self.conv5a = self.vgg16.features[34]  #512
        self.bn5a = self.vgg16.features[35]
        self.conv5b = self.vgg16.features[37]  #512
        self.bn5b = self.vgg16.features[38]
        self.conv5c = self.vgg16.features[40]  #512
        self.bn5c = self.vgg16.features[41]

        self.deconv1 = DeConvBlk(512, 512)
        self.deconv2 = DeConvBlk(512, 256)
        self.deconv3 = DeConvBlk(256, 128)
        self.deconv4 = DeConvBlk(128, 64)
        self.outc = outConv(64, n_classes)
        if fixed_vgg:
            for param in self.vgg16.parameters():
                param.requires_grad = False

    def forward(self, x):
        t = self.relu(self.bn1a(self.conv1a(x)))
        x1 = self.relu(self.bn1b(self.conv1b(t)))
        t = self.maxpool(x1)
        t = self.relu(self.bn2a(self.conv2a(t)))
        x2 = self.relu(self.bn2b(self.conv2b(t)))
        t = self.maxpool(x2)
        t = self.relu(self.bn3a(self.conv3a(t)))
        t = self.relu(self.bn3b(self.conv3b(t)))
        x3 = self.relu(self.bn3c(self.conv3c(t)))
        t = self.maxpool(x3)
        t = self.relu(self.bn4a(self.conv4a(t)))
        t = self.relu(self.bn4b(self.conv4b(t)))
        x4 = self.relu(self.bn4c(self.conv4c(t)))
        t = self.maxpool(x4)
        t = self.relu(self.bn5a(self.conv5a(t)))
        t = self.relu(self.bn5b(self.conv5b(t)))
        x5 = self.relu(self.bn5c(self.conv5c(t)))
        t = self.deconv1(x5, x4)
        t = self.deconv2(t, x3)
        t = self.deconv3(t, x2)
        t = self.deconv4(t, x1)
        t = self.outc(t)
        t = F.sigmoid(t)
        return t


if __name__ == '__main__':
    net = UNet()
    print(net)
    del net
    net = UNetVgg16()
    print(net)
    del net