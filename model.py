import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, dropout_rate=0.1, activation=F.relu):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size,  out_size, kernel_size, padding=1)
        #self.norm1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        #self.norm2 = nn.BatchNorm2d(out_size)
        self.activation = activation
        self.drop = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        #x = self.norm1(x)
        x = self.drop(x)
        x = self.activation(self.conv2(x))
        #x = self.norm2(x)
        return x

class ConvUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, dropout_rate=0.1, activation=F.relu):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        #self.norm1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        #self.norm2 = nn.BatchNorm2d(out_size)
        self.activation = activation
        self.drop = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, bridge):
        x = self.up(x)
        x = torch.cat([x, bridge], 1)
        x = self.activation(self.conv1(x))
        #x = self.norm1(x)
        x = self.drop(x)
        x = self.activation(self.conv2(x))
        #x = self.norm2(x)
        return x
    
class Model(nn.Module):
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

if __name__ == '__main__':
    net = Model()
    print(net)
    del net