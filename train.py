import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import config
from model import Model
from dataset import KaggleDataset, Compose

def train():
    net = Model()
    if config.cuda:
        net = net.cuda()
        #net = torch.nn.DataParallel(net).cuda()

    cost = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learn_rate, weight_decay=1e-6)

    compose = Compose(128)
    dataset = KaggleDataset('data/stage1_train', transform=compose)
    dataloader = DataLoader(
        dataset, shuffle=True,
        batch_size=config.n_batch,
        num_workers=config.n_worker)

    print('Training started...')
    for epoch in range(config.n_epoch):
        net.train() # Sets the module in training mode.
        running_loss = 0.0
        counter = 0
        for i, data in enumerate(dataloader):
            # get the inputs
            inputs = data['image']
            labels = data['label']
            if config.cuda:
               inputs, labels = inputs.cuda(), labels.cuda()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = cost(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            counter += 1
            # print every num_step mini-batches
        print('[%d, %3d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / counter))
    print('Training finished...')

if __name__ == '__main__':
    train()