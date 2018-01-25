import os
import json
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import config
from model import Model
from dataset import KaggleDataset, Compose
from multiprocessing import Manager

def ckpt_path(epoch=None):
    checkpoint_dir = os.path.join('.', 'checkpoint')
    current_path = os.path.join('.', 'checkpoint', 'current.json')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if epoch is None:
        if os.path.exists(current_path):
            with open(current_path) as infile:
                data = json.load(infile)
                epoch = data['epoch']
        else:
            return ''
    else:
        with open(current_path, 'w') as outfile:
            json.dump({
                'epoch': epoch
            }, outfile)
    return os.path.join(checkpoint_dir, 'ckpt-{}.pkl'.format(epoch))

def main(args):
    model = Model()
    if config.cuda:
        model = model.cuda()
        #model = torch.nn.DataParallel(model).cuda()

    cost = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learn_rate, weight_decay=1e-6)

    # dataloader workers are forked process thus we need a IPC manager to keep cache in same memory space
    manager = Manager()
    cache = manager.dict()
    # prepare dataset and loader
    compose = Compose(128)
    dataset = KaggleDataset('data/stage1_train', transform=compose, cache=cache)
    dataloader = DataLoader(
        dataset, shuffle=True,
        batch_size=config.n_batch,
        num_workers=config.n_worker,
        pin_memory=config.cuda)

    # validate checkpoint
    start_epoch = 1
    if args.resume:
        ckpt = ckpt_path()
        if os.path.isfile(ckpt):
            print("Loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('Grand new training ...')

    print('Training started...')
    for epoch in range(start_epoch, config.n_epoch + start_epoch):
        train(dataloader, model, cost, optimizer, epoch)

        # save checkpoint per 10 epoch
        if epoch % 10 == 0:
            ckpt = ckpt_path(epoch)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt)

    print('Training finished...')


def train(loader, model, cost, optimizer, epoch):
    model.train() # Sets the module in training mode.
    running_loss = 0.0
    counter = 0
    for i, data in enumerate(loader):
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
        outputs = model(inputs)
        loss = cost(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.data[0]
        counter += 1
        # print every num_step mini-batches
    print('[%d, %3d] loss: %.3f' %
        (epoch, i + 1, running_loss / counter))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    main(args)