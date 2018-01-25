# python built-in library
import os
import json
import argparse
import time
from multiprocessing import Manager
# 3rd party library
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
# own code
import config
from model import Model
from dataset import KaggleDataset, Compose
from helper import AverageMeter

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
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # Sets the module in training mode.
    model.train()
    end = time.time()
    for i, data in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # get the inputs
        inputs = data['image']
        labels = data['label']
        if config.cuda:
            inputs, labels = inputs.cuda(async=True), labels.cuda(async=True)
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward step
        outputs = model(inputs)
        loss = cost(outputs, labels)
        # measure accuracy and record loss
        losses.update(loss.data[0], inputs.size(0))
        # compute gradient and do backward step
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % config.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses
                )
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(resume=True)
    parser.set_defaults(cuda=config.cuda)
    args = parser.parse_args()

    # final check whether cuda is avaiable
    config.cuda = torch.cuda.is_available() and args.cuda

    main(args)