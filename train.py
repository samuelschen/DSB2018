# python built-in library
import os
import argparse
import time
from multiprocessing import Manager
# 3rd party library
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter
# own code
from model import UNet, UNetVgg16, CAUNet, CAMUNet, DCAN
from dataset import KaggleDataset, Compose
from helper import config, AverageMeter, iou_mean, save_ckpt, load_ckpt
from loss import contour_criterion, focal_criterion


def main(resume=True, n_epoch=None, learn_rate=None):
    model_name = config['param']['model']
    cv_ratio = config['param'].getfloat('cv_ratio')
    if learn_rate is None:
        learn_rate = config['param'].getfloat('learn_rate')
    width = config[model_name].getint('width')
    weight_map = config['param'].getboolean('weight_map')
    c = config['train']
    log_name = c.get('log_name')
    n_batch = c.getint('n_batch')
    n_worker = c.getint('n_worker')
    n_ckpt_epoch = c.getint('n_ckpt_epoch')
    if n_epoch is None:
        n_epoch = c.getint('n_epoch')

    # initialize model
    if model_name == 'unet_vgg16':
        model = UNetVgg16(3, 1, fixed_vgg=True)
    elif model_name == 'dcan':
        model = DCAN(3, 1)
    elif model_name == 'caunet':
        model = CAUNet()
    elif model_name == 'camunet':
        model = CAMUNet()
    else:
        model = UNet()

    if torch.cuda.is_available():
        model = model.cuda()
        # model = torch.nn.DataParallel(model).cuda()

    # define optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learn_rate,
        weight_decay=1e-6
        )

    # dataloader workers are forked process thus we need a IPC manager to keep cache in same memory space
    manager = Manager()
    cache = manager.dict()
    # prepare dataset and loader
    dataset = KaggleDataset('data/stage1_train', transform=Compose(), cache=cache)
    train_idx, valid_idx = dataset.split()
    train_loader = DataLoader(
        dataset, sampler=SubsetRandomSampler(train_idx),
        batch_size=n_batch,
        num_workers=n_worker,
        pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(
        dataset, sampler=SubsetRandomSampler(valid_idx),
        batch_size=n_batch,
        num_workers=n_worker)

    # resume checkpoint
    start_epoch = 0
    if resume:
        start_epoch = load_ckpt(model, optimizer)
    if start_epoch == 0:
        print('Grand new training ...')

    # decide log directory name
    log_dir = os.path.join(
        'logs', log_name, '{}-{}'.format(model_name, width),
        'ep_{},{}-lr_{}'.format(
            start_epoch,
            n_epoch + start_epoch,
            learn_rate,
        )
    )

    with SummaryWriter(log_dir) as writer:
        if start_epoch == 0 and False:
            # dump graph only for very first training, disable by default
            dump_graph(model, writer, n_batch, width)
        print('Training started...')
        for epoch in range(start_epoch, n_epoch + start_epoch):
            train(train_loader, model, optimizer, epoch, writer)
            if cv_ratio > 0 and epoch % 3 == 2:
                valid(valid_loader, model, epoch, writer, len(train_loader))
            # save checkpoint per n epoch
            if epoch % n_ckpt_epoch == n_ckpt_epoch - 1:
                save_ckpt(model, optimizer, epoch+1)
        print('Training finished...')

def dump_graph(model, writer, n_batch, width):
    # Prerequisite
    # $ sudo apt-get install libprotobuf-dev protobuf-compiler
    # $ pip3 install onnx
    print('Dump model graph...')
    dummy_input = Variable(torch.rand(n_batch, 3, width, width))
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
    torch.onnx.export(model, dummy_input, "checkpoint/model.pb", verbose=False)
    writer.add_graph_onnx("checkpoint/model.pb")

def train(loader, model, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    iou = AverageMeter()   # semantic IoU
    iou_c = AverageMeter() # contour IoU
    iou_m = AverageMeter() # marker IoU
    print_freq = config['train'].getfloat('print_freq')
    only_contour = config['contour'].getboolean('exclusive')
    weight_map = config['param'].getboolean('weight_map')

    # Sets the module in training mode.
    model.train()
    end = time.time()
    n_step = len(loader)
    for i, data in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # split sample data
        inputs, labels, labels_c, labels_m = data['image'], data['label'], data['label_c'], data['label_m']
        if torch.cuda.is_available():
            inputs, labels, labels_c, labels_m = \
                inputs.cuda(async=True), labels.cuda(async=True), labels_c.cuda(async=True), labels_m.cuda(async=True)
        # wrap them in Variable
        inputs, labels, labels_c, labels_m = Variable(inputs), Variable(labels), Variable(labels_c), Variable(labels_m)
        # get loss weight
        weights = None
        if weight_map and 'weight' in data:
            weights = data['weight']
            if torch.cuda.is_available():
                weights = weights.cuda(async=True)
            weights = Variable(weights)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward step
        outputs = model(inputs)
        if isinstance(model, CAMUNet):
            outputs, outputs_c, outputs_m = outputs
        elif isinstance(model, DCAN) or isinstance(model, CAUNet):
            outputs, outputs_c = outputs
        # compute loss
        if only_contour:
            loss = contour_criterion(outputs, labels_c)
        else:
            # weight_criterion equals to segment_criterion if weights is none
            loss = focal_criterion(outputs, labels, weights)
            if isinstance(model, CAMUNet):
                loss += focal_criterion(outputs_c, labels_c, weights)
                loss += focal_criterion(outputs_m, labels_m, weights)
            elif isinstance(model, DCAN) or isinstance(model, CAUNet):
                loss += focal_criterion(outputs_c, labels_c, weights)
        # compute gradient and do backward step
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # measure accuracy and record loss
        # NOT instance-level IoU in training phase, for better speed & instance separation handled in post-processing
        losses.update(loss.data[0], inputs.size(0))
        if only_contour:
            batch_iou = iou_mean(outputs, labels_c)
        else:
            batch_iou = iou_mean(outputs, labels)
        iou.update(batch_iou, inputs.size(0))
        if isinstance(model, CAMUNet):
            batch_iou_c, batch_iou_m = iou_mean(outputs_c, labels_c), iou_mean(outputs_m, labels_m)
            iou_c.update(batch_iou_c, inputs.size(0))
            iou_m.update(batch_iou_m, inputs.size(0))
        elif isinstance(model, DCAN) or isinstance(model, CAUNet):
            batch_iou_c = iou_mean(outputs_c, labels_c)
            iou_c.update(batch_iou_c, inputs.size(0))
        # log to summary
        step = i + epoch * n_step
        writer.add_scalar('training/loss', loss.data[0], step)
        writer.add_scalar('training/batch_elapse', batch_time.val, step)
        writer.add_scalar('training/batch_iou', iou.val, step)
        writer.add_scalar('training/batch_iou_c', iou_c.val, step)
        writer.add_scalar('training/batch_iou_m', iou_m.val, step)
        if (i + 1) % print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time: {batch_time.avg:.2f} (io: {data_time.avg:.2f})\t'
                'Loss: {loss.val:.4f} (avg: {loss.avg:.4f})\t'
                'IoU: {iou.avg:.3f} (Coutour: {iou_c.avg:.3f}, Marker: {iou_m.avg:.3f})\t'
                .format(
                    epoch, i, n_step, batch_time=batch_time,
                    data_time=data_time, loss=losses, iou=iou, iou_c=iou_c, iou_m=iou_m
                )
            )
    # end of loop, dump epoch summary
    writer.add_scalar('training/epoch_loss', losses.avg, epoch)
    writer.add_scalar('training/epoch_iou', iou.avg, epoch)
    writer.add_scalar('training/epoch_iou_c', iou_c.avg, epoch)
    writer.add_scalar('training/epoch_iou_m', iou_m.avg, epoch)

def valid(loader, model, epoch, writer, n_step):
    iou = AverageMeter()   # semantic IoU
    iou_c = AverageMeter() # contour IoU
    iou_m = AverageMeter() # marker IoU
    losses = AverageMeter()
    only_contour = config['contour'].getboolean('exclusive')
    weight_map = config['param'].getboolean('weight_map')

    # Sets the model in evaluation mode.
    model.eval()
    for i, data in enumerate(loader):
        # get the inputs
        inputs, labels, labels_c, labels_m = data['image'], data['label'], data['label_c'], data['label_m']
        if torch.cuda.is_available():
            inputs, labels, labels_c, labels_m = inputs.cuda(), labels.cuda(), labels_c.cuda(), labels_m.cuda()
        # wrap them in Variable
        inputs, labels, labels_c, labels_m = Variable(inputs), Variable(labels), Variable(labels_c), Variable(labels_m)
        # get loss weight
        weights = None
        if weight_map and 'weight' in data:
            weights = data['weight']
            if torch.cuda.is_available():
                weights = weights.cuda(async=True)
            weights = Variable(weights)
        # forward step
        outputs = model(inputs)
        if isinstance(model, CAMUNet):
            outputs, outputs_c, outputs_m = outputs
        elif isinstance(model, DCAN) or isinstance(model, CAUNet):
            outputs, outputs_c = outputs
        # compute loss
        if only_contour:
            loss = contour_criterion(outputs, labels_c)
        else:
            # weight_criterion equals to segment_criterion if weights is none
            loss = focal_criterion(outputs, labels, weights)
            if isinstance(model, CAMUNet):
                loss += focal_criterion(outputs_c, labels_c, weights)
                loss += focal_criterion(outputs_m, labels_m, weights)
            if isinstance(model, DCAN) or isinstance(model, CAUNet):
                loss += focal_criterion(outputs_c, labels_c, weights)
        # measure accuracy and record loss (Non-instance level IoU)
        losses.update(loss.data[0], inputs.size(0))
        if only_contour:
            batch_iou = iou_mean(outputs, labels_c)
        else:
            batch_iou = iou_mean(outputs, labels)
        iou.update(batch_iou, inputs.size(0))
        if isinstance(model, CAMUNet):
            batch_iou_c, batch_iou_m = iou_mean(outputs_c, labels_c), iou_mean(outputs_m, labels_m)
            iou_c.update(batch_iou_c, inputs.size(0))
            iou_m.update(batch_iou_m, inputs.size(0))
        elif isinstance(model, DCAN) or isinstance(model, CAUNet):
            batch_iou_c = iou_mean(outputs_c, labels_c)
            iou_c.update(batch_iou_c, inputs.size(0))
    # end of loop, dump epoch summary
    writer.add_scalar('CV/epoch_loss', losses.avg, epoch)
    writer.add_scalar('CV/epoch_iou', iou.avg, epoch)
    writer.add_scalar('CV/epoch_iou_c', iou_c.avg, epoch)
    writer.add_scalar('CV/epoch_iou_m', iou_m.avg, epoch)
    print(
        'Epoch: [{0}]\t\tcross-validation\t'
        'Loss: N/A    (avg: {loss.avg:.4f})\t'
        'IoU: {iou.avg:.3f} (Coutour: {iou_c.avg:.3f}, Marker: {iou_m.avg:.3f})\t'
        .format(
            epoch, loss=losses, iou=iou, iou_c=iou_c, iou_m=iou_m
        )
    )

if __name__ == '__main__':
    learn_rate = config['param'].getfloat('learn_rate')
    n_epoch = config['train'].getint('n_epoch')
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--epoch', type=int, help='run number of epoch')
    parser.add_argument('--lr', type=float, dest='learn_rate', help='learning rate')
    parser.set_defaults(resume=True, epoch=n_epoch, learn_rate=learn_rate)
    args = parser.parse_args()

    main(args.resume, args.epoch, args.learn_rate)
