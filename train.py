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
from model import UNet, UNetVgg16, CAUNet, DCAN
from dataset import KaggleDataset, NuclearDataset, Compose
from helper import config, AverageMeter, iou_mean, save_ckpt, load_ckpt
from loss import criterion, segment_criterion, contour_criterion, weight_criterion


def main(resume=True, n_epoch=None, learn_rate=None):
    model_name = config['param']['model']
    cv_ratio = config['param'].getfloat('cv_ratio')
    if learn_rate is None:
        learn_rate = config['param'].getfloat('learn_rate')
    width = config[model_name].getint('width')
    cell_level = config['param'].getboolean('cell_level')
    weight_bce = config['param'].getboolean('weight_bce')
    c = config['train']
    log_name = c.get('log_name')
    n_batch = c.getint('n_batch')
    n_worker = c.getint('n_worker')
    n_ckpt_epoch = c.getint('n_ckpt_epoch')
    if n_epoch is None:
        n_epoch = c.getint('n_epoch')

    if model_name == 'unet_vgg16':
        model = UNetVgg16(3, 1, fixed_vgg=True)
    elif model_name == 'dcan':
        model = DCAN(3, 1)
    elif model_name == 'caunet':
        model = CAUNet()
    else:
        model = UNet()

    if torch.cuda.is_available():
        model = model.cuda()
        # model = torch.nn.DataParallel(model).cuda()

    if isinstance(model, DCAN) or isinstance(model, CAUNet):
        if weight_bce:
            cost = (weight_criterion, weight_criterion)
        else:
            cost = (segment_criterion, contour_criterion)
    elif weight_bce:
        cost = weight_criterion
    else:
        cost = criterion

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learn_rate,
        weight_decay=1e-6
        )

    # dataloader workers are forked process thus we need a IPC manager to keep cache in same memory space
    manager = Manager()
    cache = manager.dict()
    # prepare dataset and loader
    if cell_level:
        dataset = NuclearDataset('data/stage1_train', transform=Compose(), cache=cache)
    else:
        dataset = KaggleDataset('data/stage1_train', transform=Compose(), cache=cache)
        # dataset = KaggleDataset('data/stage1_train', transform=Compose(), cache=cache, category='Histology')
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
            train(train_loader, model, cost, optimizer, epoch, writer)
            if cv_ratio > 0 and epoch % 3 == 2:
                valid(valid_loader, model, cost, epoch, writer, len(train_loader))
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

def train(loader, model, cost, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    iou = AverageMeter()    # semantic IoU
    if isinstance(model, DCAN) or isinstance(model, CAUNet):
        iou_c = AverageMeter() # contour IoU
    print_freq = config['train'].getfloat('print_freq')
    only_contour = config['pre'].getboolean('train_contour_only')
    weight_bce = config['param'].getboolean('weight_bce')

    # Sets the module in training mode.
    model.train()
    end = time.time()
    n_step = len(loader)
    for i, data in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # get the inputs
        inputs, labels, labels_e = data['image'], data['label'], data['label_e']
        if torch.cuda.is_available():
            inputs, labels, labels_e = inputs.cuda(async=True), labels.cuda(async=True), labels_e.cuda(async=True)
        # wrap them in Variable
        inputs, labels, labels_e = Variable(inputs), Variable(labels), Variable(labels_e)
        # get loss weight
        if weight_bce and 'weight' in data:
            weights = data['weight']
            if torch.cuda.is_available():
                weights = weights.cuda(async=True)
            weights = Variable(weights)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward step
        if isinstance(model, DCAN) or isinstance(model, CAUNet):
            outputs, outputs_c = model(inputs)
            if weight_bce:
                loss = cost[0](outputs, labels, weights) + cost[1](outputs_c, labels_e, weights)
            else:
                loss = cost[0](outputs, labels) + cost[1](outputs_c, labels_e)
            batch_iou_c = iou_mean(outputs_c, labels_e)
            iou_c.update(batch_iou_c, inputs.size(0))
        else:
            outputs = model(inputs)
            if only_contour:
                loss = cost(outputs, labels_e)
            elif weight_bce:
                loss = cost(outputs, labels, weights)
            else:
                loss = cost(outputs, labels)

        # measure accuracy and record loss
        # NOT instance-level IoU in training phase, for better speed & instance separation handled in post-processing
        batch_iou = iou_mean(outputs, labels_e) if only_contour else iou_mean(outputs, labels)
        iou.update(batch_iou, inputs.size(0))

        losses.update(loss.data[0], inputs.size(0))
        # compute gradient and do backward step
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # log to summary
        step = i + epoch * n_step
        writer.add_scalar('training/loss', loss.data[0], step)
        writer.add_scalar('training/batch_elapse', batch_time.val, step)
        writer.add_scalar('training/batch_iou', iou.val, step)
        if isinstance(model, DCAN) or isinstance(model, CAUNet):
            writer.add_scalar('training/batch_iou_c', iou_c.val, step)
            if (i + 1) % print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.avg:.3f} (io: {data_time.avg:.3f})\t\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'IoU(Semantic): {iou.val:.3f} ({iou.avg:.3f})\t'
                    'IoU(Contour): {iou_c.val:.3f} ({iou_c.avg:.3f})\t'
                    .format(
                        epoch, i, n_step, batch_time=batch_time,
                        data_time=data_time, loss=losses, iou=iou, iou_c=iou_c
                    )
                )
        else:
            if (i + 1) % print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.avg:.3f} (io: {data_time.avg:.3f})\t\t'
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'IoU: {iou.val:.3f} ({iou.avg:.3f})\t'
                    .format(
                        epoch, i, n_step, batch_time=batch_time,
                        data_time=data_time, loss=losses, iou=iou
                    )
                )
    writer.add_scalar('training/epoch_loss', losses.avg, epoch)
    writer.add_scalar('training/epoch_iou', iou.avg, epoch)
    if isinstance(model, DCAN) or isinstance(model, CAUNet):
        writer.add_scalar('training/epoch_iou_c', iou_c.avg, epoch)

def valid(loader, model, cost, epoch, writer, n_step):
    iou = AverageMeter() # semantic IoU
    if isinstance(model, DCAN) or isinstance(model, CAUNet):
        iou_c = AverageMeter() # contour IoU
    losses = AverageMeter()
    only_contour = config['pre'].getboolean('train_contour_only')
    weight_bce = config['param'].getboolean('weight_bce')

    # Sets the model in evaluation mode.
    model.eval()
    for i, data in enumerate(loader):
        # get the inputs
        inputs, labels, labels_e = data['image'], data['label'], data['label_e']
        if torch.cuda.is_available():
            inputs, labels, labels_e = inputs.cuda(), labels.cuda(), labels_e.cuda()
        # wrap them in Variable
        inputs, labels, labels_e = Variable(inputs), Variable(labels), Variable(labels_e)
        # get loss weight
        if weight_bce and 'weight' in data:
            weights = data['weight']
            if torch.cuda.is_available():
                weights = weights.cuda(async=True)
            weights = Variable(weights)

        # forward step
        if isinstance(model, DCAN) or isinstance(model, CAUNet):
            outputs, outputs_c = model(inputs)
            if weight_bce:
                loss = cost[0](outputs, labels, weights) + cost[1](outputs_c, labels_e, weights)
            else:
                loss = cost[0](outputs, labels) + cost[1](outputs_c, labels_e)
            # measure accuracy and record loss
            batch_iou_c = iou_mean(outputs_c, labels_e)
            iou_c.update(batch_iou_c, inputs.size(0))
        else:
            outputs = model(inputs)
            if only_contour:
                loss = cost(outputs, labels_e)
            elif weight_bce:
                loss = cost(outputs, labels, weights)
            else:
                loss = cost(outputs, labels)

        # measure accuracy and record loss (Non-instance level IoU)
        batch_iou = iou_mean(outputs, labels_e) if only_contour else iou_mean(outputs, labels)
        iou.update(batch_iou, inputs.size(0))
        losses.update(loss.data[0], inputs.size(0))

    # log to summary
    writer.add_scalar('CV/epoch_loss', losses.avg, epoch)
    writer.add_scalar('CV/epoch_iou', iou.avg, epoch)
    if isinstance(model, DCAN) or isinstance(model, CAUNet):
        writer.add_scalar('CV/epoch_iou_c', iou_c.avg, epoch)
        print(
            'Epoch: [{0}]\t\tcross-validation\t\t'
            'Loss: N/A    ({loss.avg:.4f})\t'
            'IoU(Semantic): N/A   ({iou.avg:.3f})\t'
            'IoU(Contour): N/A   ({iou_c.avg:.3f})\t'
            .format(
                epoch, loss=losses, iou=iou, iou_c=iou_c
            )
        )
    else:
        print(
            'Epoch: [{0}]\t\tcross-validation\t\t'
            'Loss: N/A    ({loss.avg:.4f})\t'
            'IoU: N/A   ({iou.avg:.3f})\t'
            .format(
                epoch, loss=losses, iou=iou
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
