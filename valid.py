# python built-in library
import argparse
import time
import csv
from multiprocessing import Manager
# 3rd party library
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.transform import resize
# own code
import config
from model import UNet, UNetVgg16, DCAN, CAUNet
from dataset import KaggleDataset, Compose
from helper import load_ckpt, prob_to_rles, seg_ws
from skimage.morphology import label

def main(args):
    if args.model == 'unet_vgg16':
        model = UNetVgg16(3, 1, fixed_vgg=True)
    elif args.model == 'dcan':
        model = DCAN(3, 1)
    elif args.model == 'caunet':
        model = CAUNet()
    else:
        model = UNet()
    if config.cuda:
        model = model.cuda()
    # Sets the model in evaluation mode.
    model.eval()

    epoch = load_ckpt(model)
    if epoch == 0:
        print("Aborted: checkpoint not found!")
        return

    # prepare dataset
    compose = Compose(augment=False)
    data_dir = 'data/stage1_test' if args.dataset == 'test' else 'data/stage1_train'
    dataset = KaggleDataset(data_dir, transform=compose)
    # dataset = KaggleDataset(data_dir, transform=compose, category='Histology')
    iter = predict(model, dataset, compose)

    if args.csv:
        with open('result.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ImageId', 'EncodedPixels'])
            for uid, _, y, _, _, _, _, _ in iter:
                for rle in prob_to_rles(y):
                    writer.writerow([uid, ' '.join([str(i) for i in rle])])
    else:
        for uid, x, y, gt, y_s, gt_s, y_c, gt_c in iter:
            if args.dataset == 'test':
                show(uid, x, y)
            else:
                show_groundtruth(uid, x, y, gt, y_s, gt_s, y_c, gt_c)

def predict(model, dataset, compose, regrowth=True):
    ''' iterate dataset and yield ndarray result tuple per sample '''
    for data in dataset:
        # get prediction
        uid = data['uid']
        inputs = x = data['image']
        gt_s, gt_c, gt = data['label'], data['label_e'], data['label_gt']
        inputs = inputs.unsqueeze(0)
        if config.cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        if isinstance(model, DCAN) or isinstance(model, CAUNet):
            outputs_s, outputs_c = model(inputs)
            cond1 = (outputs_s >= config.threshold_sgmt)
            cond2 = (outputs_c < config.threshold_edge)
            outputs = (cond1 * cond2)
        else:
            outputs = model(inputs)

        # convert image to numpy array
        x = compose.denorm(x)
        x = compose.pil(x)
        gt_s = compose.pil(gt_s)
        gt_c = compose.pil(gt_c)
        gt = compose.pil(gt)
        if regrowth:
            x = x.resize(data['size'])
            gt_s = gt_s.resize(data['size'])
            gt_c = gt_c.resize(data['size'])
            gt = gt.resize(data['size'])

        x = np.asarray(x)
        gt_s = np.asarray(gt_s)
        gt_c = np.asarray(gt_c)
        gt = np.asarray(gt)

        # convert predict to numpy array
        if config.cuda:
            outputs = outputs.cpu()
            if isinstance(model, DCAN) or isinstance(model, CAUNet):
                outputs_s = outputs_s.cpu()
                outputs_c = outputs_c.cpu()

        y = outputs.data.numpy()[0]
        y = np.transpose(y, (1, 2, 0))
        y = np.squeeze(y)
        if regrowth:
            y = resize(y, data['size'][::-1], mode='constant', preserve_range=True)

        if isinstance(model, DCAN) or isinstance(model, CAUNet):
            y_s = outputs_s.data.numpy()[0]
            y_s = np.transpose(y_s, (1, 2, 0))
            y_s = np.squeeze(y_s)
            y_c = outputs_c.data.numpy()[0]
            y_c = np.transpose(y_c, (1, 2, 0))
            y_c = np.squeeze(y_c)
            if regrowth:
                y_s = resize(y_s, data['size'][::-1], mode='constant', preserve_range=True)
                y_c = resize(y_c, data['size'][::-1], mode='constant', preserve_range=True)
        else:
            y_s = y_c = None

        # yield result
        yield uid, x, y, gt, y_s, gt_s, y_c, gt_c

def show(uid, x, y):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(14, 6))
    fig.suptitle(uid, y=1)
    ax1.set_title('Image')
    ax2.set_title('Predict, P > {}'.format(config.threshold))
    ax3.set_title('Region, P > {}'.format(config.threshold))
    ax4.set_title('Overlay, P > {}'.format(config.threshold))
    ax1.imshow(x)
    y = y > config.threshold
    ax2.imshow(y, cmap='gray')
    y = label(y)
    if config.post_segmentation:
        y = seg_ws(y)
    y = y.astype(float)
    y[y == 0] = np.nan # workaround: matplotlib cmap mistreat vmin(1) as background(0) sometimes
    cmap = plt.get_cmap('prism') # prism for high frequence color bands
    cmap.set_bad('w', alpha=0) # map background(0) as transparent/white
    ax3.imshow(y, cmap=cmap)
    # alpha 
    ax4.imshow(x)
    ax4.imshow(y, cmap=cmap, alpha=0.3)
    plt.tight_layout()
    plt.show()

def show_groundtruth(uid, x, y, gt, y_s, gt_s, y_c, gt_c):
    if y_s is not None and y_c is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 4, sharey=True, figsize=(21, 9))
    else:
        fig, ax1 = plt.subplots(1, 4, sharey=True, figsize=(14, 6))
    fig.suptitle(uid, y=1)
    ax1[0].set_title('Image')
    ax1[1].set_title('Final Predict, P > {}'.format(config.threshold))
    ax1[2].set_title('Instance Ground Truth')
    ax1[3].set_title('Overlay (Instance), P > {}'.format(config.threshold))
    ax1[0].imshow(x)
    y = y > config.threshold
    ax1[1].imshow(y, cmap='gray')
    ax1[2].imshow(gt)
    # overlay
    ax1[3].imshow(gt)
    y2 = label(y)
    if config.post_segmentation:
        y2 = seg_ws(y2)
    y2 = y2.astype(float)
    y2[y2 == 0] = np.nan # workaround: matplotlib cmap mistreat vmin(1) as background(0) sometimes
    cmap = plt.get_cmap('prism') # prism for high frequence color bands
    cmap.set_bad('w', alpha=0) # map background(0) as transparent/white
    ax1[3].imshow(y2, cmap=cmap, alpha=0.3)

    if y_s is not None and y_c is not None:
        ax2[0].set_title('Image')
        ax2[1].set_title('Semantic Predict, P > {}'.format(config.threshold))
        ax2[2].set_title('Semantic Ground Truth')
        ax2[3].set_title('Overlay (Semantic), P > {}'.format(config.threshold_sgmt))
        ax2[0].imshow(x)
        y_s = y_s > config.threshold_sgmt
        ax2[1].imshow(y_s, cmap='gray')
        ax2[2].imshow(gt_s, cmap='gray')
        # overlay
        ax2[3].imshow(gt_s, cmap='gray')
        y_s2 = label(y_s)
        if config.post_segmentation:
            y_s2 = seg_ws(y_s2)
        y_s2 = y_s2.astype(float)
        y_s2[y_s2 == 0] = np.nan # workaround: matplotlib cmap mistreat vmin(1) as background(0) sometimes
        cmap = plt.get_cmap('prism') # prism for high frequence color bands
        cmap.set_bad('w', alpha=0) # map background(0) as transparent/white
        ax2[3].imshow(y_s2, cmap=cmap, alpha=0.3)

        ax3[0].set_title('image')
        ax3[1].set_title('Contour Predict, P > {}'.format(config.threshold))
        ax3[2].set_title('Contour Ground Truth')
        ax3[3].set_title('Overlay (Contour), P > {}'.format(config.threshold_edge))
        ax3[0].imshow(x)
        y_c = y_c > config.threshold_edge
        ax3[1].imshow(y_c, cmap='gray')
        ax3[2].imshow(gt_c, cmap='gray')
        # overlay
        ax3[3].imshow(gt_c, cmap='gray')
        y_c2 = label(y_c)
        if config.post_segmentation:
            y_c2 = seg_ws(y_c2)
        y_c2 = y_c2.astype(float)
        y_c2[y_c2 == 0] = np.nan # workaround: matplotlib cmap mistreat vmin(1) as background(0) sometimes
        cmap = plt.get_cmap('prism') # prism for high frequence color bands
        cmap.set_bad('w', alpha=0) # map background(0) as transparent/white
        ax3[3].imshow(y_c2, cmap=cmap, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', choices=['unet', 'unet_vgg16', 'caunet', 'dcan'], help='model name', required=True)
    parser.add_argument('--dataset', action='store', choices=['train', 'test'], help='dataset to eval', required=True)
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--csv', dest='csv', action='store_true')
    parser.add_argument('--show', dest='csv', action='store_false')
    parser.add_argument('--width', type=int, help='width of image to evaluate')
    parser.set_defaults(cuda=config.cuda, width=config.width, csv=False)
    args = parser.parse_args()

    config.width = args.width
    # final check whether cuda is avaiable
    config.cuda = torch.cuda.is_available() and args.cuda

    main(args)