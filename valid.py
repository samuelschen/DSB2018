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
from model import UNet, UNetVgg16, DCAN
from dataset import KaggleDataset, Compose
from helper import load_ckpt, prob_to_rles, seg_ws
from skimage.morphology import label

def main(args):
    if args.model == 'unet_vgg16':
        model = UNetVgg16(3, 1, fixed_vgg=True)
    elif args.model == 'dcan':
        model = DCAN(3, 1)
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
    dataset = KaggleDataset('data/stage1_test', transform=compose)
    # dataset = KaggleDataset('data/stage1_test', transform=compose, category='Histology')
    iter = predict(model, dataset, compose)

    if args.csv:
        with open('result.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ImageId', 'EncodedPixels'])
            for _, y, uid in iter:
                for rle in prob_to_rles(y):
                    writer.writerow([uid, ' '.join([str(i) for i in rle])])
    else:
        for x, y, uid in iter:
            show(x, y, uid)

def predict(model, dataset, compose, regrowth=True):
    ''' iterate dataset and yield ndarray result tuple per sample '''
    for data in dataset:
        # get prediction
        uid = data['uid']
        inputs = x = data['image']
        inputs = inputs.unsqueeze(0)
        if config.cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        if isinstance(model, DCAN):
            outputs_s, outputs_c = model(inputs)
            # measure accuracy and record loss
            cond1 = (outputs_s >= config.threshold_sgmt)
            cond2 = (outputs_c < config.threshold_edge)
            outputs = (cond1 * cond2)
        else:
            outputs = model(inputs)
        # convert image to numpy array
        x = compose.denorm(x)
        x = compose.pil(x)
        if regrowth:
            x = x.resize(data['size'])
        x = np.asarray(x)
        # convert predict to numpy array
        if config.cuda:
            outputs = outputs.cpu()
        y = outputs.data.numpy()[0]
        y = np.transpose(y, (1, 2, 0))
        y = np.squeeze(y)
        if regrowth:
            y = resize(y, data['size'][::-1], mode='constant', preserve_range=True)
        # yield result
        yield x, y, uid

def show(x, y, uid):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(14, 6))
    fig.suptitle(uid, y=1)
    ax1.set_title('image')
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store', choices=['unet', 'unet_vgg16', 'dcan'], help='model name')
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