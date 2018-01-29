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
from model import Model
from dataset import KaggleDataset, Compose
from helper import load_ckpt, prob_to_rles
from skimage.morphology import label

def main(args):
    model = Model()
    if config.cuda:
        model = model.cuda()
    # Sets the module in evaluation mode.
    model.eval()

    epoch = load_ckpt(model)
    if epoch == 0:
        print("Aborted: checkpoint not found!")
        return

    # prepare dataset
    compose = Compose(argument=False)
    dataset = KaggleDataset('data/stage1_test', transform=compose)
    iter = predict(model, dataset, compose)

    if args.csv:
        with open('result.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ImageId', 'EncodedPixels'])
            for _, y, uid in iter:
                for rle in prob_to_rles(y):
                    writer.writerow([uid, ' '.join([str(i) for i in rle])])
    else:
        for x, y, _ in iter:
            show(x, y)

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

def show(x, y):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(10, 6))
    ax1.set_title('Image')
    ax2.set_title('Predict, P > {}'.format(config.threshold))
    ax3.set_title('Region, P > {}'.format(config.threshold))
    ax1.imshow(x)
    y = y > config.threshold
    ax2.imshow(y, cmap='gray')
    y = label(y)
    ax3.imshow(y, cmap='Set3')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.add_argument('--csv', dest='csv', action='store_true')
    parser.add_argument('--show', dest='csv', action='store_false')
    parser.add_argument('--width', type=int, help='width of image to evaluate')
    parser.set_defaults(cuda=config.cuda, width=512, csv=False)
    args = parser.parse_args()

    config.width = args.width
    # final check whether cuda is avaiable
    config.cuda = torch.cuda.is_available() and args.cuda

    main(args)