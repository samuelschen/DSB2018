import os
import json
import numpy as np
import torch
from skimage.morphology import label
import config

# copy from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# copy from https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > config.threshold)
    y_pred = label(y_pred_in > config.threshold)

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_mean(y_true_in, y_pred_in):
    y_true_in = y_true_in.data.cpu().numpy()
    y_pred_in = y_pred_in.data.cpu().numpy()
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(y):
    dots = np.where(y.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(y):
    lab_img = label(y > config.threshold)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

# checkpoint handling
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

def save_ckpt(model, optimizer, epoch):
    ckpt = ckpt_path(epoch)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, ckpt)

def load_ckpt(model, optimizer=None):
    ckpt = ckpt_path()
    epoch = 0
    if os.path.isfile(ckpt):
        print("Loading checkpoint '{}'".format(ckpt))
        checkpoint = torch.load(ckpt)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
    return epoch