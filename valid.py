# python built-in library
import os
import argparse
import time
import csv
import uuid
# 3rd party library
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from skimage.transform import resize
from skimage.morphology import label, remove_small_objects
from tqdm import tqdm
from PIL import Image
# own code
from model import UNet, UNetVgg16, DCAN, CAUNet, CAMUNet
from dataset import KaggleDataset, Compose
from helper import config, load_ckpt, prob_to_rles, seg_ws, seg_ws_by_edge, seg_ws_by_marker, iou_metric

def main(tocsv=False, save=False, mask=False, valid_train=False):
    model_name = config['param']['model']
    use_padding = config['valid'].getboolean('pred_orig_size')

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

    # Sets the model in evaluation mode.
    model.eval()

    epoch = load_ckpt(model)
    if epoch == 0:
        print("Aborted: checkpoint not found!")
        return

    # prepare dataset
    compose = Compose(augment=False, padding=use_padding)
    data_dir = 'data/stage1_train' if valid_train else 'data/stage1_test'
    dataset = KaggleDataset(data_dir, transform=compose)
    # dataset = KaggleDataset(data_dir, transform=compose, category='Histology')
    iter = predict(model, dataset, compose, not use_padding)

    if tocsv:
        with open('result.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ImageId', 'EncodedPixels'])
            for uid, _, y, y_c, _, _, _ in iter:
                for rle in prob_to_rles(y, y_c):
                    writer.writerow([uid, ' '.join([str(i) for i in rle])])
    else:
        for uid, x, y, y_c, y_m, gt, gt_s, gt_c, gt_m in tqdm(iter):
            if valid_train:
                show_groundtruth(uid, x, y, y_c, y_m, gt, gt_s, gt_c, gt_m, save)
            elif mask:
                save_mask(uid, y, y_c, y_m)
            else:
                show(uid, x, y, y_c, y_m, save)

def predict(model, dataset, compose, regrowth=True):
    ''' iterate dataset and yield ndarray result tuple per sample '''
    for data in dataset:
        # get prediction
        uid = data['uid']
        size = data['size']
        inputs = x = data['image']
        gt_s, gt_c, gt_m, gt = data['label'], data['label_c'], data['label_m'], data['label_gt']
        inputs = inputs.unsqueeze(0)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        if isinstance(model, CAMUNet):
            outputs, outputs_c, outputs_m = model(inputs)
        elif isinstance(model, DCAN) or isinstance(model, CAUNet):
            outputs, outputs_c = model(inputs)
        else:
            outputs = model(inputs)

        # convert image to numpy array
        x = compose.denorm(x)
        x = compose.pil(x)
        gt_s = compose.pil(gt_s)
        gt_c = compose.pil(gt_c)
        gt_m = compose.pil(gt_m)
        gt = compose.pil(gt)
        if regrowth:
            x = x.resize(size)
            gt_s = gt_s.resize(size)
            gt_c = gt_c.resize(size)
            gt_m = gt_m.resize(size)
            gt = gt.resize(size)
        else:
            rect = (0, 0, size[0], size[1])
            x = x.crop(rect)
            gt_s = gt_s.crop(rect)
            gt_c = gt_c.crop(rect)
            gt_m = gt_m.crop(rect)
            gt = gt.crop(rect)

        x = np.asarray(x)
        gt_s = np.asarray(gt_s)
        gt_c = np.asarray(gt_c)
        gt_m = np.asarray(gt_m)
        gt = np.asarray(gt)

        # convert predict to numpy array
        if torch.cuda.is_available():
            outputs = outputs.cpu()
            if isinstance(model, CAMUNet):
                outputs_c, outputs_m = outputs_c.cpu(), outputs_m.cpu()
            if isinstance(model, DCAN) or isinstance(model, CAUNet):
                outputs_c = outputs_c.cpu()

        y = outputs.data.numpy()[0]
        y = np.transpose(y, (1, 2, 0))
        y = np.squeeze(y)
        if regrowth:
            y = resize(y, size[::-1], mode='constant', preserve_range=True)
        else:
            w, h = size
            y = y[:h, :w]

        if isinstance(model, CAMUNet):
            y_c, y_m = outputs_c.data.numpy()[0], outputs_m.data.numpy()[0]
            y_c, y_m = np.transpose(y_c, (1, 2, 0)), np.transpose(y_m, (1, 2, 0))
            y_c, y_m = np.squeeze(y_c), np.squeeze(y_m)
            if regrowth:
                y_c = resize(y_c, size[::-1], mode='constant', preserve_range=True)
                y_m = resize(y_m, size[::-1], mode='constant', preserve_range=True)
            else:
                w, h = size
                y_c, y_m = y_c[:h, :w], y_m[:h, :w]
        elif isinstance(model, DCAN) or isinstance(model, CAUNet):
            y_c = outputs_c.data.numpy()[0]
            y_c = np.transpose(y_c, (1, 2, 0))
            y_c = np.squeeze(y_c)
            if regrowth:
                y_c = resize(y_c, size[::-1], mode='constant', preserve_range=True)
            else:
                w, h = size
                y_c = y_c[:h, :w]
            y_m = None
        else:
            y_c = y_m = None

        yield uid, x, y, y_c, y_m, gt, gt_s, gt_c, gt_m

def _make_overlay(img_array):
    img_array = img_array.astype(float)
    img_array[img_array == 0] = np.nan # workaround: matplotlib cmap mistreat vmin(1) as background(0) sometimes
    cmap = plt.get_cmap('prism') # prism for high frequence color bands
    cmap.set_bad('w', alpha=0) # map background(0) as transparent/white
    return img_array, cmap

def show_figure():
    backend = matplotlib.get_backend()
    _x = config['valid'].getint('figure_pos_x')
    _y = config['valid'].getint('figure_pos_y')
    mgr = plt.get_current_fig_manager()
    if backend == 'TkAgg':
        mgr.window.wm_geometry("+%d+%d" % (_x, _y))
    elif backend == 'WXAgg':
        mgr.window.SetPosition((_x, _y))
    elif backend == 'Qt5Agg':
        mgr.window.move(_x, _y)
    else:
        # jupyter notebook etc.
        pass
    plt.show()

def show(uid, x, y, y_c, y_m, save=False):
    threshold = config['param'].getfloat('threshold')
    segmentation = config['post'].getboolean('segmentation')
    remove_objects = config['post'].getboolean('remove_objects')
    min_object_size = config['post'].getint('min_object_size')
    model_name = config['param']['model']
    if model_name == 'camunet':
        threshold_edge = config[model_name].getfloat('threshold_edge')
        threshold_mark = config[model_name].getfloat('threshold_mark')
    elif model_name == 'dcan' or model_name == 'caunet':
        threshold_edge = config[model_name].getfloat('threshold_edge')

    fig, (ax1, ax2) = plt.subplots(2, 3, sharey=True, figsize=(10, 8))
    fig.suptitle(uid, y=1)
    ax1[0].set_title('Image')
    ax1[1].set_title('Final Pred, P > {}'.format(threshold))
    ax1[2].set_title('Overlay, P > {}'.format(threshold))
    y_bw = y > threshold
    ax1[0].imshow(x, aspect='auto')
    if segmentation:
        if y_m is not None:
            y, markers = seg_ws_by_marker(y, y_m)
        elif y_c is not None:
            y, markers = seg_ws_by_edge(y, y_c)
        else:
            y, markers = seg_ws(y)
    if remove_objects:
        y = remove_small_objects(y, min_size=min_object_size)
    y, cmap = _make_overlay(y)
    ax1[1].imshow(y, cmap=cmap, aspect='auto')
    # alpha
    ax1[2].imshow(x, aspect='auto')
    ax1[2].imshow(y, cmap=cmap, alpha=0.3, aspect='auto')

    ax2[0].set_title('Semantic Pred, P > {}'.format(threshold))
    ax2[0].imshow(y_bw, cmap='gray', aspect='auto')
    _, count = label(markers, return_num=True)
    ax2[1].set_title('Markers, #={}'.format(count))
    ax2[1].imshow(markers, cmap='gray', aspect='auto')
    if y_c is not None:
        ax2[2].set_title('Contour Pred, P > {}'.format(threshold_edge))
        y_c = y_c > threshold_edge
        ax2[2].imshow(y_c, cmap='gray', aspect='auto')
    plt.tight_layout()

    if save:
        dir = predict_save_folder()
        fp = os.path.join(dir, uid + '.png')
        plt.savefig(fp)
    else:
        show_figure()

def show_groundtruth(uid, x, y, y_c, y_m, gt, gt_s, gt_c, gt_m, save=False):
    threshold = config['param'].getfloat('threshold')
    segmentation = config['post'].getboolean('segmentation')
    remove_objects = config['post'].getboolean('remove_objects')
    min_object_size = config['post'].getint('min_object_size')
    only_contour = config['contour'].getboolean('exclusive')
    model_name = config['param']['model']
    if model_name == 'camunet':
        threshold_edge = config[model_name].getfloat('threshold_edge')
        threshold_mark = config[model_name].getfloat('threshold_mark')
    elif model_name == 'dcan' or model_name == 'caunet':
        threshold_edge = config[model_name].getfloat('threshold_edge')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 4, sharey=True, figsize=(12, 8))
    fig.suptitle(uid, y=1)

    y_s = y # to show pure semantic predict later
    ax1[0].set_title('Image')
    ax1[0].imshow(x, aspect='auto')
    if segmentation :
        if y_m is not None:
            y, markers = seg_ws_by_marker(y, y_m)
        elif y_c is not None:
            y, markers = seg_ws_by_edge(y, y_c)
        else:
            y, markers = seg_ws(y)
    if remove_objects:
        y = remove_small_objects(y, min_size=min_object_size)
    _, count = label(y, return_num=True)
    ax1[1].set_title('Final Pred, #={}'.format(count))
    ax1[1].imshow(y, cmap='gray', aspect='auto')
    # overlay contour to semantic ground truth (another visualized view for instance ground truth, eg. gt)
    _, count = label(gt, return_num=True)
    ax1[2].set_title('Instance Lbls, #={}'.format(count))
    ax1[2].imshow(gt_s, cmap='gray', aspect='auto')
    gt_c2, cmap = _make_overlay(gt_c)
    ax1[2].imshow(gt_c2, cmap=cmap, alpha=0.7, aspect='auto')
    if only_contour: # can not tell from instances in this case
        iou = iou_metric(y, gt)
    else:
        iou = iou_metric(y, gt, instance_level=True)
    ax1[3].set_title('Overlay, IoU={:.3f}'.format(iou))
    ax1[3].imshow(gt_s, cmap='gray', aspect='auto')
    y, cmap = _make_overlay(y)
    ax1[3].imshow(y, cmap=cmap, alpha=0.3, aspect='auto')

    y_s = y_s > threshold
    _, count = label(y_s, return_num=True)
    ax2[0].set_title('Semantic Predict, #={}'.format(count))
    ax2[0].imshow(y_s, cmap='gray', aspect='auto')
    _, count = label(gt_s, return_num=True)
    ax2[1].set_title('Semantic Lbls, #={}'.format(count))
    ax2[1].imshow(gt_s, cmap='gray', aspect='auto')

    if y_c is not None:
        y_c = y_c > threshold_edge
        _, count = label(y_c, return_num=True)
        ax2[2].set_title('Contour Predict, #={}'.format(count))
        ax2[2].imshow(y_c, cmap='gray', aspect='auto')
        _, count = label(gt_c, return_num=True)
        ax2[3].set_title('Contour Lbls, #={}'.format(count))
        ax2[3].imshow(gt_c, cmap='gray', aspect='auto')

    if y_m is not None:
        y_m = y_m > threshold_mark
        _, count = label(y_m, return_num=True)
        ax3[0].set_title('Marker Predict, #={}'.format(count))
        ax3[0].imshow(y_m, cmap='gray', aspect='auto')
        _, count = label(gt_m, return_num=True)
        ax3[1].set_title('Marker Lbls, #={}'.format(count))
        ax3[1].imshow(gt_m, cmap='gray', aspect='auto')
        _, count = label(markers, return_num=True)
        ax3[2].set_title('Final Markers, #={}'.format(count))
        ax3[2].imshow(markers, cmap='gray', aspect='auto')

    plt.tight_layout()

    if save:
        dir = predict_save_folder()
        fp = os.path.join(dir, uid + '.png')
        plt.savefig(fp)
    else:
        show_figure()

def predict_save_folder():
    return os.path.join('data', 'predict')

def save_mask(uid, y, y_c, y_m):
    threshold = config['param'].getfloat('threshold')
    segmentation = config['post'].getboolean('segmentation')
    remove_objects = config['post'].getboolean('remove_objects')
    min_object_size = config['post'].getint('min_object_size')

    if segmentation:
        if y_m is not None:
            y, _ = seg_ws_by_marker(y, y_m)
        elif y_c is not None:
            y, _ = seg_ws_by_edge(y, y_c)
        else:
            y, _ = seg_ws(y)
    if remove_objects:
        y = remove_small_objects(y, min_size=min_object_size)

    idxs = np.unique(y) # sorted, 1st is background (e.g. 0)

    dir = os.path.join(predict_save_folder(), uid, 'masks')
    if not os.path.exists(dir):
        os.makedirs(dir)

    for idx in idxs[1:]:
        mask = (y == idx).astype(np.uint8) 
        mask *= 255
        img = Image.fromarray(mask, mode='L')
        img.save(os.path.join(dir, str(uuid.uuid4()) + '.png'), 'PNG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', choices=['train', 'test'], help='dataset to eval')
    parser.add_argument('--csv', dest='csv', action='store_true')
    parser.add_argument('--show', dest='csv', action='store_false')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.set_defaults(csv=False, save=False, mask=False, dataset='test')
    args = parser.parse_args()

    if not args.csv:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError as err:
            print(err)
            print("[ERROR] No GUI library for rendering, consider to save as RLE '--csv'")
            exit(-1)

        if args.save:
            print("[INFO] Save side-by-side prediction figure in 'data/predict' folder...")
            dir = predict_save_folder()
            if not os.path.exists(dir):
                os.makedirs(dir)

    main(args.csv, args.save, args.mask, args.dataset == 'train')
