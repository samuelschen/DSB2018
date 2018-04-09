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
from skimage.morphology import label, remove_small_objects
from tqdm import tqdm
from PIL import Image
# own code
from dataset import KaggleDataset, Compose
from helper import config, load_ckpt, prob_to_rles, partition_instances, iou_metric, clahe, filter_fiber

def main(ckpt, tocsv=False, save=False, mask=False, target='test', toiou=False):
    # load one or more checkpoint
    models = []
    for fn in ckpt or [None]:
        # load model
        model = load_ckpt(filepath=fn)
        if not model:
            print("Aborted: checkpoint {} not found!".format(fn))
            return
        # Sets the model in evaluation mode.
        model.eval()
        # put model to GPU
        if torch.cuda.is_available():
            model = model.cuda()
            # model = torch.nn.DataParallel(model).cuda()
        # append to model list
        models.append(model)

    resize = not config['valid'].getboolean('pred_orig_size')
    compose = Compose(augment=False, resize=resize)
    # decide which dataset to pick sample
    data_dir = os.path.join('data', target)
    if target == 'test':
        dataset = KaggleDataset(data_dir, transform=compose)
    elif os.path.exists('data/valid'):
        # advance mode: use valid folder as CV
        dataset = KaggleDataset(data_dir, transform=compose)
    else:
        # auto mode: split part of train dataset as CV
        dataset = KaggleDataset('data/train', transform=compose, use_filter=True)
        if target == 'train':
            dataset, _ = dataset.split()
        elif target == 'valid':
            _, dataset = dataset.split()

    # iterate dataset and inference each sample
    writer = csvfile = None
    for data in tqdm(dataset):
        uid, y, y_c, y_m = inference(data, models, resize)
        x, gt, gt_s, gt_c, gt_m = unpack_data(data, compose, resize)

        if tocsv:
            if writer is None:
                csvfile = open('result.csv', 'w')
                writer = csv.writer(csvfile)
                writer.writerow(['ImageId', 'EncodedPixels'])
            for rle in prob_to_rles(y, y_c, y_m):
                writer.writerow([uid, ' '.join([str(i) for i in rle])])
        elif toiou:
            assert target != 'test'
            if writer is None:
                csvfile = open('iou.csv', 'w')
                writer = csv.writer(csvfile)
                writer.writerow(['ImageId', 'IoU'])
            iou = get_iou(y, y_c, y_m, gt)
            writer.writerow([uid, iou])
        elif mask:
            save_mask(uid, y, y_c, y_m)
        elif target == 'test':
            show(uid, x, y, y_c, y_m, save)
        else: # train or valid
            show_groundtruth(uid, x, y, y_c, y_m, gt, gt_s, gt_c, gt_m, save)

    # end of for-loop
    if csvfile is not None:
        csvfile.close()
# end of main()

def unpack_data(data, compose, resize):
    x = data['image']
    size = data['size']
    gt_s = data['label']
    gt_c = data['label_c']
    gt_m = data['label_m']
    gt = data['label_gt']
    # convert input to numpy array
    x = compose.denorm(x)
    s = size if resize else None
    x = compose.to_numpy(x, s)
    gt = compose.to_numpy(gt, s)
    gt_s = compose.to_numpy(gt_s, s)
    gt_c = compose.to_numpy(gt_c, s)
    gt_m = compose.to_numpy(gt_m, s)
    return x, gt, gt_s, gt_c, gt_m

def inference(data, models, resize):
    threshold = config['param'].getfloat('threshold')
    threshold_edge = config['param'].getfloat('threshold_edge')
    threshold_mark = config['param'].getfloat('threshold_mark')
    ensemble_policy = config['post']['ensemble']

    # sub-rountine to convert output tensor to numpy
    def convert(t):
        assert isinstance(t, (torch.FloatTensor, torch.cuda.FloatTensor))
        if len(t) == 0:
            return None
        # pixel wise ensemble output of models
        t = torch.mean(t, 0, True)
        # to numpy array
        t = t.cpu()
        t = t.numpy()[0]
        if ensemble_policy == 'vote':
            t = np.where(t >= 0.5, 1., 0.) # majority vote
        # channel first [C, H, W] -> channel last [H, W, C]
        t = np.transpose(t, (1, 2, 0))
        # Remove single-dimensional channel from the shape of an array
        t = np.squeeze(t)
        t = align_size(t, size, resize)
        return t

    # get input data
    uid = data['uid']
    size = data['size']
    inputs = data['image']
    # prepare input variables
    inputs = inputs.unsqueeze(0)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    if not resize:
        inputs = pad_tensor(inputs, size)
    else:
        inputs = Variable(inputs)

    y_s = y_c = y_m = torch.FloatTensor()
    if torch.cuda.is_available():
        y_s, y_c, y_m = y_s.cuda(), y_c.cuda(), y_m.cuda()
    for model in models:
        model_name = type(model).__name__.lower()
        with_contour = config.getboolean(model_name, 'branch_contour', fallback=False)
        with_marker = config.getboolean(model_name, 'branch_marker', fallback=False)
        # predict model output
        c = m = Variable()
        if torch.cuda.is_available():
            c, m = c.cuda(), m.cuda()
        s = model(inputs)
        # convert predict to numpy array
        if with_contour and with_marker:
            s, c, m = s
        elif with_contour:
            s, c = s
        # concat outputs
        if ensemble_policy == 'avg':
            y_s = torch.cat([y_s, s.data], 0)
            y_c = torch.cat([y_c, c.data], 0)
            y_m = torch.cat([y_m, m.data], 0)
        elif ensemble_policy == 'vote':
            y_s = torch.cat([y_s, (s.data > threshold).float()], 0)
            y_c = torch.cat([y_c, (c.data > threshold_edge).float()], 0)
            y_m = torch.cat([y_m, (m.data > threshold_mark).float()], 0)
        else:
            raise NotImplementedError("Ensemble policy not implemented")
    return uid, convert(y_s), convert(y_c), convert(y_m)
# end of predict()

def pad_tensor(img_tensor, size, mode='reflect'):
    # get proper mini-width required for model input
    # for example, 32 for 5 layers of max_pool
    gcd = config['param'].getint('gcd_depth')
    # estimate border padding margin
    # (paddingLeft, paddingRight, paddingTop, paddingBottom)
    pad_w = pad_h = 0
    w, h = size
    if 0 != (w % gcd):
        pad_w = gcd - (w % gcd)
    if 0 != (h % gcd):
        pad_h = gcd - (h % gcd)
    pad = (0, pad_w, 0, pad_h)
    # decide padding mode
    if mode == 'replica':
        f = nn.ReplicationPad2d(pad)
    elif mode == 'constant':
        # padding color should honor each image background, default is black (0)
        bgcolor = 0 if np.median(img_tensor) < 100 else 255
        f = nn.ConstantPad2d(pad, bgcolor)
    elif mode == 'reflect':
        f = nn.ReflectionPad2d(pad)
    else:
        raise NotImplementedError()
    return f(img_tensor)

def align_size(img_array, size, regrowth=True):
    from skimage.transform import resize
    if img_array is None:
        return img_array
    elif regrowth:
        return resize(img_array, size[::-1], mode='constant', preserve_range=True)
    else:
        w, h = size
        # crop padding
        return img_array[:h, :w]

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
    threshold_edge = config['param'].getfloat('threshold_edge')
    threshold_mark = config['param'].getfloat('threshold_mark')
    segmentation = config['post'].getboolean('segmentation')
    remove_objects = config['post'].getboolean('remove_objects')
    min_object_size = config['post'].getint('min_object_size')
    remove_fiber = config['post'].getboolean('filter_fiber')
    view_color_equalize = config['valid'].getboolean('view_color_equalize')

    fig, (ax1, ax2) = plt.subplots(2, 3, sharey=True, figsize=(10, 8))
    fig.suptitle(uid, y=1)
    ax1[1].set_title('Final Pred, P > {}'.format(threshold))
    ax1[2].set_title('Overlay, P > {}'.format(threshold))
    y_bw = y > threshold

    if view_color_equalize:
        x = clahe(x)
    ax1[0].set_title('Image')
    ax1[0].imshow(x, aspect='auto')
    if segmentation:
        y, markers = partition_instances(y, y_m, y_c)
    if remove_objects:
        y = remove_small_objects(y, min_size=min_object_size)
    if remove_fiber:
        y = filter_fiber(y)
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
    threshold_edge = config['param'].getfloat('threshold_edge')
    threshold_mark = config['param'].getfloat('threshold_mark')
    segmentation = config['post'].getboolean('segmentation')
    remove_objects = config['post'].getboolean('remove_objects')
    remove_fiber = config['post'].getboolean('filter_fiber')
    min_object_size = config['post'].getint('min_object_size')
    only_contour = config['contour'].getboolean('exclusive')
    view_color_equalize = config['valid'].getboolean('view_color_equalize')
    print_table = config['valid'].getboolean('print_table')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 4, sharey=True, figsize=(12, 8))
    fig.suptitle(uid, y=1)

    y_s = y # to show pure semantic predict later

    if view_color_equalize:
        x = clahe(x)
    ax1[0].set_title('Image')
    ax1[0].imshow(x, aspect='auto')
    if segmentation :
        y, markers = partition_instances(y, y_m, y_c)
    if remove_objects:
        y = remove_small_objects(y, min_size=min_object_size)
    if remove_fiber:
        y = filter_fiber(y)
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
        iou = iou_metric(y, label(gt > 0), print_table)
    else:
        iou = iou_metric(y, gt, print_table)
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

    _, count = label(markers, return_num=True)
    ax3[0].set_title('Final Markers, #={}'.format(count))
    ax3[0].imshow(markers, cmap='gray', aspect='auto')
    if y_m is not None:
        y_m = y_m > threshold_mark
        _, count = label(y_m, return_num=True)
        ax3[1].set_title('Marker Predict, #={}'.format(count))
        ax3[1].imshow(y_m, cmap='gray', aspect='auto')
        _, count = label(gt_m, return_num=True)
        ax3[2].set_title('Marker Lbls, #={}'.format(count))
        ax3[2].imshow(gt_m, cmap='gray', aspect='auto')

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
    remove_fiber = config['post'].getboolean('filter_fiber')

    if segmentation:
        y, _ = partition_instances(y, y_m, y_c)
    if remove_objects:
        y = remove_small_objects(y, min_size=min_object_size)
    if remove_fiber:
        y = filter_fiber(y)
    idxs = np.unique(y) # sorted, 1st is background (e.g. 0)

    dir = os.path.join(predict_save_folder(), uid, 'masks')
    if not os.path.exists(dir):
        os.makedirs(dir)

    for idx in idxs[1:]:
        mask = (y == idx).astype(np.uint8)
        mask *= 255
        img = Image.fromarray(mask, mode='L')
        img.save(os.path.join(dir, str(uuid.uuid4()) + '.png'), 'PNG')


def get_iou(y, y_c, y_m, gt):
    segmentation = config['post'].getboolean('segmentation')
    remove_objects = config['post'].getboolean('remove_objects')
    min_object_size = config['post'].getint('min_object_size')
    remove_fiber = config['post'].getboolean('filter_fiber')
    only_contour = config['contour'].getboolean('exclusive')

    if segmentation:
        y, markers = partition_instances(y, y_m, y_c)
    if remove_objects:
        y = remove_small_objects(y, min_size=min_object_size)
    if remove_fiber:
        y = filter_fiber(y)
    if only_contour:
        iou = iou_metric(y, label(gt > 0))
    else:
        iou = iou_metric(y, gt)
    return iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store', choices=['train', 'valid', 'test'], help='Specify dataset to evaluate')
    parser.add_argument('--csv', dest='csv', action='store_true', help='Generate running length encoding')
    parser.add_argument('--show', dest='csv', action='store_false', help='Render overlay prediction on screen')
    parser.add_argument('--save', action='store_true', help='Save overlay prediction as PNG files')
    parser.add_argument('--mask', action='store_true', help='Save prediction as PNG files per nuclei')
    parser.add_argument('--iou', action='store_true', help='Generate IoU CSV report')
    parser.add_argument('ckpt', nargs='*', help='filepath of checkpoint(s), otherwise lookup checkpoint/current.json')
    parser.set_defaults(csv=False, save=False, mask=False, dataset='test', iou=False)
    args = parser.parse_args()

    if not args.csv and not args.iou:
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

    main(args.ckpt, args.csv, args.save, args.mask, args.dataset, args.iou)
