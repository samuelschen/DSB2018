import os
import uuid
import numpy as np

import argparse
from PIL import Image
from skimage.io import imread, imsave
from scipy.ndimage.morphology import binary_fill_holes

# Ignore skimage convertion warnings
import warnings
warnings.filterwarnings("ignore")

def stitch_pathes(input_dir, output_dir, min_width, min_height):
    ids = next(os.walk(input_dir))[1]
    ids.sort()
    for id in ids:
        img_path = os.path.join(input_dir, id, 'images', id + '.png')
        mask_dir = os.path.join(input_dir, id, 'masks')
        new_img, new_masks = stitch_patch(img_path, mask_dir, min_width, min_height)

        img_dir = os.path.join(output_dir, id, 'images')
        os.makedirs(img_dir)
        imsave(os.path.join(img_dir, id + '.png'), new_img)
        mask_dir = os.path.join(output_dir, id, 'masks')
        os.makedirs(mask_dir)
        idxs = np.unique(new_masks) # sorted, 1st is background (e.g. 0)
        for idx in idxs[1:]:
            mask = (new_masks == idx).astype(np.uint8)
            mask *= 255
            img = Image.fromarray(mask, mode='L')
            img.save(os.path.join(mask_dir, str(uuid.uuid4()) + '.png'), 'PNG')

def stitch_patch(image_path, mask_dir, min_width, min_height):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # overlay masks to single mask
    w, h = image.size
    label_gt = np.zeros((h, w), dtype=np.int32) # instance labels, might > 256
    masks = []
    for fn in next(os.walk(mask_dir))[2]:
        fp = os.path.join(mask_dir, fn)
        m = imread(fp)
        m = binary_fill_holes(m).astype(np.uint8)*255
        masks.append(m)
    label_gt = compose_mask(masks)

    image = np.asarray(image)
    print('before stitch size: ', image.shape, ' instance #: ', label_gt.max())
    while w < min_width:
        image = np.concatenate((image, np.fliplr(image)), axis=1)
        label_gt_ex = np.where(label_gt>0, label_gt + label_gt.max(), 0)
        label_gt = np.concatenate((label_gt, np.fliplr(label_gt_ex)), axis=1)
        w = w*2
    while h < min_height:
        image = np.concatenate((image, np.flipud(image)), axis=0)
        label_gt_ex = np.where(label_gt>0, label_gt + label_gt.max(), 0)
        label_gt = np.concatenate((label_gt, np.flipud(label_gt_ex)), axis=0)
        h = h*2
    print('after stitch size: ', image.shape, ' instance #: ', label_gt.max())
    return image, label_gt


def compose_mask(masks, pil=False):
    result = np.zeros_like(masks[0], dtype=np.int32)
    for i, m in enumerate(masks):
        mask = np.array(m) if pil else m.copy()
        mask = mask.astype(np.int32)
        mask[mask > 0] = i + 1 # zero for background, starting from 1
        result = np.maximum(result, mask) # overlay mask one by one via np.maximum, to handle overlapped labels if any
    if pil:
        result = Image.fromarray(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='source image patch dir')
    parser.add_argument('output_dir', type=str, help='target image patch dir')
    parser.add_argument('--min_width', type=int, help='min width of stitched image')
    parser.add_argument('--min_height', type=int, help='min height of stitched image')
    parser.set_defaults(min_width=256, min_height=256)
    args = parser.parse_args()

    stitch_pathes(args.input_dir, args.output_dir, args.min_width, args.min_height)