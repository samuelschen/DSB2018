import os
import random
import numpy as np
import uuid
import argparse
from tqdm import tqdm
from PIL import Image

def split(image, uid, root, folder, step, width):
    w, h = image.size
    for y in range(0, h, step):
        for x in range(0, w, step):
            w_ = min(width, w - x)
            h_ = min(width, h - y)
            crop = image.crop((x, y, x + w_, y + h_))
            if np.sum(crop) == 0:
                continue # ignore empty crop
            crop_id = uid + '_{}_{}'.format(x, y)
            dir = os.path.join(root, crop_id, folder)
            if not os.path.exists(dir):
                os.makedirs(dir)
            if folder == 'masks':
                crop.save(os.path.join(dir, str(uuid.uuid4()) + '.png'), 'PNG')
            else:
                crop.save(os.path.join(dir, crop_id + '.png'), 'PNG')

def main(source, step, width):
    root = source + '_split'
    for uid in next(os.walk(source))[1]:
        fn = os.path.join(source, uid, 'images', uid + '.png')
        image = Image.open(fn)
        print("process {} ... ".format(fn))
        split(image, uid, root, 'images', step, width)
        mask_dir = os.path.join(source, uid, 'masks')
        if os.path.isdir(mask_dir):
            for fn in tqdm(next(os.walk(mask_dir))[2]):
                fn = os.path.join(mask_dir, fn)
                image = Image.open(fn)
                # print("\tprocess {} ... ".format(fn))
                split(image, uid, root, 'masks', step, width)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root filepath')
    parser.add_argument('--step', type=int, default=200, help='slice step')
    parser.add_argument('--width', type=int, default=256, help='slice width')
    args = parser.parse_args()

    main(args.root, args.step, args.width)
