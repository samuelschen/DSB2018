import os
import random
import numpy as np
import uuid
import argparse
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image

def do_crop(image, uid, root, folder, step, width, df=None):
    w, h = image.size
    visited = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            x -= max(0, width - min(width, w - x))
            y -= max(0, width - min(width, h - y))
            crop = image.crop((x, y, x+width, y+width))
            assert crop.size == (width, width)
            if np.sum(crop) == 0:
                continue # ignore empty crop
            crop_id = uid + '_{}_{}'.format(x, y)
            if crop_id in visited:
                continue
            else:
                visited.append(crop_id)
            dir = os.path.join(root, crop_id, folder)
            if not os.path.exists(dir):
                os.makedirs(dir)
            if folder == 'masks':
                crop.save(os.path.join(dir, str(uuid.uuid4()) + '.png'), 'PNG')
            else:
                crop.save(os.path.join(dir, crop_id + '.png'), 'PNG')
                if df is not None:
                    row = df[ np.where(df[:, 1] == uid) ][0]
                    row[1] = crop_id
                    df = np.vstack([df, row])
    return df

def main(source, step, width, csvfile=None):
    root = source + '_crop'
    df = None
    if csvfile and os.path.isfile(csvfile):
        df = pd.read_csv(csvfile)
        columns = df.columns
        # datamframe to numpy array
        # DataFrame append() is millions times slower than numpy
        df = df.values
        assert len(df) > 0
    for uid in next(os.walk(source))[1]:
        fn = os.path.join(source, uid, 'images', uid + '.png')
        image = Image.open(fn)
        print("process {} ... ".format(fn))
        df = do_crop(image, uid, root, 'images', step, width, df)
        mask_dir = os.path.join(source, uid, 'masks')
        if os.path.isdir(mask_dir):
            for fn in tqdm(next(os.walk(mask_dir))[2]):
                fn = os.path.join(mask_dir, fn)
                image = Image.open(fn)
                # print("\tprocess {} ... ".format(fn))
                do_crop(image, uid, root, 'masks', step, width)
    # sanity check and remove cropped image without mask
    print("Sanity check and remove no ground truth data ... ")
    for uid in next(os.walk(root))[1]:
        mask_dir = os.path.join(root, uid, 'masks')
        if not os.path.exists(mask_dir):
            shutil.rmtree(os.path.join(root, uid))
            if df is not None:
                df = df[ np.where(df[:, 1] != uid) ]
    if df is not None:
        # convert numpy back to dataframe
        df = pd.DataFrame(df, columns=columns)
        df.to_csv(csvfile, index=False)
    print("Crop task completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str, help='root filepath')
    parser.add_argument('--step', type=int, default=200, help='slice step')
    parser.add_argument('--width', type=int, default=256, help='slice width')
    parser.add_argument('--csv', type=str, help='csv to bookkeep')
    args = parser.parse_args()

    main(args.root, args.step, args.width, args.csv)
