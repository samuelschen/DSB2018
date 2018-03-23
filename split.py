import os
import random
import json
import numpy as np
import pandas as pd
import uuid
import argparse
import shutil
from tqdm import tqdm
from helper import config

def csv_list(root):
    c = config['dataset']
    csv = c.get('csv_file')
    df = pd.read_csv(csv)
    assert len(df) > 0
    # filter by existing file first
    files = next(os.walk(root))[1]
    ok = df['image_id'].isin(files)
    print("Number of existed file in csv file:", sum(ok))
    # filter by source
    src = c.get('source')
    if src is not None:
        src = [e.strip() for e in src.split(',')]
        # filter only sub-category
        ok &= df['source'].isin(src)
    # filter by major-category
    cat = c.get('major_category')
    if cat is not None:
        cat = [e.strip() for e in cat.split(',')]
        # filter only sub-category
        ok &= df['major_category'].isin(cat)
    # filter by sub-category
    cat = c.get('sub_category')
    if cat is not None:
        cat = [e.strip() for e in cat.split(',')]
        # filter only sub-category
        ok &= df['sub_category'].isin(cat)
    # final list of valid training data
    print("Number of remaining file in csv file:", sum(ok))
    df = df[ok]
    return list(df['image_id'])

def cv_split(n):
    indices = list(range(n))
    # random shuffle the list
    s = random.getstate()
    random.seed(config['dataset'].getint('cv_seed'))
    random.shuffle(indices)
    random.setstate(s)
    # return splitted lists
    split = int(np.floor(config['dataset'].getfloat('cv_ratio') * n))
    return indices[split:], indices[:split]

def main(dir_src):
    csv = config['dataset'].get('csv_file')
    if os.path.isfile(csv):
        files = csv_list(dir_src)
    else:
        files = next(os.walk(dir_src))[1]
    files.sort()
    print("Number of valid files:", len(files))
    # prepare link target dir
    dir_train = 'data/train'
    dir_valid = 'data/valid'
    if not os.path.isdir(dir_train):
        os.makedirs(dir_train)
    if not os.path.isdir(dir_valid):
        os.makedirs(dir_valid)
    # split cv ratio
    idx_train, idx_valid = cv_split(len(files))
    # recursively make hardlink to origin files
    for i in idx_train:
        src = os.path.join(dir_src, files[i])
        dst = os.path.join(dir_train, files[i])
        shutil.copytree(src, dst, copy_function=os.link)
    for i in idx_valid:
        src = os.path.join(dir_src, files[i])
        dst = os.path.join(dir_valid, files[i])
        shutil.copytree(src, dst, copy_function=os.link)
    print("Number of train after split:", len(idx_train))
    print("Number of valid after split:", len(idx_valid))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_src', type=str, help='dataset filepath')
    args = parser.parse_args()

    main(args.dir_src)
