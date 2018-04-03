import os
import random
import json
import numpy as np
import pandas as pd
import uuid
import argparse
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from helper import config, filter_by_group

def main(dir_src, oversample):
    df = filter_by_group(dir_src, use_filter=True)
    # prepare link target dir
    dir_train = 'data/train'
    dir_valid = 'data/valid'
    if not os.path.isdir(dir_train):
        os.makedirs(dir_train)
    if not os.path.isdir(dir_valid):
        os.makedirs(dir_valid)
    # split cv ratio
    train, valid = train_test_split(
        df,
        test_size=config['dataset'].getfloat('cv_ratio'), 
        random_state=config['dataset'].getint('cv_seed'))
    # recursively make hardlink to origin files
    for uid in train.image_id:
        src = os.path.join(dir_src, uid)
        if oversample:
            # use random uuid to ensure no name conflict
            dst = os.path.join(dir_train, str(uuid.uuid4()))
        else:
            dst = os.path.join(dir_train, uid)
        assert not os.path.exists(dst), "Exist folder name: " + dst+ "\nDo you want to --oversample dataset?"
        shutil.copytree(src, dst, copy_function=os.link)
    for uid in valid.image_id:
        src = os.path.join(dir_src, uid)
        if oversample:
            # use random uuid to ensure no name conflict
            dst = os.path.join(dir_valid, str(uuid.uuid4()))
        else:
            dst = os.path.join(dir_valid, uid)
        assert not os.path.exists(dst), "Exist folder name: " + dst + "\nDo you want to --oversample dataset?"
        shutil.copytree(src, dst, copy_function=os.link)
    print("Number of train after split:", len(train))
    print("Number of valid after split:", len(valid))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_src', type=str, help='dataset filepath')
    parser.add_argument('--oversample', dest='oversample', action='store_true')
    parser.add_argument('--no-oversample', dest='oversample', action='store_false')
    parser.set_defaults(oversample=False)
    args = parser.parse_args()

    main(args.dir_src, args.oversample)
