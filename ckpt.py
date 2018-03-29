import os
import argparse
import torch
from helper import config, load_ckpt, save_ckpt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', nargs='*', help='checkpoint filepath')
    parser.add_argument('--model', help='model name of checkpoint')
    args = parser.parse_args()

    if args.model:
        model_name = args.model.lower()
    else:
        model_name = config['param']['model']
    
    for fn in args.ckpt:
        # load ckpt
        if torch.cuda.is_available():
            # Load all tensors onto previous state
            checkpoint = torch.load(fn)
        else:
            # Load all tensors onto the CPU
            checkpoint = torch.load(fn, map_location=lambda storage, loc: storage)

        if 'name' in checkpoint:
            print("Model name {} has existed in checkpoint".format(checkpoint['name'], fn))
            continue

        checkpoint['name'] = model_name
        torch.save(checkpoint, fn)
        print("Model name {} has insert into checkpoint {}".format(model_name, fn))
