import os
import random
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as tx

from PIL import Image, ImageOps, ImageDraw
from skimage.io import imread
from skimage import filters, img_as_ubyte
from skimage.morphology import remove_small_objects, dilation, erosion
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import center_of_mass

# Ignore skimage convertion warnings
import warnings
warnings.filterwarnings("ignore")

from helper import config, clahe, filter_by_group

class KaggleDataset(Dataset):
    """Kaggle dataset."""

    def __init__(self, root, transform=None, cache=None, use_filter=False):
        """
        Args:
            root_dir (string): Directory of data (train or test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        self.cache = cache
        self.df = filter_by_group(root, use_filter)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            uid = self.df.loc[idx]['image_id']
        except:
            raise IndexError()

        if self.cache is not None and uid in self.cache:
            sample = self.cache[uid]
        else:
            image_dir = os.path.join(self.root, uid, 'images')
            fn = next(os.walk(image_dir))[2][0] # pick first image file
            fp = os.path.join(image_dir, fn)
            image = Image.open(fp)
            # ignore alpha channel if any, because they are constant in all training set
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # overlay masks to single mask
            w, h = image.size
            label_gt = np.zeros((h, w), dtype=np.int32) # instance labels, might > 256
            label = np.zeros((h, w), dtype=np.uint8) # semantic labels
            label_c = np.zeros((h, w), dtype=np.uint8) # contour labels
            label_m = np.zeros((h, w), dtype=np.uint8) # marker labels
            mask_dir = os.path.join(self.root, uid, 'masks')
            if os.path.isdir(mask_dir):
                masks = []
                for fn in next(os.walk(mask_dir))[2]:
                    fp = os.path.join(mask_dir, fn)
                    m = imread(fp)
                    if (m.ndim > 2):
                        m = np.mean(m, -1).astype(np.uint8)
                    if config['pre'].getboolean('fill_holes'):
                        m = binary_fill_holes(m).astype(np.uint8)*255
                    masks.append(m)
                label_gt = compose_mask(masks)
                label = (label_gt > 0).astype(np.uint8)*255 # semantic masks, generated from merged instance mask
                label_c, label_m, _ = get_instances_contour_interior(label_gt)

            label_gt = Image.fromarray(label_gt)
            label = Image.fromarray(label, 'L') # specify it's grayscale 8-bit
            label_c = Image.fromarray(label_c, 'L')
            label_m = Image.fromarray(label_m, 'L')
            sample = {'image': image,
                      'label': label,
                      'label_c': label_c,
                      'label_m': label_m,
                      'label_gt': label_gt,
                      'uid': uid,
                      'size': image.size}
            if config['contour'].getboolean('precise'):
                pil_masks = [Image.fromarray(m) for m in masks]
                sample['pil_masks'] = pil_masks
            if self.cache is not None:
                self.cache[uid] = sample
        if self.transform:
            sample = self.transform(sample)
        return sample

    def split(self):
        ''' return CV split dataset object '''
        from copy import copy
        from sklearn.model_selection import train_test_split
        train, valid = train_test_split(
            self.df,
            test_size=config['dataset'].getfloat('cv_ratio'), 
            random_state=config['dataset'].getint('cv_seed'))
        train_dataset = copy(self)
        valid_dataset = copy(self)
        train_dataset.df = train.reset_index(drop=True)
        valid_dataset.df = valid.reset_index(drop=True)
        return train_dataset, valid_dataset

    def class_weight(self):
        ''' return np.array of class weights, # of dataset '''
        weight = np.asarray(self.df.groupby('group').count()['image_id'])
        # ratio: number of major class divided by average samples of each class after oversample balance
        ratio = np.max(weight) / (np.sum(weight) / len(weight))
        weight = 1. / weight
        weight = np.asarray([weight[t] for t in self.df['group']])
        return weight, ratio

class Compose():
    def __init__(self, augment=True, resize=False, tensor=True):
        self.tensor = tensor
        self.augment = augment
        self.resize = resize

        model_name = config['param']['model']
        width = config[model_name].getint('width')
        self.size = (width, width)
        self.weight_map = config['param'].getboolean('weight_map')

        c = config['pre']
        self.mean = json.loads(c.get('mean'))
        self.std = json.loads(c.get('std'))
        self.label_binary = c.getboolean('label_to_binary')
        self.color_invert = c.getboolean('color_invert')
        self.color_jitter = c.getboolean('color_jitter')
        self.elastic_distortion = c.getboolean('elastic_distortion')
        self.color_equalize = c.getboolean('color_equalize')
        self.min_scale = c.getfloat('min_scale')
        self.max_scale = c.getfloat('max_scale')
        self.add_noise = c.getboolean('add_noise')

        c = config['contour']
        self.detect_contour = c.getboolean('detect')
        self.only_contour = c.getboolean('exclusive')
        self.precise_contour = c.getboolean('precise')

    def __call__(self, sample):
        image, label, label_c, label_m, label_gt = \
                sample['image'], sample['label'], sample['label_c'], sample['label_m'], sample['label_gt']
        if self.precise_contour:
            pil_masks = sample['pil_masks']
        weight = None

        if self.augment:
            if self.color_equalize and random.random() > 0.5:
                image = clahe(image)

            # perform RandomResize() or just enlarge for image size < model input size
            if random.random() > 0.5:
                new_size = int(random.uniform(self.min_scale, self.max_scale) * np.min(image.size))
            else:
                new_size = int(np.min(image.size))
            if new_size < np.max(self.size): # make it viable for cropping
                new_size = int(np.max(self.size))
            image, label, label_c, label_m = [tx.resize(x, new_size) for x in (image, label, label_c, label_m)]
            if self.precise_contour:
                # regenerate all resized masks (bilinear interpolation) and compose them afterwards
                pil_masks = [tx.resize(m, new_size) for m in pil_masks]
                label_gt = compose_mask(pil_masks, pil=True)
            else:
                # label_gt use NEAREST instead of BILINEAR (default) to avoid polluting instance labels after augmentation
                label_gt = tx.resize(label_gt, new_size, interpolation=Image.NEAREST)

            # perform RandomCrop()
            i, j, h, w = transforms.RandomCrop.get_params(image, self.size)
            image, label, label_c, label_m, label_gt = [tx.crop(x, i, j, h, w) for x in (image, label, label_c, label_m, label_gt)]
            if self.precise_contour:
                pil_masks = [tx.crop(m, i, j, h, w) for m in pil_masks]

            # Note: RandomResizedCrop() is popularly used to train the Inception networks, but might not the best choice for segmentation?
            # # perform RandomResizedCrop()
            # i, j, h, w = transforms.RandomResizedCrop.get_params(
            #     image,
            #     scale=(0.5, 1.0)
            #     ratio=(3. / 4., 4. / 3.)
            # )
            # # label_gt use NEAREST instead of BILINEAR (default) to avoid polluting instance labels after augmentation
            # image, label, label_c, label_m = [tx.resized_crop(x, i, j, h, w, self.size) for x in (image, label, label_c, label_m)]
            # label_gt = tx.resized_crop(label_gt, i, j, h, w, self.size, interpolation=Image.NEAREST)

            # perform Elastic Distortion
            if self.elastic_distortion and random.random() > 0.75:
                indices = ElasticDistortion.get_params(image)
                image, label, label_c, label_m = [ElasticDistortion.transform(x, indices) for x in (image, label, label_c, label_m)]
                if self.precise_contour:
                    pil_masks = [ElasticDistortion.transform(m, indices) for m in pil_masks]
                    label_gt = compose_mask(pil_masks, pil=True)
                else:
                    label_gt = ElasticDistortion.transform(label_gt, indices, spline_order=0) # spline_order=0 to avoid polluting instance labels

            # perform RandomHorizontalFlip()
            if random.random() > 0.5:
                image, label, label_c, label_m, label_gt = [tx.hflip(x) for x in (image, label, label_c, label_m, label_gt)]

            # perform RandomVerticalFlip()
            if random.random() > 0.5:
                image, label, label_c, label_m, label_gt = [tx.vflip(x) for x in (image, label, label_c, label_m, label_gt)]

            # perform Random Rotation (0, 90, 180, and 270 degrees)
            random_degree = random.randint(0, 3) * 90
            image, label, label_c, label_m, label_gt = [tx.rotate(x, random_degree) for x in (image, label, label_c, label_m, label_gt)]

            # perform random color invert, assuming 3 channels (rgb) images
            if self.color_invert and random.random() > 0.5:
                image = ImageOps.invert(image)

            # perform ColorJitter()
            if self.color_jitter and random.random() > 0.5:
                color = transforms.ColorJitter.get_params(0.5, 0.5, 0.5, 0.25)
                image = color(image)

            if self.add_noise and random.random() > 0.5:
                image = add_noise(image)

        elif self.resize:  # resize down image
            image, label, label_c, label_m = [tx.resize(x, self.size) for x in (image, label, label_c, label_m)]
            if self.precise_contour:
                pil_masks = [tx.resize(m, self.size) for m in pil_masks]
                label_gt = compose_mask(pil_masks, pil=True)
            else:
                label_gt = tx.resize(label_gt, self.size, interpolation=Image.NEAREST)

        # replaced with 'thinner' contour based on augmented/transformed mask
        if self.detect_contour:
            label_c, label_m, weight = get_instances_contour_interior(np.asarray(label_gt))
            label_c, label_m = Image.fromarray(label_c), Image.fromarray(label_m)

        # Due to resize algorithm may introduce anti-alias edge, aka. non binary value,
        # thereafter map every pixel back to 0 and 255
        if self.label_binary:
            label, label_c, label_m = [x.point(lambda p, threhold=100: 255 if p > threhold else 0)
                                        for x in (label, label_c, label_m)]
            # For train contour only, leverage the merged instances contour label (label_c)
            # the side effect is losing instance count information
            if self.only_contour:
                label_gt = label_c

        # perform ToTensor()
        if self.tensor:
            image, label, label_c, label_m, label_gt = \
                    [tx.to_tensor(x) for x in (image, label, label_c, label_m, label_gt)]
            # perform Normalize()
            image = tx.normalize(image, self.mean, self.std)

        # prepare a shadow copy of composed data to avoid screwup cached data
        x = sample.copy()
        x['image'], x['label'], x['label_c'], x['label_m'], x['label_gt'] = \
                image, label, label_c, label_m, label_gt

        if self.weight_map and weight is not None:
            weight = np.expand_dims(weight, 0)
            x['weight'] = torch.from_numpy(weight)

        if 'pil_masks' in x:
            del x['pil_masks']

        return x

    def denorm(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def pil(self, tensor):
        return tx.to_pil_image(tensor)

    def to_numpy(self, tensor, size=None):
        x = tx.to_pil_image(tensor)
        if size is not None:
            x = x.resize(size)
        return np.asarray(x)

    def show(self, sample):
        image, label, label_c, label_m, label_gt = \
                sample['image'], sample['label'], sample['label_c'], sample['label_m'], sample['label_gt']
        for x in (image, label, label_c, label_m, label_gt):
            if x.dim == 4:  # only dislay first sample
                x = x[0]
            if x.shape[0] > 1: # channel > 1
                x = self.denorm(x)
            x = self.pil(x)
            x.show()

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

def decompose_mask(mask):
    num = mask.max()
    result = []
    for i in range(1, num+1):
        m = mask.copy()
        m[m != i] = 0
        m[m == i] = 255
        result.append(m)
    return result

def get_contour_interior(mask):
    if 'camunet' == config['param']['model']:
        # 2-pixel contour (1out+1in), 2-pixel shrinked interior
        outer = dilation(mask)
        inner = erosion(mask)
        contour = ((outer != inner) > 0).astype(np.uint8)*255
        interior = (erosion(inner) > 0).astype(np.uint8)*255
    else:
        contour = filters.scharr(mask)
        scharr_threshold = np.amax(abs(contour)) / 2.
        contour = (np.abs(contour) > scharr_threshold).astype(np.uint8)*255
        interior = (mask - contour > 0).astype(np.uint8)*255
    return contour, interior

def get_center(mask):
    r = 2
    y, x = center_of_mass(mask)
    center_img = Image.fromarray(np.zeros_like(mask).astype(np.uint8))
    if not np.isnan(x) and not np.isnan(y):
        draw = ImageDraw.Draw(center_img)
        draw.ellipse([x-r, y-r, x+r, y+r], fill='White')
    center = np.asarray(center_img)
    return center

def get_instances_contour_interior(instances_mask):
    result_c = np.zeros_like(instances_mask, dtype=np.uint8)
    result_i = np.zeros_like(instances_mask, dtype=np.uint8)
    weight = np.ones_like(instances_mask, dtype=np.float32)
    masks = decompose_mask(instances_mask)
    for m in masks:
        contour, interior = get_contour_interior(m)
        center = get_center(m)
        result_c = np.maximum(result_c, contour)
        result_i = np.maximum(result_i, interior)
        contour += center
        contour = np.where(contour > 0, 255, 0)
        # magic number 50 make weight distributed to [1, 5) roughly
        weight *= (1 + gaussian_filter(contour, sigma=1) / 50)
    return result_c, result_i, weight

def add_noise(x, mode='gaussian'):
    from skimage.util import random_noise
    is_pil = isinstance(x, Image.Image)
    if is_pil:
        x = np.asarray(x, dtype=np.uint8)
    # input numpy array, and return [0, 1] or [-1, 1] array
    x = random_noise(x, mode=mode)
    if is_pil:
        x = (x * 255).astype(np.uint8)
        x = Image.fromarray(x)
    return x

class ElasticDistortion():
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_params(img, alpha=2000, sigma=30):
        w, h = img.size
        dx = gaussian_filter((np.random.rand(*(h, w)) * 2 - 1),
                            sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*(h, w)) * 2 - 1),
                            sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
        return indices

    @staticmethod
    def transform(img, indices, spline_order=1, mode='nearest'):
        x = np.asarray(img)
        if x.ndim == 2:
            x = np.expand_dims(x, -1)
        shape = x.shape[:2]
        result = np.empty_like(x)
        for i in range(x.shape[2]):
            result[:, :, i] = map_coordinates(
                x[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
        if result.shape[-1] == 1:
            result = np.squeeze(result)
        return Image.fromarray(result, mode=img.mode)

    def __call__(self, img, spline_order=1, mode='nearest'):
        """
        Args:
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Randomly distorted image.
        """
        indices = self.get_params(img)
        return self.transform(img, indices)

if __name__ == '__main__':
    compose = Compose(augment=True)
    train = KaggleDataset('data/train')
    idx = random.randint(0, len(train)-1)
    sample = train[idx]
    print(sample['uid'])
    # display original image
    sample['image'].show()
    sample['label'].show()
    sample['label_c'].show()
    sample['label_m'].show()
    sample['label_gt'].show()
    # display composed image
    sample = compose(sample)
    compose.show(sample)

    if 'weight' in sample:
        w = sample['weight']
        # brighten the pixels
        w = (w.numpy() * 10).astype(np.uint8)
        w = np.squeeze(w)
        w = Image.fromarray(w, 'L')
        w.show()
