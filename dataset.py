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

from PIL import Image, ImageOps
from skimage.io import imread
from skimage import filters, img_as_ubyte
from skimage.morphology import remove_small_objects
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.filters import gaussian_filter
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian

# Ignore skimage convertion warnings
import warnings
warnings.filterwarnings("ignore")

from helper import config, pad_image

bright_field_list = [
    '091944f1d2611c916b98c020bd066667e33f4639159b2a92407fe5a40788856d',
    '1a11552569160f0b1ea10bedbd628ce6c14f29edec5092034c2309c556df833e',
    '3594684b9ea0e16196f498815508f8d364d55fea2933a2e782122b6f00375d04',
    '2a1a294e21d76efd0399e4eb321b45f44f7510911acd92c988480195c5b4c812',
    '76a372bfd3fad3ea30cb163b560e52607a8281f5b042484c3a0fc6d0aa5a7450',
    '54793624413c7d0e048173f7aeee85de3277f7e8d47c82e0a854fe43e879cd12',
    '8f94a80b95a881d0efdec36affc915dca9609f4cba8134c4a91b219d418778aa',
    '5e263abff938acba1c0cff698261c7c00c23d7376e3ceacc3d5d4a655216b16d',
    '5d58600efa0c2667ec85595bf456a54e2bd6e6e9a5c0dff42d807bc9fe2b822e',
    '8d05fb18ee0cda107d56735cafa6197a31884e0a5092dc6d41760fb92ae23ab4',
    '1b44d22643830cd4f23c9deadb0bd499fb392fb2cd9526d81547d93077d983df',
    '08275a5b1c2dfcd739e8c4888a5ee2d29f83eccfa75185404ced1dc0866ea992',
    '7f38885521586fc6011bef1314a9fb2aa1e4935bd581b2991e1d963395eab770',
    'c395870ad9f5a3ae651b50efab9b20c3e6b9aea15d4c731eb34c0cf9e3800a72',
    '4217e25defac94ff465157d53f5a24b8a14045b763d8606ec4a97d71d99ee381',
    '4e07a653352b30bb95b60ebc6c57afbc7215716224af731c51ff8d430788cd40'
]

class KaggleDataset(Dataset):
    """Kaggle dataset."""

    def __init__(self, root, transform=None, cache=None, category=None):
        """
        Args:
            root_dir (string): Directory of data (train or test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        if os.path.isfile(root + '.csv'):
            df = pd.read_csv(root + '.csv')
            ok = df['discard'] != 1
            if category is not None:
                # filter only sub-category
                ok &= df['category'] == category
            df = df[ok]
            self.ids = list(df['image_id'])
        else:
            self.ids = next(os.walk(root))[1]
        self.ids.sort()
        self.cache = cache

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        try:
            uid = self.ids[idx]
        except:
            raise IndexError()

        if self.cache is not None and uid in self.cache:
            sample = self.cache[uid]
        else:
            img_name = os.path.join(self.root, uid, 'images', uid + '.png')
            image = Image.open(img_name)
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
                label_c, label_m, _ = get_instances_contour_interior(uid, label_gt)

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
        # get list of dataset index
        n = len(self.ids)
        indices = list(range(n))
        # random shuffle the list
        s = random.getstate()
        random.seed(config['param'].getint('cv_seed'))
        random.shuffle(indices)
        random.setstate(s)
        # return splitted lists
        split = int(np.floor(config['param'].getfloat('cv_ratio') * n))
        return indices[split:], indices[:split]


class Compose():
    def __init__(self, augment=True, padding=False, tensor=True):
        model_name = config['param']['model']
        width = config[model_name].getint('width')
        self.size = (width, width)
        self.weight_bce = config['param'].getboolean('weight_bce')
        self.gcd_depth = config['param'].getint('gcd_depth')

        c = config['pre']
        self.mean = json.loads(c.get('mean'))
        self.std = json.loads(c.get('std'))
        self.label_binary = c.getboolean('label_to_binary')
        self.color_invert = c.getboolean('color_invert')
        self.color_jitter = c.getboolean('color_jitter')
        self.elastic_distortion = c.getboolean('elastic_distortion')
        self.color_equalize = c.getboolean('color_equalize')
        self.tensor = tensor
        self.augment = augment
        self.padding = padding
        self.min_scale = c.getfloat('min_scale')
        self.max_scale = c.getfloat('max_scale')

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

            # replaced with 'thinner' contour based on augmented/transformed mask
            if self.detect_contour:
                label_c, label_m, weight = get_instances_contour_interior(sample['uid'], np.asarray(label_gt))
                label_c, label_m = Image.fromarray(label_c), Image.fromarray(label_m)

            # perform random color invert, assuming 3 channels (rgb) images
            if self.color_invert and random.random() > 0.5:
                image = ImageOps.invert(image)

            # perform ColorJitter()
            if self.color_jitter and random.random() > 0.5:
                color = transforms.ColorJitter.get_params(0.5, 0.5, 0.5, 0.25)
                image = color(image)

        elif self.padding: # add border padding
            w, h = sample['size']
            gcd = self.gcd_depth
            pad_w = pad_h = 0
            if 0 != (w % gcd):
                pad_w = gcd - (w % gcd)
            if 0 != (h % gcd):
                pad_h = gcd - (h % gcd)
            image = pad_image(image, pad_w, pad_h)
            label = ImageOps.expand(label, (0, 0, pad_w, pad_h))
            label_c = ImageOps.expand(label_c, (0, 0, pad_w, pad_h))
            label_m = ImageOps.expand(label_m, (0, 0, pad_w, pad_h))
            label_gt = ImageOps.expand(label_gt, (0, 0, pad_w, pad_h))
            if self.detect_contour:
                label_c, label_m, weight = get_instances_contour_interior(sample['uid'], np.asarray(label_gt))
                label_c, label_m = Image.fromarray(label_c), Image.fromarray(label_m)

        else: # resize down image
            image, label, label_c, label_m = [tx.resize(x, self.size) for x in (image, label, label_c, label_m)]
            if self.precise_contour:
                pil_masks = [tx.resize(m, self.size) for m in pil_masks]
                label_gt = compose_mask(pil_masks, pil=True)
            else:
                label_gt = tx.resize(label_gt, self.size, interpolation=Image.NEAREST)
            if self.detect_contour:
                label_c, label_m, weight = get_instances_contour_interior(sample['uid'], np.asarray(label_gt))
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

        if self.weight_bce and weight is not None:
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

# TODO: the algorithm MUST guarantee (interior + contour = instance mask) & (interior within contour)
def get_contour_interior(uid, mask):
    contour = filters.scharr(mask)
    scharr_threshold = np.amax(abs(contour)) / 2.
    if uid in bright_field_list:
        scharr_threshold = 0. # nuclei are much smaller than others in bright_field slice
    contour = (np.abs(contour) > scharr_threshold).astype(np.uint8)*255
    interior = (mask - contour > 0).astype(np.uint8)*255
    return contour, interior

def get_instances_contour_interior(uid, instances_mask):
    result_c = np.zeros_like(instances_mask, dtype=np.uint8)
    result_i = np.zeros_like(instances_mask, dtype=np.uint8)
    weight = np.ones_like(instances_mask, dtype=np.float32)
    masks = decompose_mask(instances_mask)
    for m in masks:
        contour, interior = get_contour_interior(uid, m)
        result_c = np.maximum(result_c, contour)
        result_i = np.maximum(result_i, interior)
        # magic number 50 make weight distributed to [1, 5) roughly
        weight *= (1 + gaussian_filter(contour, sigma=1) / 50)
    return result_c, result_i, weight

def clahe(img):
    x = np.asarray(img, dtype=np.uint8)
    x = equalize_adapthist(x)
    x = img_as_ubyte(x)
    return Image.fromarray(x)

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
    train = KaggleDataset('data/stage1_train', category='Histology')
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
