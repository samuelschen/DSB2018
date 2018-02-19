import os
import random
import numpy as np

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
import pandas as pd
# Ignore skimage convertion warnings
import warnings
warnings.filterwarnings("ignore")

import config

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
            #srgb_profile = ImageCms.createProfile("sRGB")
            #lab_profile  = ImageCms.createProfile("LAB")
            #rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
            #image = ImageCms.applyTransform(image, rgb2lab_transform) 
            # overlay masks to single mask
            w, h = image.size
            label = np.zeros((h, w), dtype=np.uint8) # semantic labels
            label_e = np.zeros((h, w), dtype=np.uint8) # edge labels
            label_gt = np.zeros((h, w), dtype=np.uint8) # instance labels
            mask_dir = os.path.join(self.root, uid, 'masks')
            if os.path.isdir(mask_dir):
                instance_idx = 1
                for fn in next(os.walk(mask_dir))[2]:
                    fp = os.path.join(mask_dir, fn)
                    m = imread(fp)
                    if (m.ndim > 2):
                        m = np.mean(m, -1).astype(np.uint8)
                    if config.fill_holes:
                        m = binary_fill_holes(m).astype(np.uint8)*255
                    label = np.maximum(label, m) # merge semantic mask
                    edge = contour(uid, m)
                    label_e = np.maximum(label_e, edge) # merge edge mask
                    m[m > 0] = instance_idx
                    instance_idx += 1
                    label_gt = np.maximum(label_gt, m) # merge instance mask
            # label -= label_e // 2 # soft label edge
            label = Image.fromarray(label, 'L') # specify it's grayscale 8-bit
            # label = label.convert('1') # convert to 1-bit pixels, black and white
            label_gt = Image.fromarray(label_gt, 'L') # specify it's grayscale 8-bit
            # def sigmoid(x):
            #     return 1 / (1 + np.exp(-x))
            # edge_penalty = sigmoid(gaussian(edges, sigma=3)) * 2
            label_e = Image.fromarray(label_e, 'L') # specify it's grayscale 8-bit
            sample = {'image': image, 'label': label, 'label_e': label_e, 'label_gt': label_gt, 'uid': uid, 'size': image.size}
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
        random.seed(config.cv_seed)
        random.shuffle(indices)
        random.setstate(s)
        # return splitted lists
        split = int(np.floor(config.cv_ratio * n))
        return indices[split:], indices[:split]


class NuclearDataset(Dataset):
    """Single nuclei centric dataset."""

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
            cids = list(df['image_id'])
        else:
            cids = next(os.walk(root))[1]
        self.ids = []
        for cid in cids:
            mask_dir = os.path.join(root, cid, 'masks')
            for mask in next(os.walk(mask_dir))[2]:
                self.ids.append(cid + '/' + mask)
        self.ids.sort()
        self.cache = cache

    def __len__(self):
        return len(self.ids)

    def _bbox(self, img_array, margin=config.bbox_margin):
        h, w = img_array.shape
        rows = np.any(img_array, axis=1)
        cols = np.any(img_array, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cmin = max(0, cmin-margin)
        rmin = max(0, rmin-margin)
        cmax = min(w, cmax+margin)
        rmax = min(h, rmax+margin)
        return cmin, rmin, cmax+1, rmax+1

    def _crop_example(self, image, img_id, mask_id):
        mask_name = os.path.join(self.root, img_id, 'masks', mask_id)
        mask = imread(mask_name)
        if (mask.ndim > 2):
            mask = np.mean(mask, -1).astype(np.uint8)
        mask = remove_small_objects(mask, min_size=config.min_object_size)
        if config.fill_holes:
            mask = binary_fill_holes(mask).astype(np.uint8)*255
        left, top, right, bottom = self._bbox(mask)
        def crop(img_array, left, top, right, bottom):
            return img_array[top:bottom, left:right, :] if img_array.ndim > 2 else img_array[top:bottom, left:right]
        crop_img = crop(np.asarray(image), left, top, right, bottom)
        crop_mask = crop(mask, left, top, right, bottom)
        return crop_img, crop_mask

    def __getitem__(self, idx):
        try:
            uid = self.ids[idx]
        except:
            raise IndexError()

        img_id, mask_id = uid.split('/')
        if self.cache is not None and uid in self.cache:
            sample = self.cache[uid]
        elif self.cache is not None and img_id in self.cache:
            image = self.cache[img_id]
            crop_img, crop_mask = self._crop_example(image, img_id, mask_id)
            crop_img = Image.fromarray(crop_img, 'RGB')
            crop_mask = Image.fromarray(crop_mask, 'L')
            crop_edge = contour(uid, crop_mask)
            w, h = crop_img.size
            sample = {'image': crop_img, 'label': crop_mask, 'label_e': crop_edge, 'uid': uid, 'size': image.size}
            sample['label_gt'] = crop_edge if config.train_contour_only else crop_mask
            if self.cache is not None:
                self.cache[uid] = sample
        else:
            img_name = os.path.join(self.root, img_id, 'images', img_id + '.png')
            image = Image.open(img_name)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            crop_img, crop_mask = self._crop_example(image, img_id, mask_id)
            crop_img = Image.fromarray(crop_img, 'RGB')
            crop_mask = Image.fromarray(crop_mask, 'L')
            crop_edge = contour(uid, crop_mask)
            w, h = crop_img.size
            sample = {'image': crop_img, 'label': crop_mask, 'label_e': crop_edge, 'uid': uid, 'size': image.size}
            sample['label_gt'] = crop_edge if config.train_contour_only else crop_mask
            if self.cache is not None:
                self.cache[uid] = sample
                self.cache[img_id] = image
        if self.transform:
            sample = self.transform(sample)
        return sample

    def split(self):
        # get list of dataset index
        n = len(self.ids)
        indices = list(range(n))
        # random shuffle the list
        s = random.getstate()
        random.seed(config.cv_seed)
        random.shuffle(indices)
        random.setstate(s)
        # return splitted lists
        split = int(np.floor(config.cv_ratio * n))
        return indices[split:], indices[:split]


class Compose():
    def __init__(self, augment=True, tensor=True):
        self.size = (config.width, config.width)
        self.mean = config.mean
        self.std = config.std
        self.toBinary = config.label_to_binary
        self.toInvert = config.color_invert
        self.toJitter = config.color_jitter
        self.toDistortion = config.elastic_distortion
        self.toEqualize = config.color_equalize
        self.toTensor = tensor
        self.toAugment = augment
        self.toContour = config.detect_contour

    def __call__(self, sample):
        image, label, label_e, label_gt = sample['image'], sample['label'], sample['label_e'], sample['label_gt']

        if self.toEqualize:
            image = clahe(image)

        if self.toAugment:
            # perform RandomResizedCrop()
            scale = config.resized_crop_cell_scale if config.cell_level else config.resized_crop_slice_scale
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image,
                scale=scale,
                ratio=(3. / 4., 4. / 3.)
            )
            # label_gt should not be used, applied transformation for consistent dimensions
            image, label, label_e, label_gt = [tx.resized_crop(x, i, j, h, w, self.size) for x in (image, label, label_e, label_gt)]

            # perform RandomHorizontalFlip()
            if random.random() > 0.5:
                image, label, label_e, label_gt = [tx.hflip(x) for x in (image, label, label_e, label_gt)]

            # perform RandomVerticalFlip()
            if random.random() > 0.5:
                image, label, label_e, label_gt = [tx.vflip(x) for x in (image, label, label_e, label_gt)]

            # perform Elastic Distortion
            if self.toDistortion:
                indices = ElasticDistortion.get_params(image)
                image, label, label_e, label_gt = [ElasticDistortion.transform(x, indices) for x in (image, label, label_e, label_gt)]

            if self.toContour: # replaced with 'thinner' contour based on augmented/transformed mask
                label_e = contour(sample['uid'], label, config.cell_level)

            # perform random color invert, assuming 3 channels (rgb) images
            if self.toInvert and random.random() > 0.5:
                image = ImageOps.invert(image)

            # perform ColorJitter()
            if self.toJitter:
                color = transforms.ColorJitter.get_params(0.5, 0.5, 0.5, 0.25)
                image = color(image)
        else:
            image, label, label_gt = [tx.resize(x, self.size) for x in (image, label, label_gt)]
            label_e = contour(sample['uid'], label)

        # Due to resize algorithm may introduce anti-alias edge, aka. non binary value,
        # thereafter map every pixel back to 0 and 255
        if self.toBinary:
            label, label_e = [x.point(lambda p, threhold=100: 255 if p > threhold else 0)
                                for x in (label, label_e)]
            if config.train_contour_only:
                label_gt = label_gt.point(lambda p, threhold=100: 255 if p > threhold else 0)

        # perform ToTensor()
        if self.toTensor:
            image, label, label_e, label_gt = [tx.to_tensor(x) for x in (image, label, label_e, label_gt)]

        # perform Normalize()
        if self.toTensor:
            image = tx.normalize(image, self.mean, self.std)

        sample['image'], sample['label'], sample['label_e'], sample['label_gt'] = image, label, label_e, label_gt
        return sample

    def denorm(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def pil(self, tensor):
        return tx.to_pil_image(tensor)

    def show(self, sample):
        image, label, label_e = sample['image'], sample['label'], sample['label_e']
        for x in (image, label, label_e):
            if x.dim == 4:  # only dislay first sample
                x = x[0]
            if x.shape[0] > 1: # channel > 1
                x = self.denorm(x)
            x = self.pil(x)
            x.show()

def contour(uid, mask_img, cell_level=config.cell_level):
    id_list = uid.split('/')
    img_id = id_list[0]
    edge = filters.scharr(mask_img)
    scharr_threshold = np.amax(abs(edge)) / 2.
    if not cell_level and (img_id in bright_field_list):
        scharr_threshold = 0. # nuclei are much smaller than others in bright_field slice
    edge = (np.abs(edge) > scharr_threshold).astype(np.uint8)*255
    return Image.fromarray(edge, 'L')

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
    def get_params(img, alpha=1000, sigma=30):
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
    if config.cell_level:
        train = NuclearDataset('data/stage1_train', category='Histology')
    else:
        train = KaggleDataset('data/stage1_train', category='Histology')
    idx = random.randint(0, len(train)-1)
    sample = train[idx]
    print(sample['uid'])
    # display original image
    sample['image'].show()
    sample['label'].show()
    sample['label_e'].show()
    # display composed image
    sample = compose(sample)
    compose.show(sample)