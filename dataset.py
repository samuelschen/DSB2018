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
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.filters import gaussian_filter
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian

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

    def __init__(self, root, transform=None, cache=None):
        """
        Args:
            root_dir (string): Directory of data (train or test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        self.ids = next(os.walk(root))[1]
        self.ids.sort()
        self.cache = cache

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
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
            label = np.zeros((h, w), dtype=np.uint8)
            label_e = np.zeros((h, w), dtype=np.uint8) # edge labels
            mask_dir = os.path.join(self.root, uid, 'masks')
            if os.path.isdir(mask_dir):
                for fn in next(os.walk(mask_dir))[2]:
                    fp = os.path.join(mask_dir, fn)
                    m = imread(fp)
                    if (m.ndim > 2):
                        m = np.mean(m, -1).astype(np.uint8)
                    if config.fill_holes:
                        m = binary_fill_holes(m).astype(np.uint8)*255
                    label = np.maximum(label, m) # merge mask
                    edges = filters.scharr(m) # detect possible edges
                    scharr_threshold = 0. if uid in bright_field_list else (np.amax(abs(edges)) / 2.)
                    edges = (np.abs(edges) > scharr_threshold).astype(np.uint8)*255
                    label_e = np.maximum(label_e, edges)
            # label -= label_e // 2 # soft label edge
            label = Image.fromarray(label, 'L') # specify it's grayscale 8-bit
            # label = label.convert('1') # convert to 1-bit pixels, black and white
            # def sigmoid(x):
            #     return 1 / (1 + np.exp(-x))
            # edge_penalty = sigmoid(gaussian(edges, sigma=3)) * 2
            label_e = Image.fromarray(label_e, 'L') # specify it's grayscale 8-bit
            sample = {'image': image, 'label': label, 'label_e': label_e, 'uid': uid, 'size': image.size}
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

    def __call__(self, sample):
        image, label, label_e = sample['image'], sample['label'], sample['label_e']

        if self.toEqualize:
            image = clahe(image)

        if self.toAugment:
            # perform RandomResizedCrop()
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image,
                scale=(0.5, 1.0),
                ratio=(3. / 4., 4. / 3.)
            )
            image = tx.resized_crop(image, i, j, h, w, self.size)
            label = tx.resized_crop(label, i, j, h, w, self.size)
            label_e = tx.resized_crop(label_e, i, j, h, w, self.size)

            # perform RandomHorizontalFlip()
            if random.random() > 0.5:
                image = tx.hflip(image)
                label = tx.hflip(label)
                label_e = tx.hflip(label_e)

            # perform RandomVerticalFlip()
            if random.random() > 0.5:
                image = tx.vflip(image)
                label = tx.vflip(label)
                label_e = tx.vflip(label_e)


            # perform Elastic Distortion
            if self.toDistortion:
                indices = ElasticDistortion.get_params(image)
                image = ElasticDistortion.transform(image, indices)
                label = ElasticDistortion.transform(label, indices)
                label_e = ElasticDistortion.transform(label_e, indices)

            # perform random color invert, assuming 3 channels (rgb) images
            if self.toInvert and random.random() > 0.5:
                image = ImageOps.invert(image)

            # perform ColorJitter()
            if self.toJitter:
                color = transforms.ColorJitter.get_params(0.5, 0.5, 0.5, 0.25)
                image = color(image)
        else:
            image = tx.resize(image, self.size)
            label = tx.resize(label, self.size)
            label_e = tx.resize(label_e, self.size)

        # Due to resize algorithm may introduce anti-alias edge, aka. non binary value,
        # thereafter map every pixel back to 0 and 255
        if self.toBinary:
            label = label.point(lambda p, threhold=100: 255 if p > threhold else 0)
            label_e = label_e.point(lambda p, threhold=100: 255 if p > threhold else 0)

        # perform ToTensor()
        if self.toTensor:
            image = tx.to_tensor(image)
            label = tx.to_tensor(label)
            label_e = tx.to_tensor(label_e)

        # perform Normalize()
        if self.toTensor:
            image = tx.normalize(image, self.mean, self.std)

        sample['image'], sample['label'], sample['label_e'] = image, label, label_e
        return sample

    def denorm(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def pil(self, tensor):
        return tx.to_pil_image(tensor)

    def show(self, sample):
        image = sample['image']
        if image.dim == 4:
            # only dislay first sample
            image = image[0]
        image = self.denorm(image)
        image = self.pil(image)
        image.show()
        label = sample['label']
        if label.dim == 4:
            # only dislay first sample
            label = label[0]
        label = self.pil(label)
        label.show()
        label_e = sample['label_e']
        if label_e.dim == 4:
            # only dislay first sample
            label_e = label_e[0]
        label_e = self.pil(label_e)
        label_e.show()

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
    compose = Compose()
    train = KaggleDataset('data/stage1_train')
    idx = random.randint(0, len(train))
    sample = train[idx]
    # display original image
    sample['image'].show()
    sample['label'].show()
    sample['label_e'].show()
    # display composed image
    sample = compose(sample)
    compose.show(sample)

