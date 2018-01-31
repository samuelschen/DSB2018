import os
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as tx

from PIL import Image, ImageOps
from skimage.io import imread
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import config

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
            # several test set files are not 4 channel (RGBA)
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            # overlay masks to single mask
            w, h = image.size
            label = np.zeros((h, w), dtype=np.uint8)
            mask_dir = os.path.join(self.root, uid, 'masks')
            if os.path.isdir(mask_dir):
                for fn in next(os.walk(mask_dir))[2]:
                    fp = os.path.join(mask_dir, fn)
                    m = imread(fp)
                    label = np.maximum(label, m) # merge mask
            label = Image.fromarray(label, 'L') # specify it's grayscale 8-bit
            #label = label.convert('1') # convert to 1-bit pixels, black and white
            sample = {'image': image, 'label': label, 'uid': uid, 'size': image.size}
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
        self.toTensor = tensor
        self.toAugment = augment

    def __call__(self, sample):
        image, label, uid, size = sample['image'], sample['label'], sample['uid'], sample['size']

        if self.toAugment:
            # perform RandomResizedCrop()
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image,
                scale=(0.5, 1.0),
                ratio=(3. / 4., 4. / 3.)
            )
            image = tx.resized_crop(image, i, j, h, w, self.size)
            label = tx.resized_crop(label, i, j, h, w, self.size)

            # perform RandomHorizontalFlip()
            if random.random() > 0.5:
                image = tx.hflip(image)
                label = tx.hflip(label)

            # perform RandomVerticalFlip()
            if random.random() > 0.5:
                image = tx.vflip(image)
                label = tx.vflip(label)

            # perform Elastic Distortion
            if self.toDistortion:
                indices = ElasticDistortion.get_params(image)
                image = ElasticDistortion.transform(image, indices)
                label = ElasticDistortion.transform(label, indices)

            # perform random color invert
            if self.toInvert and random.random() > 0.5:
                r,g,b,a = image.split()
                rgb_image = Image.merge('RGB', (r,g,b))
                inverted_image = ImageOps.invert(rgb_image)
                r2,g2,b2 = inverted_image.split()
                image = Image.merge('RGBA', (r2,g2,b2,a))

            # perform ColorJitter()
            if self.toJitter:
                color = transforms.ColorJitter.get_params(0.5, 0.5, 0.5, 0.25)
                image = color(image)
        else:
            image = tx.resize(image, self.size)
            label = tx.resize(label, self.size)

        # Due to resize algorithm may introduce anti-alias edge, aka. non binary value,
        # thereafter map every pixel back to 0 and 255
        if self.toBinary:
            label = label.point(lambda p, threhold=100: 255 if p > threhold else 0)

        # perform ToTensor()
        if self.toTensor:
            image = tx.to_tensor(image)
            label = tx.to_tensor(label)

        # perform Normalize()
        if self.toTensor:
            image = tx.normalize(image, self.mean, self.std)

        return {'image': image, 'label': label, 'uid': uid, 'size': size}

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
    # display composed image
    sample = compose(sample)
    compose.show(sample)

