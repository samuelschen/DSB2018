import os
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as tx

from PIL import Image, ImageOps
from skimage.io import imread

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

class Compose():
    def __init__(self, argument=True, tensor=True, binary=True):
        self.size = (config.width, config.width)
        self.mean = config.mean
        self.std = config.std
        self.toTensor = tensor
        self.toBinary = binary
        self.toArgument = argument

    def __call__(self, sample):
        image, label, uid, size = sample['image'], sample['label'], sample['uid'], sample['size']

        if self.toArgument:
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

            # perform random color invert
            if random.random() > 0.5:
                r,g,b,a = image.split()
                rgb_image = Image.merge('RGB', (r,g,b))
                inverted_image = ImageOps.invert(rgb_image)
                r2,g2,b2 = inverted_image.split()
                image = Image.merge('RGBA', (r2,g2,b2,a))

            # perform ColorJitter()
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

