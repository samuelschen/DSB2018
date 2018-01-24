import os
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as tx

from PIL import Image
from skimage.io import imread

import config

class KaggleDataset(Dataset):
    """Kaggle dataset."""

    def __init__(self, root, transform=None):
        """
        Args:
            root_dir (string): Directory of data (train or test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        self.ids = next(os.walk(root))[1]
        self.cache = {} # only if dataloader do not fork subprocess (num_workers=0)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        if uid in self.cache:
            sample = self.cache[uid]
        else:
            img_name = os.path.join(self.root, uid, 'images', uid + '.png')
            image = Image.open(img_name)
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
            sample = {'image': image, 'label': label}
            self.cache[uid] = sample
        if self.transform:
            sample = self.transform(sample)
        return sample

class Compose():
    def __init__(self, size, mean=config.mean, std=config.std, tensor=True, binary=True):
        self.size = (size, size)
        self.mean = mean
        self.std = std
        self.toTensor = tensor
        self.toBinary = binary

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # perform RandomResizedCrop(), use default parameter
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, 
                                                             scale=(0.08, 1.0), 
                                                             ratio=(3. / 4., 4. / 3.))
        image = tx.resized_crop(image, i, j, h, w, self.size)
        label = tx.resized_crop(label, i, j, h, w, self.size)
        
        # perform RandomHorizontalFlip()
        if random.random() > 0.5:
            image = tx.hflip(image)
            label = tx.hflip(label)

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
        
        return {'image': image, 'label': label}

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
    compose = Compose(128)
    train = KaggleDataset('data/stage1_train')
    idx = random.randint(0, len(train))
    sample = train[idx]
    # display original image
    sample['image'].show()
    sample['label'].show()
    # display composed image
    sample = compose(sample)
    compose.show(sample)

