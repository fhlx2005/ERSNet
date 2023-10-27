import torch
import torch.utils.data as data
import torch
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision import transforms as tfs
from torchvision.transforms import AutoAugmentPolicy, AutoAugment, InterpolationMode, RandAugment

from euroSAT.GridMask.imagenet_grid.utils.grid import GridMask

subdirNames =['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

class RemoteData(data.Dataset):
    def __init__(self, filepath, transform=None):
        super(RemoteData, self).__init__()
        self.imgs, self.labels = self.getImgPath(filepath)
        self.transform = transform


    def getImgPath(self, parentPath):
        imgpaths = []
        lables =[]
        for subdirName in subdirNames:
            for imgpath in os.listdir(os.path.join(parentPath, subdirName)):
                imgpaths.append(os.path.join(parentPath, subdirName, imgpath))
                lables.append(subdirNames.index(subdirName))
        return imgpaths, lables

    def __getitem__(self, index: int):
        img = Image.open(self.imgs[index]).convert('RGB')
        label = self.labels[index]

        im_aug = tfs.Compose([
            tfs.RandomHorizontalFlip(),
            AutoAugment(policy=AutoAugmentPolicy.SVHN, interpolation=InterpolationMode.NEAREST),

            # RandAugment(),
            tfs.ToTensor(),
            tfs.Normalize(mean=mean, std=std),
            tfs.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

        ])

        im_aug_ = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=mean, std=std)
        ])
        if self.transform:
            img = im_aug(img)
        if not self.transform :
            img = im_aug_(img)
        return img.float(), label

    def __len__(self):
        return len(self.imgs)