import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import logging 
from kd.data.utils import HorizontalTranslate, VerticalTranslate, IMG_LEVELS, IMG_HORIZONTAL_SHIFT, IMG_VERTICAL_SHIFT, IMG_ROTATION, IMG_BRIGHTNESS

SVHN_MEAN, SVHN_STD = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)

def get_svhn_options():
    return "classification", (3, 32, 32), 10, [IMG_LEVELS, IMG_LEVELS, IMG_LEVELS, IMG_LEVELS], ["horizontal_shift", "vertical_shift", "rotation", "brightness"]

def get_svhn_train_valid_test_loaders(batch_size=128, random = False, level=0, augmentation=""):
    train_data = datasets.SVHN(root="~/.torch/", split="train", 
                                    download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(SVHN_MEAN, SVHN_STD)]))

    test_transform = [transforms.ToTensor()]
    if random:
        if augmentation == "horizontal_shift":
            test_transform.append(HorizontalTranslate(IMG_HORIZONTAL_SHIFT[level], (32, 32)))

        elif augmentation == "vertical_shift":
            test_transform.append(VerticalTranslate(IMG_VERTICAL_SHIFT[level], (32, 32)))
        
        elif augmentation == "rotation":
            test_transform.append(transforms.RandomAffine(degrees=(IMG_ROTATION[level],IMG_ROTATION[level]), translate=None, scale=None, shear = None, fill=0))

        elif augmentation == "brightness":
            test_transform.append(transforms.ColorJitter(brightness=(IMG_BRIGHTNESS[level], IMG_BRIGHTNESS[level])))
        else:
            raise ValueError("Unknown augmentation: {}".format(augmentation))
    test_transform.append(transforms.Normalize(SVHN_MEAN, SVHN_STD))

    test_data = datasets.SVHN(root="~/.torch/", split="test",
                                    download=True, transform=transforms.Compose(test_transform))

    valid_portion = 0.1
    indices = torch.randperm(len(train_data))
    valid_split = int(len(indices) * valid_portion)
    
    valid_idx, train_idx = indices[:valid_split], indices[valid_split:]
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    logging.info("### SVHN: Train size: %d, Validation size: %d, Test size: %d ###" % (len(train_idx), len(valid_idx), len(test_loader.dataset)))

    return train_loader, valid_loader, test_loader