from pathlib import Path
from math import sqrt
import glob
import os
from torchvision.datasets import ImageFolder as IF
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

class NormToMax:
    def __init__(self, m):
        self.m = m

    def __call__(self, x):
        return x/self.m

class DataLoaderSegmentation(Dataset):
    def __init__(self, data_path, transform, target_transform=None, loader=None):
        super(DataLoaderSegmentation, self).__init__()

        self.input_files = glob.glob(os.path.join(data_path,'input','*.tiff'))
        self.target_files = []

        for img_name in self.input_files:
             self.target_files.append(os.path.join(data_path,'target', os.path.basename(img_name)))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.colors = [[  0,   0,   0,   0], 
                       [  0,  34, 223, 255],
                       [  0, 100, 255, 255],
                       [  0, 250,   0, 255],
                       [171, 243, 255, 255]]

    def __getitem__(self, index):
            img_path = self.input_files[index]
            target_path = self.target_files[index]

            input_img = self.loader(img_path)
            target_img = self.loader(target_path)
            if self.transform is not None:
                input_img = self.transform(input_img)
            if self.target_transform is not None:
                temp = self.get_mask(self.colors[0], target_img)[..., np.newaxis]
                for color in self.colors[1:]:
                    temp = np.concatenate((self.get_mask(color, target_img)[..., np.newaxis], temp))             

                target_img = self.target_transform(temp)    

            return input_img, target_img

    def __len__(self):
        return len(self.input_files)

    def get_mask(color, img):
        return (img[..., 0] == color[0]) & (img[..., 1] == color[1]) & (img[..., 2] == color[2]) & (img[..., 3] == color[3])


class DataLoaderSegmentation(Dataset):
    def __init__(self, data_path, transforms, loader=None):
        super(DataLoaderSegmentation, self).__init__()

        self.input_hh_files = glob.glob(os.path.join(data_path,'input_hh','*.tiff'))
        print("Files: ", len(self.input_hh_files))
        self.input_hv_files = []
        self.target_files = []
        self.autocor_hh_files = []
        self.autocor_hv_files = []
        self.autocor_hh_hv_files = []
        self.autocor_hv_hh_files = []
        self.classes_files = []

        for img_name in self.input_hh_files:
            self.input_hv_files.append(os.path.join(data_path,'input_hv', os.path.basename(img_name)))
        for img_name in self.input_hh_files:
            self.target_files.append(os.path.join(data_path,'target', os.path.basename(img_name)))

        for img_name in self.input_hh_files:
            self.autocor_hh_files.append(os.path.join(data_path,'autocors_hh', os.path.basename(img_name)))
        for img_name in self.input_hh_files:
            self.autocor_hv_files.append(os.path.join(data_path,'autocors_hv', os.path.basename(img_name)))
        for img_name in self.input_hh_files:
            self.autocor_hh_hv_files.append(os.path.join(data_path,'autocors_hh_hv', os.path.basename(img_name)))
        for img_name in self.input_hh_files:
            self.autocor_hv_hh_files.append(os.path.join(data_path,'autocors_hv_hh', os.path.basename(img_name)))
        for img_name in self.input_hh_files:
            self.classes_files.append(os.path.join(data_path,'classes', os.path.basename(img_name)))#+'.npy'))
        

        self.transforms = transforms
        self.loader = loader
        self.colors = [[  0,   0,   0,   0], 
                       [  0,  34, 223, 255],
                       [  0, 100, 255, 255],
                       [  0, 250,   0, 255],
                       [171, 243, 255, 255]]

    def __getitem__(self, index):
            input_hh_path = self.input_hh_files[index]
            input_hv_path = self.input_hv_files[index]
            autocor_hh_path = self.autocor_hh_files[index]
            autocor_hv_path = self.autocor_hv_files[index]
            autocor_hh_hv_path = self.autocor_hh_hv_files[index]
            autocor_hv_hh_path = self.autocor_hv_hh_files[index]
            classes_path = self.classes_files[index]
            target_path = self.target_files[index]

            input_hh = self.loader(input_hh_path)

            input_hv = self.loader(input_hv_path)
            autocor_hh = self.loader(autocor_hh_path)
            autocor_hv = self.loader(autocor_hv_path)
            autocor_hh_hv = self.loader(autocor_hh_hv_path)
            autocor_hv_hh = self.loader(autocor_hv_hh_path)
            target = self.loader(target_path)
            classes = np.load(classes_path)


            input_hh = self.transforms['input'](input_hh)
            input_hv = self.transforms['input'](input_hv)
            autocor_hh = self.transforms['autocor'](autocor_hh)
            autocor_hv = self.transforms['autocor'](autocor_hv)
            autocor_hh_hv = self.transforms['autocor'](autocor_hh_hv)
            autocor_hv_hh = self.transforms['autocor'](autocor_hv_hh)
            classes = self.transforms['classes'](classes)
            target = self.transforms['target'](target)    
            return (input_hh, input_hv, autocor_hh, autocor_hv, autocor_hh_hv, autocor_hv_hh, target, classes)

    def __len__(self):
        return len(self.input_hh_files)


def image_loader(path):
    img = Image.open(path)
    return img

def get_dataloader(cfg):
    train_dir = Path(cfg.data) / 'train'
    valid_dir = Path(cfg.data) / 'test'

    image_size = cfg.image_size

    transform_dict = {'input':transforms.Compose([transforms.Resize(image_size),
                     transforms.CenterCrop(image_size),
                     transforms.Grayscale(1),
                     transforms.ToTensor()]),

                     'autocor': transforms.Compose([transforms.Resize(image_size//2+1),
                     transforms.CenterCrop(image_size//2+1),
                     transforms.Grayscale(1),
                     transforms.ToTensor()]),
                     'target': transforms.Compose([transforms.ToTensor()]),
                     'classes':transforms.Compose([transforms.ToTensor()]),
                     }


    train_dataset = DataLoaderSegmentation(data_path=train_dir, transforms=transform_dict, loader=image_loader)
    
    valid_dataset = DataLoaderSegmentation(data_path=valid_dir, transforms=transform_dict, loader=image_loader)
                     
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_bs, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.train_bs, shuffle=False)

    dataloaders = {'train': train_dataloader,
                    'valid': valid_dataloader, }

    return dataloaders


def get_test_dataloader(cfg):
    dir = Path(cfg.data)

    image_size = cfg.image_size

    transform_dict = {'input':transforms.Compose([transforms.Resize(image_size),
                     transforms.CenterCrop(image_size),
                     transforms.Grayscale(1),
                     transforms.ToTensor()]),

                     'autocor': transforms.Compose([transforms.Resize(image_size//2+1),
                     transforms.CenterCrop(image_size//2+1),
                     transforms.Grayscale(1),
                     transforms.ToTensor()]),
                     'target': transforms.Compose([transforms.ToTensor()]),
                     'classes':transforms.Compose([transforms.ToTensor()]),
                     }

    test_dataset = DataLoaderSegmentation(data_path=dir, transforms=transform_dict, loader=image_loader)
    return test_dataset
