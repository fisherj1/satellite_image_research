from PIL import Image, ImageEnhance
from itertools import product
import os
import numpy as np
import argparse
import random
import matplotlib.image
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
import torch
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-in', type=str, help='dir in')
    parser.add_argument('--dir-in-contrast', type=str, help='dir in')
    parser.add_argument('--size', type=int, default=240, help='size of images')
    args = parser.parse_args()

    assert args.dir_in is not None

    print(args, '\n')
    return args

if __name__ == "__main__":
    args = parse_args()
    image_size = args.size
    transforms_dict = {'input':transforms.Compose([transforms.Resize(image_size),
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


    for k, i in enumerate(os.listdir(os.path.join(args.dir_in, 'input_hh'))):
      print(k)
      f, axarr = plt.subplots(4, 2)

      img_hh = Image.open(os.path.join(args.dir_in, 'input_hh', i))
      img_hv = Image.open(os.path.join(args.dir_in, 'input_hv', i))
      img_hh_c = Image.open(os.path.join(args.dir_in_contrast, 'input_hh', i))
      img_hv_c = Image.open(os.path.join(args.dir_in_contrast, 'input_hv', i))
      
      a_hh = Image.open(os.path.join(args.dir_in, 'autocors_hh', i))
      a_hv = Image.open(os.path.join(args.dir_in, 'autocors_hv', i))
      a_hh_c = Image.open(os.path.join(args.dir_in_contrast, 'autocors_hh', i))
      a_hv_c = Image.open(os.path.join(args.dir_in_contrast, 'autocors_hv', i))
      
      
      axarr[0,0].imshow(transforms_dict['input'](img_hh).squeeze(), cmap='gray')
      axarr[0,0].set_title("1")
      axarr[0,1].imshow(transforms_dict['input'](img_hv).squeeze(), cmap='gray')
      axarr[0,1].set_title("2")
      axarr[1,0].imshow(transforms_dict['input'](img_hh_c).squeeze(), cmap='gray')
      axarr[1,1].imshow(transforms_dict['input'](img_hv_c).squeeze(), cmap='gray')
      
      axarr[2,0].imshow(transforms_dict['autocor'](a_hh).squeeze(), cmap='gray')
      axarr[2,1].imshow(transforms_dict['autocor'](a_hv).squeeze(), cmap='gray')
      axarr[3,0].imshow(transforms_dict['autocor'](a_hh_c).squeeze(), cmap='gray')
      axarr[3,1].imshow(transforms_dict['autocor'](a_hv_c).squeeze(), cmap='gray')
      
      
      
      plt.show()


     
      
