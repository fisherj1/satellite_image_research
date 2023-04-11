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
def self_ac(img, kernel):
  h = img.shape[0]-kernel.shape[0]+1
  w = img.shape[1]-kernel.shape[1]+1
  ac = torch.zeros((h, w)).to(device)
  for i in range(h):
    for j in range(w):
      f = img[i:i+kernel.shape[0], j:j+kernel.shape[1]]
      g = kernel[:, :]
      f = f - torch.mean(f)
      g = g - torch.mean(g)
      dot = f*g
      notrm = (torch.linalg.norm(f)*torch.linalg.norm(g)).item()
      ac[i, j] = torch.sum(dot)/notrm
  return ac.detach().cpu().numpy()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-in', type=str, help='dir in')
    parser.add_argument('--dir-out', type=str, help='dir out')
    parser.add_argument('--size', type=int, default=240, help='size of images')

    args = parser.parse_args()

    assert args.dir_in is not None
    assert args.dir_out is not None

    print(args, '\n')
    return args

if __name__ == "__main__":

    args = parse_args()
    image_size = 240
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
      #f, axarr = plt.subplots(4,2)

      img_hh = Image.open(os.path.join(args.dir_in, 'input_hh', i))
      img_hv = Image.open(os.path.join(args.dir_in, 'input_hv', i))
      """
      temp_img3 = np.array(img_hh)
      temp_img4 = np.array(img_hv)
      axarr[0,0].imshow(transforms_dict['input'](img_hh).squeeze(), cmap='gray')
      axarr[0,1].imshow(transforms_dict['input'](img_hv).squeeze(), cmap='gray')
      """


      enhancer1 = ImageEnhance.Contrast(img_hh)
      enhancer2 = ImageEnhance.Contrast(img_hv)
        
      factor = 3 + random.random()*2 # more contrast image
      img_hh = enhancer1.enhance(factor)
      img_hv = enhancer2.enhance(factor)  
      """
      axarr[1,0].imshow(transforms_dict['input'](img_hh).squeeze(), cmap='gray')
      axarr[1,1].imshow(transforms_dict['input'](img_hv).squeeze(), cmap='gray')
      """
      out_input1 = os.path.join(args.dir_out, 'input_hh', i)
      out_input2 = os.path.join(args.dir_out, 'input_hv', i)
      out1 = os.path.join(args.dir_out + 'autocors_hh', i)
      out2 = os.path.join(args.dir_out + 'autocors_hv', i)
      out3 = os.path.join(args.dir_out + 'autocors_hh_hv', i)
      out4 = os.path.join(args.dir_out + 'autocors_hv_hh', i)
      out_classes = os.path.join(args.dir_out, 'classes', i)
      out_seg = os.path.join(args.dir_out, 'target', i)
      
     
      matplotlib.image.imsave(out_input1, img_hh)
      matplotlib.image.imsave(out_input2, img_hv)
        
      device = torch.device('cuda:1')
      
      temp_img = torch.Tensor(np.array(img_hh).astype('float64')).to(device)
      temp_img2 = torch.Tensor(np.array(img_hv).astype('float64')).to(device)

      kernel1 = torch.Tensor(np.array(img_hh)[args.size//4:3*args.size//4, args.size//4:3*args.size//4].astype('float64')).to(device)
      kernel2 = torch.Tensor(np.array(img_hv)[args.size//4:3*args.size//4, args.size//4:3*args.size//4].astype('float64')).to(device)
      
      """
      kernel3 = np.copy(np.array(temp_img3)[args.size//4:3*args.size//4, args.size//4:3*args.size//4])
      kernel4 = np.copy(np.array(temp_img4)[args.size//4:3*args.size//4, args.size//4:3*args.size//4])
      
      axarr[2,0].imshow(self_ac(temp_img3.astype('float128'), kernel3.astype('float128')), cmap='gray')
      axarr[2,1].imshow(self_ac(temp_img4.astype('float128'), kernel4.astype('float128')), cmap='gray')
      axarr[3,0].imshow(self_ac(temp_img.astype('float128'), kernel1.astype('float128')), cmap='gray')
      axarr[3,1].imshow(self_ac(temp_img2.astype('float128'), kernel2.astype('float128')), cmap='gray')
      plt.show()
      
      
      temp_img = temp_img.astype('float64')
      temp_img2 = temp_img2.astype('float64')
      kernel1 = kernel1.astype('float64')
      kernel2 = kernel2.astype('float64')
      b = time.time()
      print(self_ac(temp_img, kernel1), time.time()-b)
      b = time.time()
      print(self_ac1(temp_img, kernel1), time.time()-b)
      """
      
      matplotlib.image.imsave(out1, self_ac(temp_img, kernel1))
      print("lol")
      matplotlib.image.imsave(out2, self_ac(temp_img2, kernel2))
      print("lol")
      matplotlib.image.imsave(out3, self_ac(temp_img, kernel2))
      print("lol")
      matplotlib.image.imsave(out4, self_ac(temp_img2, kernel1))
      print("lol")

      shutil.copy(os.path.join(args.dir_in, 'classes', i+'.npy'), out_classes)
      shutil.copy(os.path.join(args.dir_in, 'target', i), out_seg)
      
