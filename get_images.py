from PIL import Image
from itertools import product
import os
import numpy as np
import argparse
import random
import matplotlib.image
import matplotlib.pyplot as plt
def self_ac(img, kernel):
  h = img.shape[0]-kernel.shape[0]+1
  w = img.shape[1]-kernel.shape[1]+1
  ac = np.zeros((h, w))
  #print(img.shape, kernel.shape, ac.shape)
  for i in range(h):
    for j in range(w):
      f = np.copy(img[i:i+kernel.shape[0], j:j+kernel.shape[1]])
      #print(i, j, f.shape)
      g = np.copy(kernel)
      f = f - np.mean(f)
      g = g - np.mean(g)
      #print(type(np.linalg.norm(f)*np.linalg.norm(g).item()), type(np.dot(f, g)))
      dot = f*g
      #print(dot)
      #print(f, g, dot)
      notrm = (np.linalg.norm(f)*np.linalg.norm(g)).item()
     #print(dot/notrm)
      ac[i, j] = np.sum(dot)/notrm
      #print(ac[i,j])
  return ac

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random', action='store_true', help='random crop')
    parser.add_argument('--n', type=int, default=2, help='n of crop images')
    parser.add_argument('--dir-in', type=str, help='dir in')
    parser.add_argument('--dir-out', type=str, help='dir out')
    parser.add_argument('--size', type=int, default=240, help='crop size')
    parser.add_argument('--persent', action='store_true', help='persent of types of ice')

    args = parser.parse_args()

    assert args.dir_in is not None
    assert args.dir_out is not None

    print(args, '\n')
    return args

if __name__ == "__main__":
    args = parse_args()
    name, ext = os.path.splitext('s1a-ew-grd-hv-20200709t030543-20200709t030619-033372-03ddcb-001.tiff')
    name_seg, ext_seg = os.path.splitext('result.tif')

    img = Image.open(os.path.join(args.dir_in, 's1a-ew-grd-hh-20200709t030543-20200709t030619-033372-03ddcb-001.tiff'))

    img2 = Image.open(os.path.join(args.dir_in, 's1a-ew-grd-hv-20200709t030543-20200709t030619-033372-03ddcb-002.tiff'))
   
    segment = Image.open(os.path.join(args.dir_in, 'result.tif'))
    colors = [[  0,   0,   0,   0], 
                       [  0,  34, 223, 255],
                       [  0, 100, 255, 255],
                       [  0, 250,   0, 255],
                       [171, 243, 255, 255]]
    d = {0:None, 512:0, 610:1, 505:3, 924: 2}
    # 0 - Темной синий, 1 - Синий, 2 - Голубой 3 - Салатовый
    w, h = img.size
    assert (w > args.size) and (h > args.size)
    if args.random:
        if 'test' in args.dir_out:
            print('test')
        k = 0
        while k != args.n:
            
            i = random.randint(0, w-args.size-1)
            j = random.randint(0, h//2-args.size-1) if 'test' not in args.dir_out else random.randint(h//2, h-args.size-1)
            #print(k, i, j)
            box = (i, j, i+args.size, j+args.size)

            out_input1 = os.path.join(args.dir_out, 'input_hh', f'{i}_{j}{ext}')
            out_input2 = os.path.join(args.dir_out, 'input_hv', f'{i}_{j}{ext}')

            out1 = os.path.join(args.dir_out + 'autocors_hh', f'{i}_{j}{ext}')
            out2 = os.path.join(args.dir_out + 'autocors_hv', f'{i}_{j}{ext}')
            out3 = os.path.join(args.dir_out + 'autocors_hh_hv', f'{i}_{j}{ext}')
            out4 = os.path.join(args.dir_out + 'autocors_hv_hh', f'{i}_{j}{ext}')

            out_classes = os.path.join(args.dir_out, 'classes', f'{i}_{j}{ext}')

            out_seg = os.path.join(args.dir_out, 'target', f'{i}_{j}{ext}')

            seg = segment.crop(box)
           

            cols, counts= np.unique(np.asarray(seg).reshape(-1, np.asarray(seg).shape[2]), axis=0, return_counts=True)
            if len(cols) == 1 and np.sum(cols[0]).item() == 610:
                print("blue")
                continue
            land_indikator = False
            for i in cols:
                if int(np.sum(i).item()) == 0:
                    print('land')
                    land_indikator = True
                    break

            if land_indikator:
                continue

                
            classes = np.zeros((4,1))
            for index, i in enumerate(cols):
                #print(i)
                if int(np.sum(i).item()) != 0:
                    if args.persent:
                        classes[d[int(np.sum(i).item())]] = counts[index]/np.sum(counts)
                    else:
                        classes[d[int(np.sum(i).item())]] += 1
            #print(classes)
           
            np.save(out_classes, classes)

            seg.save(out_seg)
            #print(np.asarray(seg).shape)

            temp_img = img.crop(box)
            temp_img2 = img2.crop(box)

            matplotlib.image.imsave(out_input1, temp_img)
            matplotlib.image.imsave(out_input2, temp_img2)

            temp_img = np.array(temp_img)
            temp_img2 = np.array(temp_img2)


            kernel1 = np.copy(np.array(temp_img)[args.size//4:3*args.size//4, args.size//4:3*args.size//4])
            kernel2 = np.copy(np.array(temp_img2)[args.size//4:3*args.size//4, args.size//4:3*args.size//4])
            

            #images_auto_cor[name[0]].append(signal.convolve(temp_img.astype('float128'), kernel.astype('float128'),  mode='valid', method='direct'))
            #plt.imshow(self_ac(temp_img.astype('float128'), kernel1.astype('float128')))
            #plt.show()
            #print(self_ac(temp_img.astype('float128'), kernel1.astype('float128')).min())
            matplotlib.image.imsave(out1, self_ac(temp_img.astype('float128'), kernel1.astype('float128')))
            matplotlib.image.imsave(out2, self_ac(temp_img2.astype('float128'), kernel2.astype('float128')))
            matplotlib.image.imsave(out3, self_ac(temp_img.astype('float128'), kernel2.astype('float128')))
            matplotlib.image.imsave(out4, self_ac(temp_img2.astype('float128'), kernel1.astype('float128')))
            print("k = ", k)
            """
            plt.imshow(self_ac(temp_img.astype('float128'), kernel1.astype('float128')), cmap="gray")
            plt.show()
            plt.imshow(temp_img)
            plt.show()
            """
            k += 1
           
            



    """else:
        grid = product(range(0, h-h%args.size, args.size), range(0, w-w%args.size, args.size))
        for i, j in grid:
            box = (j, i, j+args.size, i+args.size)
            out = os.path.join(args.dir_out, f'{name}_{i}_{j}{ext}')
            img.crop(box).save(out)
    """