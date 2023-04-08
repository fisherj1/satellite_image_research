from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

img = Image.open('data/s1a-ew-grd-hh-20200709t030543-20200709t030619-033372-03ddcb-001.tiff')
img2 = Image.open('data/s1a-ew-grd-hv-20200709t030543-20200709t030619-033372-03ddcb-002.tiff')
segment = (Image.open('data/result.tif'))

print(np.asarray(img).shape, np.asarray(img2).shape, np.asarray(segment).shape)



colors = [[  0,   0,   0,   0],
       [  0,  34, 223, 255],
       [  0, 100, 255, 255],
       [  0, 250,   0, 255],
       [171, 243, 255, 255]]

"""d = {[  0,   0,   0,   0]:'land', 0-
       [  0,  34, 223, 255]:'dark blue'512,
       [  0, 100, 255, 255]:'blue',610
       [  0, 250,   0, 255]:'green', 505
       [171, 243, 255, 255]: 'lite blue' }924


"""
d = {0:0, 512:0, 610:0, 505:0, 924: 0}

d1 = {0:'n', 512:'d', 610:'b', 505:'g', 924: 'l'}
while min(d.values()) < 5:
    w, h = np.asarray(img).shape[0], np.asarray(img).shape[1]
    print(w,h)
    i = random.randint(50, w-50-1)
    j = random.randint(500, h-500-1)
    box = (j, i, j+240, i+240)
    res1 = img.crop(box)
    res2 = segment.crop(box)
    res3 = img2.crop(box)
    #lll = np.concatenate((np.asarray(res2), np.asarray(segment)[i:i+128, j:j+128]))
    #plt.imshow(lll)
    if len(np.unique(np.asarray(res2).reshape(-1, 4), axis=0))==1:
      co = int(np.sum(np.unique(np.asarray(res2).reshape(-1, 4), axis=0)))
      if d[co] < 5:
        d[co]+=1
        res1.save('data/input240_hh/'+d1[co]+str(i)+'.tiff')
        res2.save('data/target240_tar/'+d1[co]+str(i)+'.tiff')
        res3.save('data/input240_hv/'+d1[co]+str(i)+'.tiff')
    print(d.values())