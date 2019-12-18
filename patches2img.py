import numpy as np
from PIL import Image
import os
from functools import reduce
import argparse

#print(os.listdir(path+'/buffer/'))
def concat_x(x1,x2):
    return np.concatenate([x1,x2],axis=1)
def concat_y(y1,y2):
    return np.concatenate([y1,y2],axis=0)
def img_loader(path):
    im = np.asarray(Image.open(path))
    return im

#print(os.listdir(path+'/buffer/'))
def concat_x(x1,x2):
    return np.concatenate([x1,x2],axis=1)
def concat_y(y1,y2):
    return np.concatenate([y1,y2],axis=0)
def img_loader(path):
    im = np.asarray(Image.open(path))
    return im

def patches2img(h1,w1=None,path=None):
    if w1 is None:
        w1 = h1
    m = h1//512
    im = []
    for i in range(h1//512):
        row = [os.path.join(path,'buffer',str(h1)+'_'+str(w1)+'_'+str(i*(w1//512)+j)+'.png') \
               for j in range(w1//512)]
        row = list(map(img_loader,row))
        row = reduce(concat_x,row)
        im.append(row)
    im = reduce(concat_y,im)
    #fig = plt.figure(dpi=200)
    #plt.imshow(im)
    #plt.show()
    im = Image.fromarray(im)
    im.save(os.path.join(path,str(h1)+'_'+str(w1)+'.png'))
    print("(%i,%i) shape of image in path  %s  saved!"%(h1,w1,path))

parser = argparse.ArgumentParser()    
parser.add_argument('--path',help = 'path of image patches',type = str)
a = parser.parse_args()
if __name__ == '__main__':
    patches2img(1024,path=a.path)
    patches2img(2048,path=a.path)
    patches2img(4096,path=a.path)
    patches2img(8192,path=a.path)
    
    
    
