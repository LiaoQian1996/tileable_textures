from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from lib.model import data_loader, generator, Synthesis
from lib.ops import *
import math
import time
import numpy as np
import scipy.misc
import cv2
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

parser = argparse.ArgumentParser()

parser.add_argument(
    '--output_dir',
    help = 'output dictionary',
    default = './result/'
)

parser.add_argument(
    '--vgg_ckpt',
    help = 'checkpoint of vgg networks, the check point file of pretrained model should be downloaded',
    default = '/home/liaoqian/DATA/vgg19/vgg_19.ckpt'
)

parser.add_argument(
    '--target_dir',
    help = 'path of target img, texture sample image or style image',
    default = './imgs/tomato.png' 
)

parser.add_argument(
    '--initials',
    help = 'initialized mode of synthesis, come into force only in style_transfer task_mode',
    choices = ['noise', 'content'],
    default = 'noise'
)

parser.add_argument(
    '--top_style_layer',
    help = 'the top layer of vgg network layers used to compute style_loss',
    default = 'VGG54',
    choices = ['VGG11','VGG21','VGG31','VGG41','VGG51','VGG54']
)

parser.add_argument(
    '--texture_shape',
    help = 'img_size of synthesis output texture, if set to [-1,-1], the shape will be \
    the same as sample texture image',
    nargs = '+',
    type = int
)

parser.add_argument(
    '--pyrm_layers',
    help = 'layers number of pyramid',
    default = 6,
    type = int
)

parser.add_argument('--W_tv',help = 'weight of total variation loss',type = float,default = 0.1)
parser.add_argument('--pad',help='padding size',type=int,default=8)
parser.add_argument('--stddev',help = 'standard deviation of noise',type = float,default = 0.1)
parser.add_argument('--max_iter',help = 'max iteration',type = int,default = 100,required = True)
parser.add_argument('--print_loss',help = 'whether print current loss',action = 'store_true')

FLAGS = parser.parse_args()
print_configuration_op(FLAGS)

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')
    
# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

if not os.path.exists(os.path.join(FLAGS.output_dir, 'buffer')):
    os.mkdir(os.path.join(FLAGS.output_dir, 'buffer'))

# pyramid = get_pyramid(targets, pyramid_layers) # a list store the pyramid of target textures image
pyrm_layers = FLAGS.pyrm_layers
tar_name = FLAGS.target_dir.split('/')[-1]; tar_name = tar_name.split('.')[0]
targets0 = data_loader(img_dir=None, FLAGS=FLAGS)
# targets = to_tensor(targets0)

#print(os.listdir(path+'buffer/'))
'''
filleds = os.listdir(os.path.join(FLAGS.output_dir,'buffer'))
n = 2048
m = n//512
filleds = [int((_.split('_')[1]).split('.')[0]) for _ in filleds if str(n) in _]
def ind0(i):
    return i//(m//2)*2*m + i%(m//2)*2
flag = [(i in filleds) for i in range(m**2)]
filled_inds = [flag[ind0(i)] and flag[ind0(i)+1] and flag[ind0(i)+m] and flag[ind0(i)+m+1] \
               for i in range((m//2)**2)]
unfilled_inds = [i for i in range((m//2)**2) if filled_inds[i]==False]
begin = FLAGS.texture_shape[0]//n - 1
'''

begin = pyrm_layers - 1
start_time = time.time()
for i in range(begin, -1, -1):
    # targets = to_tensor(targets0)
    # w0, h0 = [targets0.shape[1], targets0.shape[2]]; w, h = [ w0//(2**i), h0//(2**i) ]
    # target = tf.image.resize_bicubic(targets, [w, h]) 
    target = targets0[:,::(2**i),::(2**i),:]
    if target.size > 3000000:
        target = check_size_crop(target)
    h1, w1 = [ FLAGS.texture_shape[0]//(2**i), FLAGS.texture_shape[1]//(2**i)]  
    h0,w0  = h1//1024,w1//1024
    print('\nCurrent image : ', tar_name)
    print('Target image size : (%d, %d)'%(target.shape[1],target.shape[2]))
    print('Synthesizing image size : (%d, %d)' % (h1, w1))
    print('Now in pyramid layer %i, total %i layer (from L%i to L0)\n' \
          %(i, pyrm_layers, pyrm_layers-1))  
    print('Time has past %.3f mins'%((time.time()-start_time)/60))
    if i == pyrm_layers - 1:
        try:
            Synthesis(target, layer = i, FLAGS = FLAGS)     
        except:
            raise ValueError('Pyramid is too higher ! ')        
    elif (h1//512)*(w1//512) <= 10:   
        '''
        第一阶段将在合成图像除以512之后的的长宽积,在 乘以4就将大于10时结束！
        这是因为本机硬件支持单次合成的最大尺寸为[512*2,512*5]
        '''
        Synthesis(target, layer = i, FLAGS = FLAGS)     
    else:
        if i == begin:
            print('unfilled_inds : ',unfilled_inds)
            for ind in unfilled_inds:
                print('\n\tTime has past %.3f minitues'%((time.time()-start_time)/60))
                print('\t%i/%i 512 patch of (%i,%i) image upsampled!'%(ind+1,h0*w0,h1//2,w1//2))
                Synthesis(target, layer = i, ind = ind, FLAGS = FLAGS)      
        else:
            for ind in range(h0*w0):
                print('\n\tTime has past %.3f minitues'%((time.time()-start_time)/60))
                print('\t%i/%i 512 patch of (%i,%i) image upsampled!'%(ind+1,h0*w0,h1//2,w1//2))
                Synthesis(target, layer = i, ind = ind, FLAGS = FLAGS)
os.mkdir(os.path.join(FLAGS.output_dir,'%.3f'%((time.time()-start_time)/60)))
print('Optimization done !!! ') 