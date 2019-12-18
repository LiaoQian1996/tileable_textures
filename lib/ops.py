from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import pdb
import numpy as np
from PIL import Image
import scipy.misc
import os

def save_patches(img, ind, h1, w1, FLAGS):
    '''
    img是生成的1024图像，ind为上一层512块的索引,w1为当前合成图像的宽
    '''
    w0, h0 = w1//1024,h1//1024
    ind0 = (ind//w0)*2*(2*w0) + (ind%w0)*2
    print('\t%i/%i 1024 patch of (%i,%i) image saved !'%((ind+1), w0*h0, h1, w1))
    for j in range(4):
        k, l = map(int,[j//2, j%2])
        #print('\tNow we save No.%i 512 patch of No.%i 1024 patch in [%i,%i] Image'%(j,ind,n,n))
        save_img(np.squeeze(img[:, 512*k:512*(k+1), 512*l:512*(l+1), :]),\
                os.path.join(FLAGS.output_dir, 'buffer', '%i_%i_%i.png'%\
                         (h1, w1, ind0 + k*2*w0 + l )))  
        print('\t%i 512 patch of (%i,%i) image saved !'%(ind0+k*2*w0+l,h1, w1))        

def save_whole_img(img,h,w,FLAGS):
    '''
    将图像按512分为若干块，并进行保存
    '''
    path = os.path.join(FLAGS.output_dir,'buffer')
    h0, w0 = h//512, w//512
    for ind in range(w0*h0):
        x, y = ind//w0,ind%w0
        save_img(img[:,x*512:(x+1)*512,y*512:(y+1)*512,:], path+'/%i_%i_%i.png'%(h,w,ind))
    
def save_img(img, path):
    scipy.misc.toimage(np.squeeze(img),cmin=-1.0,cmax=1.0) \
    .save(path)
    
def to_tensor(input):
    return tf.constant(input, dtype = tf.float32)

def check_size(im):
    while(1):
        if (im.size[0] * im.size[1] > 3000000):
            w, h = int(0.9 * im.size[0]), int(0.9*im.size[1])
            print('Image size (%i,%i) is too large, will be resized to [%i,%i]'\
                  %(im.size[0], im.size[1], w, h))
            im = im.resize((w, h), Image.BICUBIC) 
        else:
            return im

def check_size_crop(im):
    _, h0, w0, _ = im.shape
    h,w = h0,w0
    while(h*w > 3000000):
        h, w = map(int,[h*0.9,w*0.9])
    print('Image size (%i,%i) is too large, will be croped to (%i,%i)'\
                  %(h0, w0, h, w))
    im = im[:,(h0-h)//2:(h0-h)//2+h,(w0-w)//2:(w0-w)//2+w,:]
    print('After crop, im.shape : ',im.shape)
    return im
    
def gram(features):
    features = tf.reshape(features,[-1,features.shape[3]])
    return tf.matmul(features,features,transpose_a=True) \
                 / tf.cast(features.shape[0]*features.shape[1],dtype=tf.float32)

def total_variation_loss(image):
    tv_y_size = tf.size(image[:,1:,:,:],out_type=tf.float32)
    tv_x_size = tf.size(image[:,:,1:,:],out_type=tf.float32)
    tv_loss =   (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:-1,:]) /
                    tv_x_size))
    return tv_loss

def get_layer_scope(name):
    target_layer = 'vgg_19/conv' + name[-2] + '/conv'+ name[-2] + '_' + name[-1]  
    return target_layer  

def get_layer_list(layer, single_layer=False):
    style_layers = []
    if single_layer == True:
        if layer == 'VGG11':
            style_layers = ['VGG11']
        elif layer == 'VGG21':
            style_layers = ['VGG21']
        elif layer == 'VGG31':
            style_layers = ['VGG31']
        elif layer == 'VGG41':
            style_layers = ['VGG41']
        elif layer == 'VGG51':
            style_layers = ['VGG51']
        elif layer == 'VGG54':
            style_layers = ['VGG54']
        else:
            raise ValueError("NO THIS LAYER !")
    else:
        if layer == 'VGG11':
            style_layers = ['VGG11']
        elif layer == 'VGG21':
            style_layers = ['VGG11','VGG21']
        elif layer == 'VGG31':
            style_layers = ['VGG11','VGG21','VGG31']
        elif layer == 'VGG41':
            style_layers = ['VGG11','VGG21','VGG31','VGG41']
        elif layer == 'VGG51':
            style_layers = ['VGG11','VGG21','VGG31','VGG41','VGG51']
        elif layer == 'VGG54':
            style_layers = ['VGG11','VGG21','VGG31','VGG41','VGG51','VGG54']
        else:
            raise ValueError(" No such layer in layer list.")
    return style_layers   

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2
    
# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    FLAGS = vars(FLAGS)
    for name, value in sorted(FLAGS.items()):
        if type(value) == float:
            print('\t%s: %f'%(name, value))
        elif type(value) == int:
            print('\t%s: %d'%(name, value))
        elif type(value) == str:
            print('\t%s: %s'%(name, value))
        elif type(value) == bool:
            print('\t%s: %s'%(name, value))
        else:
            print('\t%s: %s' % (name, value))
    print('End of configuration')

# VGG19 component
def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

# VGG19 net
"""Oxford Net VGG 19-Layers version E Example.
Note: All the fully_connected layers have been transformed to conv2d layers.
    To use in classification mode, resize input to 224x224.
Args:
inputs: a tensor of size [batch_size, height, width, channels].
num_classes: number of predicted classes.
is_training: whether or not the model is being trained.
dropout_keep_prob: the probability that activations are kept in the dropout
  layers during training.
spatial_squeeze: whether or not should squeeze the spatial dimensions of the
  outputs. Useful to remove unnecessary dimensions for classification.
scope: Optional scope for the variables.
fc_conv_padding: the type of padding to use for the fully connected layer
  that is implemented as a convolutional layer. Use 'SAME' padding if you
  are applying the network in a fully convolutional manner and want to
  get a prediction map downsampled by a factor of 32 as an output. Otherwise,
  the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
Returns:
the last op containing the log predictions and end_points dict.
"""
def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse = False,
           fc_conv_padding='VALID'):
    
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
            net = slim.avg_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    return net,end_points
vgg_19.default_image_size = 224
