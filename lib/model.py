from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from functools import partial
from lib.ops import *
import collections
import os
import math
from PIL import Image
import numpy as np
import time
    
# Define the dataloader
def data_loader(img_dir = None, FLAGS = None):
    with tf.device('/cpu:0'):
        if img_dir is None:
            image_raw = Image.open(FLAGS.target_dir)
        else:
            image_raw = Image.open(img_dir)
        if FLAGS.texture_shape == [-1,-1]:
            image_raw = check_size(image_raw)
        if image_raw.mode is not 'RGB':
            image_raw = image_raw.convert('RGB')
        image_raw = np.asarray(image_raw)/255
        targets = preprocess(image_raw)  
        samples = np.expand_dims(targets, axis = 0) 
    return samples

def generator(h1,w1, initials = None, rf1=None, rf2=None, FLAGS=None):
    if initials is None:
        shape = [1, h1, w1, 3]
        var1 = tf.get_variable('gen_img',shape = shape, \
                              initializer = tf.random_normal_initializer(0,0.5),\
                              dtype=tf.float32,trainable=True, collections=None)  
    else:
        var = initials + tf.random_normal(tf.shape(initials), 0, FLAGS.stddev)
        if rf1 is None:
            '''
            没有参考图像，说明只是简单的 upsampling + gram_loss形式
            '''
            pass
        elif rf2 is None:
            '''
            只有参考图像rf1，那么需要返回的变量是一个[1024,1536]的图像
            '''
            var = tf.concat([rf1, var], axis = 2)

        else:
            var = tf.concat([rf1, var], axis = 2)
            var = tf.concat([rf2, var], axis = 1)
            '''
            返回一个[1536,1536]的大图像
            '''
        if FLAGS.pad == 0:
            var1 = tf.Variable(var)
        else:
            var1 = tf.Variable(tf.pad(var,[[0,0],[FLAGS.pad,FLAGS.pad],[FLAGS.pad,FLAGS.pad],[0,0]], "REFLECT"))        
    return tf.tanh(var1)

def Synthesis(targets, layer, ind=0, FLAGS=None, reuse=True):
    h1,w1 = FLAGS.texture_shape[0]//(2**layer),FLAGS.texture_shape[1]//(2**layer)
    h0,w0 = h1//1024, w1//1024
    '''
    旧坐标为 ind//w0, ind%w0
    新坐标为 (ind//w0)*2, (ind%w0)*2
    新索引为 (ind//w0)*2 * (2*h0) + (ind%w0)*2
    '''
    if (h1//512)*(w1//512) > 10 and ind%w0 == 0 and ind!=0 :            
        targets = tf.transpose(to_tensor(targets),[0,2,1,3])
    else:
        targets = tf.identity(to_tensor(targets))
        
    img_loader = partial(data_loader, FLAGS=FLAGS) 
    
    def upsampling(initials):
        w, h = [ initials.shape[1], initials.shape[2] ]
        initials = tf.image.resize_bicubic(initials, [2*int(w), 2*int(h)])
        return initials
    with tf.variable_scope('generator'):
        if (h1//512)*(w1//512) > 10:
            ind0 = ((ind//w0)*2)*2*w0 + (ind%w0)*2
        if layer == FLAGS.pyrm_layers - 1:
            '''
            最底层，最小分辨率的合成图像，从噪声初始化
            '''
            init = None
            gen_output = generator(h1,w1, initials = init, FLAGS = FLAGS)
        elif ind == 0:
            '''
            由上一层图像上采样作为初始化
            '''
            img_dir = os.path.join(FLAGS.output_dir,'buffer','%i_%i_0.png'%(h1//2,w1//2))
            init = img_loader(img_dir=img_dir)
            init = upsampling(init)
            gen_output = generator(h1,w1, initials = init, FLAGS = FLAGS)
        elif ind//w0 == 0:
            '''
            首行，但不是首个图像;此时init的尺寸为固定的 512 -> 1024 大小
            '''
            init_dir = os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%(h1//2,w1//2,ind))
            init = upsampling(img_loader(img_dir=init_dir))
            rf_left_up = os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%(h1,w1,ind0-1))
            rf_left_down=os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%(h1,w1,ind0-1+w0*2))
            rf_left = list(map(img_loader, [rf_left_up, rf_left_down]))
            rf_left = tf.concat(list(map(to_tensor, rf_left)), axis=1)
            gen_output = generator(h1,w1, initials=init, rf1=rf_left,FLAGS=FLAGS)
            
        elif ind%w0 == 0:
            '''
            非首行，首个图像
            '''
            init_dir = os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%(h1//2,w1//2,ind))
            init = tf.transpose(upsampling(img_loader(img_dir=init_dir)),perm=[0,2,1,3])
            rf_left_up = os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%\
                                                              (h1,w1,ind0-w0))
            rf_left_down=os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%\
                                      (h1,w1,ind0-w0+1))
            rf_left = list(map(img_loader, [rf_left_up, rf_left_down]))
            rf_left = tf.concat(list(map(to_tensor, rf_left)), axis=2)
            rf_left = tf.transpose(rf_left, perm=[0,2,1,3])
            gen_output = generator(h1, w1, initials=init, rf1=rf_left,FLAGS=FLAGS)        
        
        else:
            '''
            在当前层，当前1024块的第0个512块的索引为ind0
            '''
            init_dir = os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%(h1//2,w1//2,ind))
            init = upsampling(img_loader(img_dir=init_dir))
            rf_left_up = os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%\
                                      (h1,w1,ind0 -1))
            rf_left_down=os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%\
                                      (h1,w1,ind0 - 1 + w0*2))
            rf_left = list(map(img_loader, [rf_left_up, rf_left_down]))
            rf_left = tf.concat(list(map(to_tensor, rf_left)), axis=1)    
            
            rf_up_left = os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%\
                                      (h1,w1,ind0 - w0*2 - 1))
            rf_up_middle = os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%\
                                      (h1,w1,ind0 - w0*2))
            rf_up_right = os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%\
                                      (h1,w1,ind0 - w0*2 + 1))
            rf_up = list(map(img_loader, [rf_up_left, rf_up_middle, rf_up_right]))            
            rf_up = tf.concat(list(map(to_tensor, rf_up)), axis=2)                    
            gen_output = generator(h1,w1, initials=init, rf1=rf_left, rf2=rf_up, FLAGS=FLAGS)     
            
            rf = tf.concat([rf_left,tf.zeros([1,1024,1024,3])],axis=2)
            rf = tf.concat([rf_up,rf],axis=1)
            
    # Calculating the generator loss
    with tf.name_scope('generator_loss'):   
        #print('gen_output : ',gen_output)
        with tf.name_scope('tv_loss'):
            tv_loss = total_variation_loss(gen_output)

        with tf.name_scope('style_loss'):
            _, vgg_gen_output = vgg_19(gen_output,is_training=False, reuse = False)
            _, vgg_tar_output = vgg_19(targets,   is_training=False, reuse = True)
            style_layer_list = get_layer_list(FLAGS.top_style_layer,False)
            sl = tf.zeros([])
            ratio_list=[100.0, 1.0, 0.1, 0.0001, 1.0, 100.]  # [100.0, 1.0, 0.1, 0.0001, 1.0, 100.0] 
            for i in range(len(style_layer_list)):
                tar_layer = style_layer_list[i]
                target_layer = get_layer_scope(tar_layer)
                gen_feature = vgg_gen_output[target_layer]
                tar_feature = vgg_tar_output[target_layer]
                diff = tf.square(gram(gen_feature) - gram(tar_feature))
                sl = sl + tf.reduce_mean(tf.reduce_sum(diff, axis=0)) * ratio_list[i] 
            style_loss = sl
        
        with tf.name_scope('decay_mse_loss'):
            if ind == 0:
                decay_mse_loss = tf.zeros([])
            else:
                gen_output1 = gen_output[:,FLAGS.pad:-FLAGS.pad,FLAGS.pad:-FLAGS.pad,:]
                #print('gen_output1 : ',gen_output1)
                if ind//w0 == 0 or ind%w0 == 0:
                    decay_ratio = 1 - np.linspace(0,1,256)
                    decay_ratio = np.concatenate([np.ones(128),decay_ratio,np.zeros(128)])
                    decay_ratio = decay_ratio.reshape([1,1,512,1])
                    decay_mse_loss = decay_ratio * tf.square(gen_output1[:,:,0:512,:] - rf_left) 
                    decay_mse_loss = tf.reduce_mean(decay_mse_loss)
                else:
                    decay_ratio = 1 - np.linspace(0,1,256)
                    #decay_ratio = decay_ratio.astype(np.float32)
                    decay_ratio = np.concatenate([np.ones(128),decay_ratio,np.zeros(128+1024)])
                    decay_ratio = decay_ratio.reshape([1,1,1536,1])
                    decay_ratio_2d = decay_ratio * np.transpose(decay_ratio,[0,2,1,3])
                    decay_mse_loss = tf.reduce_mean(decay_ratio_2d * tf.square(gen_output1 - rf))
            
        gen_loss = style_loss + FLAGS.W_tv * tv_loss + 0.1*decay_mse_loss
        gen_loss = 1e6 * gen_loss

    with tf.name_scope('generator_train'):
        gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'generator')
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            gen_loss, var_list = gen_tvars, method='L-BFGS-B',
            options = {'maxiter': FLAGS.max_iter, 'disp': FLAGS.print_loss})
    '''
    for _ in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'generator'):
            print(_)
    '''
    
    vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
    vgg_restore = tf.train.Saver(vgg_var_list)

    # Start the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    def print_loss(gl, sl, tvl, dml):
        if FLAGS.print_loss is True:
            print('gen_loss : %s' % gl )
            print('style_loss : %s' % sl )
            print('tv_loss : %s' % tvl )
            print('decay_mse_loss : %s' %dml)
    
    init_op = tf.global_variables_initializer()   
    with tf.Session(config = config) as sess:
        sess.run(init_op)
        vgg_restore.restore(sess, FLAGS.vgg_ckpt)
        print('\tUnder Synthesizing ...')
        #start = time.time()
        optimizer.minimize(sess, loss_callback = print_loss,
                         fetches = [gen_loss, style_loss, tv_loss, decay_mse_loss])
        gen_output, style_loss = gen_output.eval(), style_loss.eval()
    
    tf.reset_default_graph()
    
    if init is None or FLAGS.pad == 0:
        gen_output = gen_output[:,:,:,:]        
    else:
        gen_output = gen_output[:,FLAGS.pad:-FLAGS.pad,FLAGS.pad:-FLAGS.pad,:]
    
    if (h1//512)*(w1//512) < 4:
        '''
        第一阶段，上采样+Gram_loss
        '''
        path = os.path.join(FLAGS.output_dir,'buffer','%i_%i_%i.png'%(h1,w1,ind))
        save_img(gen_output, path)
    elif (h1//512)*(w1//512) <= 10:
        '''
        第一阶段结束：上采样+Gram_loss能够合成的最大图像
        '''
        save_whole_img(gen_output,h1,w1,FLAGS)
    elif ind == 0:
        '''
        第一行第一列图像，直接将整个gen_output (1024,1024)保存
        '''
        save_patches(gen_output, ind, h1, w1 , FLAGS)
    elif ind//w0 == 0:
        '''
        第一行，不是第一列图像，保存合成图像并更新参考图像
        '''
        save_img(gen_output[:,0:512,0:512,:], rf_left_up)
        save_img(gen_output[:,512:1024,0:512,:],rf_left_down)
        save_patches(gen_output[:,:,512:,:], ind, h1, w1, FLAGS)
    elif ind%w0 == 0:
        '''
        不是第一行，第一列
        '''
        save_patches(np.transpose(gen_output[:,:,512:,:],[0,2,1,3]), ind, h1, w1, FLAGS)
        save_img(np.transpose(gen_output[:,0:512,0:512,:],[0,2,1,3]), rf_left_up)
        save_img(np.transpose(gen_output[:,512:1024,0:512,:],[0,2,1,3]), rf_left_down)        
    else:
        '''
        不是第一行，也不是第一列
        '''
        save_patches(gen_output[:,512:,512:,:], ind, h1, w1, FLAGS)
        save_img(gen_output[:,0:512,0:512,:],    rf_up_left)
        save_img(gen_output[:,0:512,512:1024,:], rf_up_middle)
        save_img(gen_output[:,0:512,1024:,:],    rf_up_right)
        save_img(gen_output[:,512:1024,0:512,:], rf_left_up)
        save_img(gen_output[:,1024:,0:512,:],    rf_left_down) 
            
            
            
            
            
            
            
            
            
