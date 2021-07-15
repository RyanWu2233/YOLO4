# -*- coding: utf-8 -*-
"""
Module name: yolo4_networks
Author:      Ryan Wu
Date:        V0.1- 2020/11/08: Initial release
Description: Network for YOLO V4
"""


#---------------------------------------------------- Import libraries
from __future__ import absolute_import, division, unicode_literals, print_function
import tensorflow as tf 
#import tensorflow_addons as tfa

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Layer, UpSampling2D
from tensorflow.keras.layers import Add, MaxPool2D, Concatenate

#---------------------------------------------------- Layer definition 
class Mish(Layer):
    def __init__(self):
        super(Mish, self).__init__()
    def compute_output_shape(self, input_shape):
        return input_shape
    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x)) 
#---------------------------------------------------- Building block definition 
    #-------------------------------------------    # Conv + Batch Norm + Mish
def CBM(x, ch, kernels, strides = 1, padding= 'same'):
    if strides == 2: padding= 'valid'
    x = Conv2D(ch, 
               kernels, 
               strides   = strides,
               padding   = padding, 
               use_bias  = False,
               activation= 'linear',
               kernel_initializer= tf.random_normal_initializer(stddev=0.01))(x);
    x = BatchNormalization()(x)
    x = Mish()(x);
    return x                               
    #-------------------------------------------    # Conv + Batch Norm + Leaky
def CBL(x, ch, kernels, strides = 1, padding= 'same'):
    if strides == 2: padding= 'valid'
    x = Conv2D(ch, 
               kernels, 
               padding   = padding, 
               strides   = strides,
               use_bias  = False,
               activation= 'linear',
               kernel_initializer= tf.random_normal_initializer(stddev=0.01))(x);
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha= 0.1)(x);
    return x    
    #-------------------------------------------    # Residual block
def RES(xin, ch, blocks= 1, ra=1):
    sc   = xin;
    for k in range(blocks):
        r0 = CBM(sc,    ch, 1);
        r1 = CBM(r0, ra*ch, 3);
        sc = Add()([sc, r1]); 
    return sc 
    #-------------------------------------------    # CSP-Darknet 1
def CSP(xin, chd, blocks= 1, ra=1):
    ch    = chd//2;
    xpad  = ZeroPadding2D(padding=((1,0),(0,1)))(xin)
    xdown = CBM( xpad, ch*2,  3, 2)
    xsc   = CBM(xdown, ch*ra, 1)
    xpre  = CBM(xdown, ch*ra, 1)
    xres  = RES( xpre, ch,    blocks, ra)
    xpost = CBM( xres, ch*ra, 1) 
    xconc = Concatenate()([xpost, xsc])
    xout  = CBM(xconc, ch*2, 1)
    return xout     
    #-------------------------------------------    # SPP network
def SPP(xin):
    x0    = CBL( xin,  512, 1)
    x0    = CBL(  x0, 1024, 3)
    x0    = CBL(  x0,  512, 1)
    max13 = MaxPool2D( (13, 13), strides= 1, padding='same')(x0)
    max9  = MaxPool2D( ( 9,  9), strides= 1, padding='same')(x0)
    max5  = MaxPool2D( ( 5,  5), strides= 1, padding='same')(x0)
    x1    = Concatenate()([max13, max9, max5, x0])
    x1    = CBL(  x1,  512, 1)
    x1    = CBL(  x1, 1024, 3)
    x1    = CBL(  x1,  512, 1)
    return x1  
    #-------------------------------------------    # FPN network 
def FPN(xin, sc, ch):
    x0    = CBL( xin,   ch, 1)
    x1    = UpSampling2D(2)(x0)
    sc0   = CBL(  sc,   ch, 1)
    x2    = Concatenate()([ sc0, x1])
    x2    = CBL(  x2,   ch, 1)
    x2    = CBL(  x2, 2*ch, 3)
    x2    = CBL(  x2,   ch, 1)
    x2    = CBL(  x2, 2*ch, 3)
    x2    = CBL(  x2,   ch, 1)
    return x2
    #-------------------------------------------    # PANet
def PAN(xin, sc, ch):
    x0    = ZeroPadding2D(padding=((1,0),(0,1)))(xin)
    x1    = CBL(  x0,   ch, 3, 2)
    x2    = Concatenate()([x1, sc])
    x2    = CBL(  x2,   ch, 1)
    x2    = CBL(  x2, 2*ch, 3)
    x2    = CBL(  x2,   ch, 1)
    x2    = CBL(  x2, 2*ch, 3)
    x2    = CBL(  x2,   ch, 1)
    return x2    
    #-------------------------------------------    # Header
def HEAD(xin, ch1, ch2):
    x0   = CBL(xin, ch1, 3)
    xout = Conv2D(ch2, 1, 1, padding='same')(x0)
    return xout
#----------------------------------------------------
#---------------------------------------------------- YOLO4 model
def YOLO4_model(img_size= 416, num_anchors= 3, num_classes= 80):
    """ Notice:
        Pre-train model is based on coco set, num_classes = 80
    """
    ch_out = num_anchors * (num_classes + 5)
    """ Backbone: CSP darknet 53 """
    inputs = Input(shape= (img_size, img_size, 3))
    s1     = CBM(inputs, 32, 3)                     # [bs, 608, 608,   32]
    s2     = CSP(  s1,   64, blocks= 1, ra= 2)      # [bs, 304, 304,   64]
    s4     = CSP(  s2,  128, blocks= 2)             # [bs, 152, 152,  128]
    s8     = CSP(  s4,  256, blocks= 8)             # [bs,  76,  76,  256]
    s16    = CSP(  s8,  512, blocks= 8)             # [bs,  38,  38,  512]
    s32    = CSP( s16, 1024, blocks= 4)             # [bs,  19,  19, 1024] 
    """ Neck: SPP, FPN """
    t32    = SPP( s32)                              # [bs,  19,  19,  512] 
    t16    = FPN( t32, s16, 256)                    # [bs,  38,  38,  256] 
    t8     = FPN( t16,  s8, 128)                    # [bs,  76,  76,  128]
    """ Header & PAN """     
    h8     = HEAD( t8,  256, ch_out)                # [bs,  76,  76,  255]
    u16    = PAN(  t8, t16, 256)                    # [bs,  38,  38,  256]
    h16    = HEAD(u16,  512, ch_out)                # [bs,  38,  38,  255]
    u32    = PAN( u16, t32, 512)                    # [bs,  19,  19,  512]
    h32    = HEAD(u32, 1024, ch_out)                # [bs,  19,  19,  255]
    return Model(inputs, [h32, h16, h8])   
#----------------------------------------------------



    
