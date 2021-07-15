# -*- coding: utf-8 -*-
"""
Module name: yolo4_tools
Author:      Ryan Wu
Date:        V0.1- 2020/11/27: Initial release
Description: Tools for yolo.v4 training
"""
#---------------------------------------------------- Import libraries
from __future__ import absolute_import, division, unicode_literals, print_function
import tensorflow as tf 
import os
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow_addons as tfa 

import yolo4_network, yolo4_decode, yolo4_utils   
from yolo4_network import YOLO4_model 
#---------------------------------------------------- Tools
    #-------------------------------------------

    #-------------------------------------------
def Tool_List_VOC_BBOX_No():
    box_no = yolo4_utils.get_stat_VOC()
    labels = yolo4_utils.get_class('VOC');
    print('VOC 2012  class and corresponding bounding box number')
    for k in range(len(box_no)): 
      print(f'{labels[k]:20s} : {str(int(box_no[k]))}')   
    #-------------------------------------------
def Tool_Convert_Weights():
    """ Original weights is downloaded from web site and saved as 'yolov4.weights'
    """
    weights_path = 'yolov4.weights'
    model_path   = os.path.expanduser('yolo4_weightx.h5') 
    yolo4_model  = YOLO4_model(img_size = 416);
    
    """ Load version control """
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(
            shape=(3, ), dtype='int32', buffer= weights_file.read(12))
    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)
    """ """
    convs_to_load = []                              # Conv2D layer list
    bns_to_load   = []                              # BatchNormalization layer list
    for i in range(len(yolo4_model.layers)):
        layer_name = yolo4_model.layers[i].name 
        if layer_name.startswith('conv2d_'):        # (number in name, index)
            convs_to_load.append((int(layer_name[7:]), i))
        if layer_name.startswith('batch_normalization_'):
            bns_to_load.append((int(layer_name[20:]), i))
    convs_sorted = sorted(convs_to_load, key=itemgetter(0))
    bns_sorted   = sorted(bns_to_load,   key=itemgetter(0))
    """ """ 
    bn_index = 0                                    # Index for Batch Normalization layer
    for i in range(len(convs_sorted)):
        print('Converting ', i, convs_sorted[i])
        if i == 93 or i == 101 or i == 109:         # Final output layer (no BN, with bias)
            weights_shape = yolo4_model.layers[convs_sorted[i][1]].get_weights()[0].shape
            bias_shape    = yolo4_model.layers[convs_sorted[i][1]].get_weights()[0].shape[3]
            filters       = bias_shape
            size          = weights_shape[0]
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size  = np.product(weights_shape)

            conv_bias = np.ndarray(
                    shape=(filters, ),
                    dtype='float32',
                    buffer=weights_file.read(filters * 4))
            conv_weights = np.ndarray(
                    shape=darknet_w_shape,
                    dtype='float32',
                    buffer=weights_file.read(weights_size * 4))
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            yolo4_model.layers[convs_sorted[i][1]].set_weights([conv_weights, conv_bias])
        else:                                       #--- Normal CBL, CBM layers
            weights_shape = yolo4_model.layers[convs_sorted[i][1]].get_weights()[0].shape
            size          = weights_shape[0]
            bn_shape      = yolo4_model.layers[bns_sorted[bn_index][1]].get_weights()[0].shape
            filters       = bn_shape[0]
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size  = np.product(weights_shape)

            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            bn_weights = np.ndarray(
                shape=(3, filters),
                dtype='float32',
                buffer=weights_file.read(filters * 12))

            bn_weight_list = [
                bn_weights[0],  # scale gamma
                conv_bias,  # shift beta
                bn_weights[1],  # running mean
                bn_weights[2]  # running var
                ]
            yolo4_model.layers[bns_sorted[bn_index][1]].set_weights(bn_weight_list)

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            yolo4_model.layers[convs_sorted[i][1]].set_weights([conv_weights])

            bn_index += 1
    """ Save weight to model file """
    weights_file.close()
    yolo4_model.save(model_path)
    print('Done') 

#---------------------------------------------------- Tools
def Tools_List_Network():
    s=[]
    s.append(' Layer    : Type     Kernel      Output          From');
    s.append('   0      : Input                (608,608,   3)  Input')  
    s.append('   1 ~   3: CBM      (3,3)*32    (608,608,  32)    0')
    s.append('------------------- Darkent Stage 1 --------------------')
    s.append('   4      : Padding              (609,609,  32)    3')
    s.append('   5 ~   7: CBM      (3,3)*64/2  (304,304,  64)    4')
    s.append('   8 ~  10: CBM      (1,1)*64    (304,304,  64)    7');
    s.append('R 11 ~  13: CBM      (1,1)*64    (304,304,  64)   10');
    s.append('R 14 ~  16: CBM      (3,3)*64    (304,304,  64)   13');
    s.append('+ 17      : Add                  (304,304,  64)   16, 10');
    s.append('  18,20,22: CBM      (1,1)*64    (304,304,  64)   17');
    s.append('  19,21,23: CBM      (1,1)*64    (304,304,  64)    7');
    s.append('  24      : Concat               (304,304, 128)   22, 23');
    s.append('  25 ~  27: CBM      (1,1)*64    (304,304,  64)   24');
    s.append('------------------- Darkent Stage 2 --------------------')
    s.append('  28      : Padding              (305,305,  64)   27');
    s.append('  29 ~  31: CBM      (3,3)*128/2 (152,152, 128)   28');
    s.append('  32 ~  34: CBM      (1,1)*64    (152,152,  64)   31');
    s.append('R 35 ~  41: Resnet        *64    (152,152,  64)   34');
    s.append('R 42 ~  48: Resnet        *64    (152,152,  64)   41');
    s.append('  49,51,53: CBM      (1,1)*64    (152,152,  64)   48');
    s.append('  50,52,54: CBM      (1,1)*64    (152,152,  64)   31');
    s.append('  55      : Concat               (152,152, 128)   53, 54');
    s.append('  56 ~  58: CBM      (1,1)*128   (152,152, 128)   55');
    s.append('------------------- Darkent Stage 3 --------------------')
    s.append('  59      : Padding              (153,153, 128)   58');
    s.append('  60 ~  62: CBM      (3,3)*256/2 ( 76, 76, 256)   59');
    s.append('  63 ~  65: CBM      (1,1)*128   ( 76, 76, 128)   62');
    s.append('R 66 ~  72: Resnet        *128   ( 76, 76, 128)   65');
    s.append('R 73 ~  79: Resnet        *128   ( 76, 76, 128)   72');
    s.append('R 80 ~  86: Resnet        *128   ( 76, 76, 128)   79');
    s.append('R 87 ~  93: Resnet        *128   ( 76, 76, 128)   86');
    s.append('R 94 ~ 100: Resnet        *128   ( 76, 76, 128)   93');
    s.append('R101 ~ 107: Resnet        *128   ( 76, 76, 128)  100');
    s.append('R108 ~ 114: Resnet        *128   ( 76, 76, 128)  107');
    s.append('R115 ~ 121: Resnet        *128   ( 76, 76, 128)  114');
    s.append(' 122,+2,+2: CBM      (1,1)*128   ( 76, 76, 128)  121');
    s.append(' 123,+2,+2: CBM      (1,1)*128   ( 76, 76, 128)   62');
    s.append(' 128      : Concat               ( 76, 76, 256)  126,127');
    s.append(' 129 ~ 131: CBM      (1,1)*256   ( 76, 76, 256)  128');
    s.append('------------------- Darkent Stage 4 --------------------')
    s.append(' 132      : Padding              ( 77, 77, 256)  131');
    s.append(' 133 ~ 135: CBM      (3,3)*512/2 ( 38, 38, 512)  132');
    s.append(' 136 ~ 138: CBM      (1,1)*256   ( 38, 38, 256)  135');
    s.append('R139 ~ 145: Resnet        *256   ( 38, 38, 256)  138');
    s.append('R146 ~ 152: Resnet        *256   ( 38, 38, 256)  145');
    s.append('R153 ~ 159: Resnet        *256   ( 38, 38, 256)  152');
    s.append('R160 ~ 166: Resnet        *256   ( 38, 38, 256)  159');
    s.append('R167 ~ 173: Resnet        *256   ( 38, 38, 256)  166');
    s.append('R174 ~ 180: Resnet        *256   ( 38, 38, 256)  173');
    s.append('R181 ~ 187: Resnet        *256   ( 38, 38, 256)  180');
    s.append('R188 ~ 194: Resnet        *256   ( 38, 38, 256)  187');
    s.append(' 195,+2,+2: CBM      (1,1)*256   ( 38, 38, 256)  194');
    s.append(' 196,+2,+2: CBM      (1,1)*256   ( 38, 38, 256)  135');
    s.append(' 201      : Concat               ( 38, 38, 512)  199,200');
    s.append(' 202 ~ 204: CBM      (1,1)*512   ( 38, 38, 512)  201');
    s.append('------------------- Darkent Stage 5 --------------------')
    s.append(' 205      : Padding              ( 39, 39, 512)  204');
    s.append(' 206 ~ 208: CBM      (3,3)*1024/2( 19, 19,1024)  205');
    s.append(' 209 ~ 211: CBM      (1,1)*512   ( 19, 19, 512)  208');
    s.append('R212 ~ 218: Resnet        *512   ( 19, 19, 512)  211');
    s.append('R219 ~ 225: Resnet        *512   ( 19, 19, 512)  218');
    s.append('R226 ~ 232: Resnet        *512   ( 19, 19, 512)  225');
    s.append('R233 ~ 239: Resnet        *512   ( 19, 19, 512)  232');
    s.append(' 240,+2,+2: CBM      (1,1)*512   ( 19, 19, 512)  239');
    s.append(' 241,+2,+2: CBM      (1,1)*512   ( 19, 19, 512)  208');
    s.append(' 246      : Concat               ( 19, 19,1024)  244,245');
    s.append(' 247 ~ 249: CBM      (1,1)*1024  ( 19, 19,1024)  246');
    s.append('------------------- PAN net (upsample 1) ---------------')
    s.append(' 250 ~ 252: CBL      (1,1)*512   ( 19, 19, 512)  249');
    s.append(' 253 ~ 255: CBL      (3,3)*1024  ( 19, 19,1024)  252');
    s.append(' 256 ~ 258: CBL      (1,1)*512   ( 19, 19, 512)  255');
    s.append(' 259      : MaxPool  (13,13)     ( 19, 19, 512)  258');
    s.append(' 260      : MaxPool  (9,9)       ( 19, 19, 512)  258');
    s.append(' 261      : MaxPool  (5,5)       ( 19, 19, 512)  258');
    s.append(' 262      : Concat               ( 19, 19,2048)  258,259,260,261');
    s.append(' 263 ~ 265: CBL      (1,1)*512   ( 19, 19, 512)  262');
    s.append(' 266 ~ 268: CBL      (3,3)*1024  ( 19, 19,1024)  265');
    s.append(' 269 ~ 271: CBL      (1,1)*512   ( 19, 19, 512)  268');
    s.append('------------------- PAN net (upsample 1) ---------------')
    s.append(' 272,+2,+2: CBL      (1,1)*256   ( 19, 19, 256)  271');
    s.append(' 273,+2,+2: CBL      (1,1)*256   ( 38, 38, 256)  204');
    s.append(' 278      : UpSample             ( 38, 38, 256)  276');
    s.append(' 279      : Concat               ( 38, 38, 512)  277,278');
    s.append(' 280 ~ 282: CBL      (1,1)*256   ( 38, 38, 256)  279');
    s.append(' 283 ~ 285: CBL      (3,3)*512   ( 38, 38, 512)  282');
    s.append(' 286 ~ 288: CBL      (1,1)*256   ( 38, 38, 256)  285');
    s.append(' 289 ~ 291: CBL      (3,3)*512   ( 38, 38, 512)  288');
    s.append(' 292 ~ 294: CBL      (1,1)*256   ( 38, 38, 256)  291');
    s.append('------------------- PAN net (upsample 2) ---------------')
    s.append(' 295,+2,+2: CBL      (1,1)*128   ( 38, 38, 128)  294');
    s.append(' 296,+2,+2: CBL      (1,1)*128   ( 76, 76, 128)  131');
    s.append(' 301      : UpSample             ( 76, 76, 128)  299');
    s.append(' 302      : Concat               ( 76, 76, 256)  300,301');
    s.append(' 303 ~ 305: CBL      (1,1)*128   ( 76, 76, 128)  302');
    s.append(' 306 ~ 308: CBL      (3,3)*256   ( 76, 76, 256)  305');
    s.append(' 309 ~ 311: CBL      (1,1)*128   ( 76, 76, 128)  308');
    s.append(' 312 ~ 314: CBL      (3,3)*256   ( 76, 76, 256)  311');
    s.append(' 315 ~ 317: CBL      (1,1)*128   ( 76, 76, 128)  314');
    s.append('------------------- PAN net (downsample 1) -------------')
    s.append(' 318      : Padding              ( 77, 77, 128)  317');
    s.append(' 319 ~ 321: CBL      (3,3)*256/2 ( 38, 38, 256)  318');
    s.append(' 322      : Concat               ( 38, 38, 512)  294,321');
    s.append(' 323 ~ 325: CBL      (1,1)*256   ( 38, 38, 256)  322');
    s.append(' 326 ~ 328: CBL      (3,3)*512   ( 38, 38, 512)  325');
    s.append(' 329 ~ 331: CBL      (1,1)*256   ( 38, 38, 256)  328');
    s.append(' 332 ~ 334: CBL      (3,3)*512   ( 38, 38, 512)  331');
    s.append(' 335 ~ 337: CBL      (1,1)*256   ( 38, 38, 256)  334');
    s.append('------------------- PAN net (downsample 2) -------------')
    s.append(' 338      : Padding              ( 39, 39, 256)  327');
    s.append(' 339 ~ 341: CBL      (3,3)*512/2 ( 19, 19, 512)  328');
    s.append(' 342      : Concat               ( 19, 19,1024)  314,341');
    s.append(' 343 ~ 345: CBL      (1,1)*512   ( 19, 19, 512)  342');
    s.append(' 346 ~ 348: CBL      (3,3)*1024  ( 19, 19,1024)  345');
    s.append(' 349 ~ 351: CBL      (1,1)*512   ( 19, 19, 512)  348');
    s.append(' 352 ~ 354: CBL      (3,3)*1024  ( 19, 19,1024)  351');
    s.append(' 355 ~ 357: CBL      (1,1)*512   ( 19, 19, 512)  354');
    s.append('------------------- Head -----------------------------')
    s.append(' 358,+3,+3: CBL      (3,3)*1024  ( 19, 19,1024)  357');
    s.append(' 359,+3,+3: CBL      (3,3)*512   ( 38, 38, 512)  337');
    s.append(' 360,+3,+3: CBL      (3,3)*256   ( 76, 76, 256)  317');
    s.append(' 367      : CBL      (1,1)*1024  ( 19, 19, 255)  364');
    s.append(' 368      : CBL      (1,1)*512   ( 38, 38, 255)  365');
    s.append(' 369      : CBL      (1,1)*256   ( 76, 76, 255)  366');
    s.append('------------------- Finish ---------------------------')
    s.append('Total parameters:         64,429,405')
    s.append('Trainable parameters:     64,363,101')
    s.append('Non-trainable parameters:     66,304')
    print_list(s)
    #-------------------------------------------    #--- Print string list
def print_list(slist):
    for k in range(len(slist)): 
      print(slist[k])
    #-------------------------------------------
        
#----------------------------------------------------    
    #-------------------------------------------


#---------------------------------------------------- Demo
    #-------------------------------------------    
def Demo_Mish(): 
    """ Demo behavior of Mish activation """
    x = np.arange(1000)/1000; x=2*x-1; x= x * 5;
    x = tf.cast(x, dtype= tf.float32)
    y = yolo4_network.Mish()(x)
    plt.plot(x,y); 
    plt.grid(color='b', linestyle=':', linewidth=0.5);
    plt.xlabel('Input value'); plt.ylabel('Output value')
    plt.title('Mish activator')
    plt.show();
    #-------------------------------------------    
def Demo_Network():
    Tools_List_Network()
    #-------------------------------------------    
def Demo_Random_Pick():
    """ Random pickup a image from VOC2012 database """
    img = yolo4_utils.random_pick()
    plt.imshow(img); plt.axis('off'); plt.show()
    #-------------------------------------------    
def Demo_Image_Align():
    """ Random pick image and align it as network input size """
    img  = yolo4_utils.random_pick()
    bimg = yolo4_utils.image_align(img)   
    plt.imshow(bimg[0]); plt.axis('off'); plt.show()   
    #-------------------------------------------    
def Demo_Index_Generator():
    """ Index_Generator generates index for batch training samples.
        User should define following parameters first:
            batch_size:     Size of batch 
            buf_size:       Size of shuffle buffer
            sample_size:    Size of total samples
            
        Operating principle:
            step 1: Fill shuffle buffer from sample 0 ~ M. M= buf_size
            step 2: Random shuffle
            step 3: Got N sample. N= batch_size * (13/4)
            step 4: Got consecutive N samples into buffer and repeat step 2
            step 5: If all samples are accessed, stop fill buffer.
            step 6: After buffer is empty, stop whole operation
    """ 
    #--- Method 1 ---
    IG  = yolo4_utils.Index_generator(batch_size     = 4,
                          buf_size       = 100,
                          sample_size    = 17125,);
    for index in IG: print(index)
    #--- Method 2 ---
    IG  = yolo4_utils.Index_generator();
    index = IG.take(1);  print(index)
    #.............................................. #---  
def Demo_Image_Processor():
    """ Image proccessor get image from path = img_path with specificed file index.
        Function: 
          .process_sample:   Process single sample with random cue, flip, and crop
          .augment(indexes): If indexes list contains only 1 sample. Use 1 sample only.
                             If indexes list contains 4 samples. Mosaic 4 samples as 1 image.
    """
    IG  = yolo4_utils.Index_generator(); index = IG.take(1);
    IP  = yolo4_utils.Image_processor(voc_list   = None,
                          img_size   = 416,
                          img_path   = '../../../Repository/VOC2012/JPEGImages/',
                          iou_th     = 0.5);
    simg, sbox, sconf = IP.process_sample(index[0]);
    IP.show_image_box(simg, sbox, title= 'Process 1 sample (random flip, crop, cue)')
    
    simg, sbox, sconf = IP.augment(index[0:4]);
    IP.show_image_box(simg, sbox, title= 'Mosaic 4 samples as 1 image')
    #.............................................. #---  
def Demo_Batch_Generator():
    """ Batch generator generate training samples from specified index list
        It generates following data:
           bimg:    (bs, 416, 416, 3)  Trainging image batch (416 is trainging input size)
           blabels: (bs, 10647, 85)    Training labels (10647 = 13*13*3*21)
           bboxes:  (bs, 100, 4)       Ground truth boxes in XYWH fromat (pixel)        
    """    
    BG  = yolo4_utils.Batch_generator();
    IG  = yolo4_utils.Index_generator(); 
    index = IG.take(1);
    bimg, blabels, bboxes = BG.convert(index);
    output = yolo4_decode.filter_box(blabels[:,:,:4], blabels[:,:,5:])
    classes= yolo4_utils.get_class('VOC')
    yolo4_utils.show_result(bimg[0], output, classes= classes)
    return bimg, blabels, bboxes
    #.............................................. #---  
#---------------------------------------------------- Main 
#Tool_List_VOC_BBOX_No();                            # List bounding box number for each class     
#Tool_Convert_Weights();                             # Convert original weights into Tensorflow format
#Demo_Mish()                                         # Demo Mish activation
#Demo_Network();                                     # Demo YOLO.v4 network
#Demo_Random_Pick();                                 # Random pick a image and show
#Demo_Image_Align();                                 # Demo image align function
#Demo_Index_Generator();                             # Demo Index generator    
#Demo_Image_Processor();                             # Demo Image processor
#Demo_Batch_Generator();                             # Demo Batch generator 
    


