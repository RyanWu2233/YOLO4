# -*- coding: utf-8 -*-
"""
Module name: yolo4_decode
Author:      Ryan Wu
Date:        V0.1- 2020/11/08: Initial release
Description: Decode network output
"""


#---------------------------------------------------- Import libraries
from __future__ import absolute_import, division, unicode_literals, print_function
import tensorflow as tf 
import numpy as np 
import yolo4_utils  
#---------------------------------------------------- Decoder

    #-------------------------------------------    # Decode 1 layer
def decode_layer(conv, anchors, stride, x_off= 0):
    """ Decode YOLO.v4 head output (conv) into prediction
        Input:  (assume image size = 618, COCO dataset)
        - conv:     Conv2D (head) otput feature map (bs,19,19,255) 
        - anchors:  YOLO.v4 anchors [3,2] 
        - stride:    8 for small  size object (bs,76,76,255)
                    16 for medium size object (bs,38,38,255)
                    32 for large  size object (bs,19,19,255)
        Return:           
        - pred_xywh: [bs,N,4]  Predicted bounding box size (x,y,w,h) in pixel
        - pred_conf: [bs,N]    Predicted bounding box confidence
        - pred_prob: [bs,N,80] Predicted bounding box probability
        N = 19*19*3 for large size object layer
    """
    """ Reshape (bs,19,19,255) --> (bs,19,19,3,85) """
    conv_shape       = tf.shape(conv)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]
    anchor_per_scale = len(anchors) 
    num_class        = (conv_shape[3]//3) - 5;    
    conv = tf.reshape(conv, (batch_size, 
                             output_size, 
                             output_size, 
                             anchor_per_scale, 
                             5 + num_class))
    """ decompose 85 columns into xy(2), wh(2), confidence(1), probability (80) """
    conv_raw_dxdy = conv[:, :, :, :, 0:2]           # [bs,19,19,3,2]
    conv_raw_dwdh = conv[:, :, :, :, 2:4]           # [bs,19,19,3,2]
    conv_raw_conf = conv[:, :, :, :, 4:5]           # [bs,19,19,3,1]
    conv_raw_prob = conv[:, :, :, :, 5: ]           # [bs,19,19,3,80]
    """ Generate XY mesh grid (bs,19,19,3,2)
        xy_grid[0,:,:,0,0] = [[0,1,2, ...],
                              [0,1,2, ...],
                              [0,1,2, ...]]
        xy_grid[0,:,:,0,1] = [[0,0,0, ...],
                              [1,1,1, ...],
                              [2,1,2, ...]]
        One strange thing is that original weights from YOLO.v4 has 1 offset on x_grid
    """
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    x = x+ x_off                                    # Don't know why?    
    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)
    """ Convert feature map into prediction """
    pred_xy = (tf.sigmoid(conv_raw_dxdy)*1.2-0.1 + xy_grid) * stride
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors)    
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)
    """ change tensor dimension """
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))  # [bs, 19*19*3, 4]
    pred_conf = tf.reshape(pred_conf, (batch_size, -1, 1))  # [bs, 19*19*3, 1]
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, num_class))  # [bs, 19*19*3, 80]
    return pred_xywh, pred_conf, pred_prob
    #-------------------------------------------    # Decode network output as boxes 
def decode_all(convs, x_off):
    """ conv:   [13*13,  26*26,  52*52]
        ReturnL
        - pred_xywh: [bs,N,4]  Predicted bounding box size (x,y,w,h) in pixel
        - pred_conf: [bs,N]    Predicted bounding box confidence
        - pred_prob: [bs,N,80] Predicted bounding box probability
        N = 19*19*3*(1 + 4 + 16)  = 22743
    """
    conv_l, conv_m, conv_s = convs;                 # l: large stride (bs,13,13,255)     
    anchors = yolo4_utils.get_anchors();
    """ Decode each layer """
    pred_xywh_s, pred_conf_s, pred_prob_s = decode_layer(conv_s, anchors[0:3], 8 ,x_off)
    pred_xywh_m, pred_conf_m, pred_prob_m = decode_layer(conv_m, anchors[3:6], 16,x_off)
    pred_xywh_l, pred_conf_l, pred_prob_l = decode_layer(conv_l, anchors[6:9], 32,x_off)
    """ compute score """
    pred_score_s = pred_conf_s * pred_prob_s
    pred_score_m = pred_conf_m * pred_prob_m
    pred_score_l = pred_conf_l * pred_prob_l
    """ Concatane """
    pred_boxes = tf.concat([pred_xywh_s,  pred_xywh_m,  pred_xywh_l],  axis = 1)
    pred_scores= tf.concat([pred_score_s, pred_score_m, pred_score_l], axis = 1)
    pred_confs = tf.concat([pred_conf_s,  pred_conf_m,  pred_conf_l],  axis = 1)
    return pred_boxes, pred_scores, pred_confs 
#---------------------------------------------------- Compute IoU
def compute_IoU(box1, box2):
    """ Input:
        - box1:(4,)     Reference box
        - box2:(k,4)    Box array to be computed IoU
        Return:
        - IoU:          IoU scores
    """
    box1_a = box1[2]  * box1[3];                    # Reference box area
    box1_lt= box1[:2] - box1[2:]*0.5;               # Reference box left, top
    box1_rb= box1[:2] + box1[2:]*0.5;               # Reference box right, bottom
    box2_a = box2[:,2]  * box2[:,3];                # Box array area
    box2_lt= box2[:,:2] - box2[:,2:]*0.5;           # Box array left, top
    box2_rb= box2[:,:2] + box2[:,2:]*0.5;           # Box array right, bottom
    boxi_lt= np.maximum(box1_lt, box2_lt);          # Intersection box left,  top
    boxi_rb= np.minimum(box1_rb, box2_rb);          # Intersection box right, bottom
    boxi_wh= np.maximum(0,(boxi_rb - boxi_lt))      # Intersection box width, height
    boxi_a = boxi_wh[:,1] * boxi_wh[:,0];           # Intersection box area
    boxu_a = box1_a + box2_a -boxi_a;               # Union area = A1 + A2 - AI
    IoU    = boxi_a/ (boxu_a + 1e-9)                # IoU
    return IoU    
#---------------------------------------------------- Filter 
def filter_box(pred_boxes, 
               pred_scores, 
               pred_confs,
               conf_thresh  = 0.15, 
               nms_thresh   = 0.45, 
               keep_top_k   = 100, 
               nms_top_k    = 100):    
    """ Input:
        - pred_boxes:   (1,N,3):  Prediction box location (XYWH)
        - pred_scores:  (1,N,80): Scores for each box-location 
        Return:
        - [kboxes:      (M,4):    Box size (x,y,w,h) in pixel format
           kclass:      (M,):     Predict class number
           kscores:     (M,):     Predict robability  
           kprobs:]     (M,):     Index of probability
    """    
    #    N = 10647 = 3*13*13*(1+4+16) (@ img_size= 416)
    #    N = 22742 = 3*19*19*(1+4+16) (@ img_size= 608)
    n_classes = tf.shape(pred_scores)[2];
    pred_scores = pred_scores.numpy(); 
    kboxes = [];                                    # Filtered bounding box (X,Y,W,H) in pixel
    kclass = [];                                    # Corresponding class number
    kscores= [];                                    # Corresponding score = confidence * probability
    kindex = [];                                    # Corresponding box index
    kprobs = [];                                    # Corresponding probability
    #--------
    for clas in range(n_classes):
      cidx   = tf.where(pred_scores[0, :, clas] > conf_thresh)[:,0];
      if len(cidx)>0:
        cboxes = tf.gather(pred_boxes[0], cidx).numpy();        # (k, 4) Candidate boxes  
        cscores= tf.gather(pred_scores[0,:,clas], cidx).numpy() # (k,) Candidate scores
        """ Sort scores from high to low """
        sidx   = np.argsort(-cscores)               # Sort index for scores (in descending way)         
        sboxes = cboxes[ sidx,:]                    # Box after sorting
        sscores= cscores[sidx]                      # Scores after sorting
        valid  = np.ones((len(cidx),))              # 1: To be processed
        """ NMS """
        while np.sum(valid) > 0:                    # Check NMS for every valid box
          idx_cur = np.where(valid>0)[0][0];        # Get valid box with highest score
          IoU  = compute_IoU(sboxes[idx_cur], sboxes[idx_cur:])
          mask = np.where(IoU > nms_thresh)[0];     # Find high IoU boxes relative to current box
          valid[idx_cur + mask] = 0                 # Remove high IoU boxes
          #--- Record filtered result ---
          findex = cidx[sidx[ idx_cur]];            # Filtered box index
          fboxes = pred_boxes[ 0,findex,:]          # Filtered box XYWH
          fscore = pred_scores[0,findex,clas];      # Filtered box score
          fconf  = pred_confs[ 0,findex,0];         # Filtered box confidence
          fprob  = fscore / fconf;                  # Filtered box confidence
          #---- ----          
          kboxes.append(fboxes)
          kclass.append(clas)
          kscores.append(fscore)
          kindex.append(findex)          
          kprobs.append(fprob)
    #--- ---
    kboxes = np.array(kboxes)
    kclass = np.array(kclass)
    kscores= np.array(kscores) 
    kindex = np.array(kindex)
    kprobs = np.array(kprobs)
    return kboxes, kclass, kscores, kprobs

#====================================================  







