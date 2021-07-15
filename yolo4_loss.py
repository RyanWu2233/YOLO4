# -*- coding: utf-8 -*-
"""
Module name: yolo4_loss
Author:      Ryan Wu
Date:        V0.1- 2020/11/22: Initial release
Description: YOLO 4 loss function
"""
#---------------------------------------------------- Import libraries
from __future__ import absolute_import, division, unicode_literals, print_function  
import tensorflow as tf
from yolo4_decode import decode_layer
from yolo4_utils  import get_anchors
#---------------------------------------------------- 

#---------------------------------------------------- 
def compute_loss(convs,                             # Head output (bs, 10647, 25)
                 x_off,
                 blabels,                           # Ground truth label (bs, 10647, 85)
                 bboxes,                            # Ground truth boxes (bs, 100, 4) (xywh)
                 prob_scale,                        # Scale of probability to solve imblance between class
                 img_size = 416,                    # Training input size
                 IoU_thN  = 0.4,                    # Lower IoU threshold for focal loss
                 **_kwargs):
    """ Input:
        - convs:        [conv_l, conv_m, conv_s]: network output
        - x_off:
        - blabels:      [bs, 10647, 85]:  Labels 
        - bboxes:       [bs, 3, 100, 4]:  Boxes of ground truth (xywh in pixel)
        - prob_scale:   [1,1,80]
        Return:
        - total_loss:   Loss value of YOLO.v4
    """ 
    """ Decode each layer prediction result """
    bs     = convs[0].shape[0];                     # Batch size
    anchors = get_anchors();    
    conv_l, conv_m, conv_s = convs;                 # l: large stride (bs,13,13,255)        
    pred_xywh_s, pred_conf_s, pred_prob_s = decode_layer(conv_s, anchors[0:3], 8 ,x_off)
    pred_xywh_m, pred_conf_m, pred_prob_m = decode_layer(conv_m, anchors[3:6], 16,x_off)
    pred_xywh_l, pred_conf_l, pred_prob_l = decode_layer(conv_l, anchors[6:9], 32,x_off)
    
    """ Reshape ConV2D output Tensor """
    conv_lc    = tf.reshape(conv_l, (bs, -1, 85));      # (bs,  507, 85) 
    conv_mc    = tf.reshape(conv_m, (bs, -1, 85));      # (bs, 2028, 85) 
    conv_sc    = tf.reshape(conv_s, (bs, -1, 85));      # (bs, 8112, 85) 
    convsc     = tf.concat([conv_sc, conv_mc, conv_lc], axis= 1)
    conv_xywh,  conv_conf,  conv_prob  = tf.split(convsc,  [4, 1, 80], axis=2)

    """ Compute probability loss """
    label_xywh, label_conf, label_prob = tf.split(blabels, [4, 1, 80], axis=2)
    N0         = tf.reduce_sum(label_conf)/bs;    
    prob_loss  = label_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels = label_prob, 
                                                                      logits = conv_prob);
    prob_loss  = prob_loss * prob_scale;            # Compensate class imbalance
     
    
    """ Compute CIoU loss """
    pred_boxes = tf.concat([pred_xywh_s,  pred_xywh_m,  pred_xywh_l],  axis = 1)
    label_xy, label_w, label_h = tf.split(label_xywh, [2, 1, 1], axis= 2) 
    w_IoU      = 2.0 - 1.0* (label_w/img_size) *( label_h/img_size); # (bs,10647)
    CIoU       = Compute_CIoU(pred_boxes, label_xywh);  # In pixel format (XYWH)
    CIoU_loss  = label_conf * w_IoU * (1- CIoU) 
    
    """ Background estimation """
    bbox_s, bbox_m, bbox_l = tf.split(bboxes, [1,1,1], axis = 1)   
    bbox_s     = tf.squeeze(bbox_s, axis =1);
    bbox_m     = tf.squeeze(bbox_m, axis =1);
    bbox_l     = tf.squeeze(bbox_l, axis =1);
    
    IoU_s      = Compute_IoU(pred_xywh_s, bbox_s);   # DIoU dimenshion (bs, 8112, 100, 1)
    IoU_m      = Compute_IoU(pred_xywh_m, bbox_m);   # DIoU dimenshion (bs, 2028, 100, 1)
    IoU_l      = Compute_IoU(pred_xywh_l, bbox_l);   # DIoU dimenshion (bs, 507, 100, 1)

    max_IoU_s  = tf.reduce_max(IoU_s, axis = 2);     # Max IoU at every grid (bs,8112,1)
    max_IoU_m  = tf.reduce_max(IoU_m, axis = 2);     # Max IoU at every grid (bs,2028,1)
    max_IoU_l  = tf.reduce_max(IoU_l, axis = 2);     # Max IoU at every grid (bs,507, 1)
    max_IoU    = tf.concat([max_IoU_s, max_IoU_m, max_IoU_l], axis= 1)
    label_bgd  = (1.0 - label_conf) * tf.cast( max_IoU < IoU_thN, tf.float32)

    """ focal loss estimation """
    pred_confs = tf.concat([pred_conf_s,  pred_conf_m,  pred_conf_l],  axis = 1)    
    w_conf     = tf.pow(label_conf - pred_confs, 2) # High for FP and FN    
        
    """ Compute confidence loss """
    CE_loss    = tf.nn.sigmoid_cross_entropy_with_logits(labels = label_conf, 
                                                         logits =  conv_conf);  
    #pos_loss   =                     label_conf * CE_loss;
    #neg_loss   =   (pred_confs**2) * label_bgd  * CE_loss;
    pos_loss   =  w_conf * label_conf * CE_loss;
    neg_loss   =  w_conf * label_bgd  * CE_loss;
    
    conf_loss  = (pos_loss + neg_loss);
    
    """ Take summation over each anchors and then averaged over whole batch """ 
    CIoU_loss  = tf.reduce_mean(tf.reduce_sum(CIoU_loss, axis= [1,2]))/ N0;
    conf_loss  = tf.reduce_mean(tf.reduce_sum(conf_loss, axis= [1,2]))/ N0; 
    prob_loss  = tf.reduce_mean(tf.reduce_sum(prob_loss, axis= [1,2]))/ N0;
    
    total_loss = CIoU_loss + conf_loss + prob_loss      
    return total_loss, CIoU_loss, conf_loss, prob_loss 
 
    #................................................ Compute IoU for background anchor detection
def Compute_IoU(box1, box2):
    """ Input:
        - box1:     (bs, 10647, 4) Predicted xywh (normalized scale)
        - box2:     (bs, M, 4)     Ground truth box xywh (normalized scale)
        Return:
        - iou:      (bs, 10647, M, 1) IOU score
    """
    eps = 1e-7;
    """ Box 1 """
    box1_xy, box1_wh = tf.split(box1,    [2,2], axis= -1); # (bs,10647,2)
    box1_xy = tf.expand_dims(box1_xy, axis= 2);
    box1_wh = tf.expand_dims(box1_wh, axis= 2);
    box1_w , box1_h  = tf.split(box1_wh, [1,1], axis= -1); # (bs,10647,1)
    box1_area = box1_w * box1_h;                    # Predict  box area 
    box1_lt   = box1_xy - box1_wh*0.5;              # Left-top of box 1
    box1_rd   = box1_xy + box1_wh*0.5;              # Right-down of box 1
    """ Box 2 """        
    box2_xy, box2_wh = tf.split(box2,    [2,2], axis= -1); #
    box2_xy = tf.expand_dims(box2_xy, axis= 1);
    box2_wh = tf.expand_dims(box2_wh, axis= 1);
    box2_w , box2_h  = tf.split(box2_wh, [1,1], axis= -1); # (bs,10647,1)
    box2_area = box2_w * box2_h;                    # Labelled box area    
    box2_lt   = box2_xy - box2_wh*0.5;              # Left-top of box 2
    box2_rd   = box2_xy + box2_wh*0.5;              # Right-down of box 2
    """ Intersection area """    
    inter_lt  = tf.maximum(box1_lt, box2_lt);       # Intersection area
    inter_rd  = tf.minimum(box1_rd, box2_rd);       # (bs,10647,2)
    inter_wh  = tf.maximum(inter_rd-inter_lt, 0.0); # 
    inter_area= inter_wh[:,:,:,0] * inter_wh[:,:,:,1]   # Area = width * height
    inter_area= tf.expand_dims(inter_area, axis=-1)
    union_area= box1_area + box2_area - inter_area; # Union area
    IoU       = tf.math.divide(inter_area, union_area + eps)
    return IoU
    
    #................................................ 
def Compute_CIoU(box1, box2):
    """ Input:
        - box1:     (bs, 10647, 4) Predicted xywh (normalized scale)
        - box2:     (bs, 10647, 4) Labelled xywh (normalized scale)
        Return:
        - ciou:     (bs, 10647, 1) CIOU score
    """ 
    eps = 1e-9;
    """ Box 1 """
    box1_xy, box1_wh = tf.split(box1,    [2,2], axis= -1); # (bs,10647,2)
    box1_w , box1_h  = tf.split(box1_wh, [1,1], axis= -1); # (bs,10647,1)
    box1_area = box1_w * box1_h;                    # Predict  box area 
    box1_lt   = box1_xy - box1_wh*0.5;              # Left-top of box 1
    box1_rd   = box1_xy + box1_wh*0.5;              # Right-down of box 1
    """ Box 2 """                
    box2_xy, box2_wh = tf.split(box2,    [2,2], axis= -1); #
    box2_w , box2_h  = tf.split(box2_wh, [1,1], axis= -1); # (bs,10647,1)
    box2_area = box2_w * box2_h;                    # Labelled box area    
    box2_lt   = box2_xy - box2_wh*0.5;              # Left-top of box 2
    box2_rd   = box2_xy + box2_wh*0.5;              # Right-down of box 2
    """ Intersection area """    
    inter_lt  = tf.maximum(box1_lt, box2_lt);       # Intersection area
    inter_rd  = tf.minimum(box1_rd, box2_rd);       # (bs,10647,2)
    inter_wh  = tf.maximum(inter_rd-inter_lt, 0.0); # 
    inter_area= inter_wh[:,:,0] * inter_wh[:,:,1]   # Area = width * height
    inter_area= tf.expand_dims(inter_area, axis=-1)
    union_area= box1_area + box2_area - inter_area; # Union area
    IoU       = tf.math.divide(inter_area, union_area + eps)
    """ DIoU estimation """   
    enc_lt    = tf.minimum(box1_lt, box2_lt);       # Enclosure area
    enc_rd    = tf.maximum(box1_rd, box2_rd);       #
    enc_wh    = enc_rd  - enc_lt;
    enc_diag  = enc_wh[ :,:,0]**2 + enc_wh[ :,:,1]**2; # Outlier box diagonal square    
    cent_wh   = box2_xy - box1_xy; 
    cent_diag = cent_wh[:,:,0]**2 + cent_wh[:,:,1]**2; # Center-to-center diagonal square
    dist_ratio= tf.expand_dims(tf.math.divide(cent_diag, enc_diag + eps), axis= -1)
    DIoU      = IoU - dist_ratio;
    """ CIoU estimation (0.40528 = 4/pi/pi) """
    box1_w_h  = tf.math.divide(box1_w, box1_h + eps);   # =atan(w/h)
    box2_w_h  = tf.math.divide(box2_w, box2_h + eps);   # 1.621 = 4/pi/pi
    v         = 0.40528* (tf.math.atan(box1_w_h) - tf.math.atan(box2_w_h))**2
    alpha     = tf.math.divide(v, 1-IoU + v + eps)
    CIoU      = DIoU - alpha * v; 
    return CIoU  

 
#----------------------------------------------------   


