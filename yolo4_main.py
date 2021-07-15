# -*- coding: utf-8 -*-
"""
Module name: yolo4_main
Author:      Ryan Wu
Date:        V0.1- 2020/12/01: Initial release
Description: Main class for YOLO.v4
"""
#---------------------------------------------------- Import libraries
from __future__ import absolute_import, division, unicode_literals, print_function
from yolo4_network import YOLO4_model
import yolo4_decode, yolo4_utils, yolo4_loss

import tensorflow as tf 
import numpy as np
import time 
import matplotlib.pyplot as plt

"""
"""
#---------------------------------------------------- YOLO V4 main
class YOLO_V4():                                    #
    def __init__(self,
                 img_size   = 416, 
                 dataset    = 'COCO',
                 **kwargs):
      self.model      = None;                       # YOLO4 model
      self.anchors    = yolo4_utils.get_anchors();  # Anchors definition
      self.img_size   = img_size;                   # Detection of training image size
      self.dataset    = dataset;                    # Training dataset
      self.conf_thresh= 0.15                        # Inference confidence threshold
      self.classes    = yolo4_utils.get_class(dataset); # Classes of object
      self.x_off      = 1;                          # Original Darknet has 1 point bias
      
    #................................................  
    def evaluate(self, count=10, thresh = None):  
      if self.dataset == 'COCO': self.x_off = 1;
      if self.dataset == 'VOC':  self.x_off = 0;
      if thresh is not None:                        # Detection threshold
          self.conf_thresh = thresh
      if self.model is None:  
        self.model = YOLO4_model(img_size = self.img_size)
        model_name = 'yolo4_weight_' + self.dataset + '.h5'  
        self.model.load_weights(model_name)
      """ """  
      for k in range(count):
        img, index = yolo4_utils.pick_VOC_image(index= k);
        bimg  = yolo4_utils.image_align(img, img_size = self.img_size)
        convs = self.model(bimg); 
        pred_boxes, pred_scores, pred_confs = yolo4_decode.decode_all(convs, self.x_off ) 
        output = yolo4_decode.filter_box(pred_boxes, pred_scores, pred_confs, self.conf_thresh)      
        print(f':: Image index= {k}')
        yolo4_utils.show_result(bimg, output, classes = self.classes, title = str(index)  )  
        
    #................................................  
    def interference(self, 
                     img    = None,                 # Image to be predict
                     ret    = False,                # Return value or not
                     thresh = None):                # Detection confidence threshold
      """ Interference result """
      if self.dataset == 'COCO': self.x_off = 1;
      if self.dataset == 'VOC':  self.x_off = 0;
      if img is None:                               # Random choose image if not define
          img, index = yolo4_utils.pick_VOC_image();
      if thresh is not None:                        # Detection threshold
          self.conf_thresh = thresh
      if self.model is None:  
        self.model = YOLO4_model(img_size = self.img_size)
        model_name = 'yolo4_weight_' + self.dataset + '.h5'  
        self.model.load_weights(model_name)
      """ """  
      bimg  = yolo4_utils.image_align(img, img_size = self.img_size)
      convs = self.model(bimg); 
      pred_boxes, pred_scores, pred_confs = yolo4_decode.decode_all(convs, self.x_off ) 
      output = yolo4_decode.filter_box(pred_boxes, pred_scores, pred_confs, self.conf_thresh)      
      yolo4_utils.show_result(bimg, output, classes = self.classes, title = str(index)  )  
      if ret == True:
        return output, convs, bimg[0]
    #................................................
    def train(self,
              restart    = True,                    # Training start from scratch
              dataset    = 'VOC',                   # Train image dataset
              batch_size = 32,                      # Training batch size
              train_path = '../../../Repository/VOC2012/JPEGImages/',
              **kwargs):
      """ Train model """
      self.dataset    = dataset;                    # Training dataset
      self.train_path = train_path;                 # File path point to VOC2012 train
      self.batch_size = batch_size;                 # Training batch size
      self.num_epochs = 200;                        # Number of training epoches
      self.IoU_thP    = 0.5;                        # IoU threshold for positive samples 
      self.IoU_thN    = 0.4;                        # IoU threshold for negative samples 
      self.IoU_crop   = 0.1;                        # IoU threshold for augment crop filtering
      self.lr_init    = 1e-4*3;                     # Initial learning rate
      self.prob_scale = self.get_prob_scale();      # Scale of each class 
      
      """ Pre-process before training """
      self._setup_network(restart = restart);       # Initialize network 
      BG  = yolo4_utils.Batch_generator(self.img_size, 
                                        self.IoU_thP,
                                        self.IoU_crop); # Build batch generator
      print('Begin training')
      """ Train loop """
      while self.epoc < self.num_epochs:
        start= time.time()                          # Epoch training start time
        """ learning rate control """
        lr_new = self.next_learning_rate(self.epoc, 
                                         self.num_epochs );  # Implement cosine annealing         
        self.lr_curve[self.epoc] = lr_new;          # Record learning rate
        self.optimizer.lr.assign(lr_new)            # Modify optimizer learning rate
        """ get dataset """
        IG  = yolo4_utils.Index_generator(batch_size= self.batch_size);  # Index generator
        beta= (self.batch_size*13/4) / IG.sample_size; # = (32*13/4)/ 17124 = 1/164.7
        mean_loss = 0;
        m = 0;                                      # Temp
        for index in IG:          
          bimg, blabels, bboxes = BG.convert(index); # Generate training images, and labels
          print('.', end='');                       # Show progress for each batch             
          loss, l1, l2, l3 = self._train_step(bimg, blabels, bboxes); 
          
          
          """ Loss value monitor """
          mean_loss = mean_loss + beta* loss.numpy();
          if str(loss.numpy()) == 'nan': print('Nan happens');  return # Nan Check           
          m= m + 1;  
          if np.mod(m,10)== 0:   
              print(loss.numpy(), l1.numpy(), l2.numpy(), l3.numpy()) # Temp solution  
        """ show simulation progress """  
        duration = time.time() - start;             # Simulation time for 1 epoch
        lr0 = self.lr_curve[self.epoc];
        print(' ')        
        print(f' Epoch {self.epoc} = {duration} sec; loss = {mean_loss}; lr = {lr0}') 
        if np.mod(self.epoc, 5)==0 or self.epoc < 10: 
          self.interference(thresh=0.2);            # Show inference result 
        """ Save weights """    
        self.loss_curve[self.epoc] = mean_loss;     # Record loss curve
        self.epoc +=1;                              # Increase epochs
        if np.mod(self.epoc, 2)==0:                 # Record weights
          self.save_weights();
        if np.mod(self.epoc, 20)==0:                # Record weights
          self.save_weights(self.epoc);
    #................................................ For VOC2012 only
    def get_prob_scale(self):                       #        
      box_no     = yolo4_utils.get_stat_VOC();
      box_total  = np.sum(box_no);
      prob_scale = np.ones((1,1,80))*1e-3;
      prob_scale[0,0,:20] = 1./box_no               # Normalized to box count
      prob_scale = prob_scale * box_total/ 100;
      prob_scale = tf.cast(prob_scale, dtype= tf.float32)
      return prob_scale
    #................................................
    def _setup_network(self,
                       restart = True,
                       **kwargs):  
      """ Private function: setup network """  
      self.optimizer  = tf.keras.optimizers.Adam(self.lr_init) # Optimizer = Adam
      self.classes    = yolo4_utils.get_class(self.dataset);   # Classes of object
      self.epoc       = 0;                          # How many epochs trained
      self.loss_curve = np.zeros((self.num_epochs,)) # History curve for mean loss
      self.lr_curve   = np.zeros((self.num_epochs,)) # History curve for learning rate
      """ Create or load model and weights"""
      if self.model is None:                        # Create blank model if not exist
        self.model = YOLO4_model(img_size = self.img_size);
      if restart == True:                           # Train from scratch:
        model_name = 'yolo4_weight.h5'              # Load original darknet weights as begining
        self.model.load_weights(model_name)         # to realize transfer learning
      else:
        model_name = 'yolo4_weight_' + self.dataset + '.h5'  
        self.model.load_weights(model_name)         # Continue training
        A = np.load('yolo4_train_epoch.npy' , allow_pickle = True)
        self.epoc       = A[0];
        self.loss_curve = A[1];
        self.lr_curve   = A[2];
        print('Continue training')
        #self.epoc= 199;
      """ Freeze backbone """  
      self.x_off = 0;                               # 
      #for layer_no in range(358):                   # Freeze layers (darknet + PANet)
      for layer_no in range(249):                   # Freeze layers (darknet)
        self.model.layers[layer_no].trainable=False # Only training heads       
    #................................................
    def next_learning_rate(self, current, total,
                           warmup   = 4,            # Warmup epochs
                           segment  = 1):           # Restart for 3 times
        """ Implement cosine annealing learning """
        init       = self.lr_init;                  # Initial scale
        period     = total // segment;
        cos_anneal = np.cos(1.5/ period *np.mod(current, period))
        amplitude  = np.exp(- current/ total *3);   # Envelope decay from 1.00 to 0.05
        warmupx    = np.minimum(warmup, current+1)/ warmup;
        lr_new     = init* amplitude* cos_anneal *warmupx;
        return lr_new    
    #................................................
    @tf.function  
    def _train_step(self, bimg, blabels, bboxes):
      """ Compute gradient """
      with tf.GradientTape() as tape:
          convs = self.model(bimg);  
          loss, l1, l2, l3 = yolo4_loss.compute_loss(convs, 
                                                     self.x_off,
                                                     blabels,
                                                     bboxes,
                                                     self.prob_scale,
                                                     self.img_size,
                                                     self.IoU_thN);
           
      gradients = tape.gradient(loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
      return loss, l1, l2, l3 
    #................................................
    def save_weights(self, epochs= None):
      model_name = 'yolo4_weight_' + self.dataset
      if epochs is not None:    
        model_name = model_name + '_'+ str(epochs)
      model_name = model_name + '.h5'
      self.model.save(model_name)                    # Save model weights
      A = [self.epoc, self.loss_curve, self.lr_curve]
      np.save('yolo4_train_epoch.npy' , A)  
    #................................................
    def show_curve(self):
      No= int(np.sum(self.loss_curve > 0));
      plt.subplot(121); plt.plot(self.lr_curve[:No]*10000);
      plt.xlabel('Epochs'); plt.ylabel('Learning rate (1e-4)');
      plt.grid(color='b', linestyle=':', linewidth=0.5);

      plt.subplot(122); plt.plot(self.loss_curve[:No]);
      plt.xlabel('Epochs'); plt.ylabel('Loss');
      plt.grid(color='b', linestyle=':', linewidth=0.5);      
      plt.show();
    #................................................
 

#---------------------------------------------------- 

