# -*- coding: utf-8 -*-
"""
Module name: yolo4_train
Author:      Ryan Wu
Date:        V0.1- 2020/11/11: Initial release
Description: Use transfer learning to train new model
             Original model is designed for MS-COCO
"""
#---------------------------------------------------- Import libraries
from __future__ import absolute_import, division, unicode_literals, print_function
import tensorflow as tf
import numpy as np 
import time, os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorsys   
#=================================================== Basic
    #-------------------------------------------    #--- Get stat for VOC
def get_stat_VOC():
    """ Compute box number for each class """
    voc_list = np.load('db_voc.npy',allow_pickle= True)
    box_no   = np.zeros((20,))
    for k in range(len(voc_list)):
      vlabel = voc_list[k, 3];
      for m in range(len(vlabel)):
        index = int(vlabel[m])  
        box_no[index] += 1;  
    return box_no    
    #-------------------------------------------    #--- Get anchors of YOLO.v4  
def get_anchors():
    anc= [[ 12.,  16.], [ 19.,  36.],  [ 40.,  28.],  
          [ 36.,  75.], [ 76.,  55.],  [ 72., 146.],
          [142., 110.], [192., 243.],  [459., 401.]]
    return np.array(anc)
    #............................................   #--- Get class definition of Coco
def get_class(db_name='COCO'):
    if db_name == 'COCO':
      classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
               'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'sofa', 'pottedplant', 'bed',
               'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    if db_name == 'VOC':
      classes=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable','dog', 'horse','motorbike', 'person',
               'pottedplant','sheep', 'sofa', 'train', 'tvmonitor',
               '20','21','22','23','24','25','26','27','28','29',
               '30','31','32','33','34','35','36','37','38','39',
               '40','41','42','43','44','45','46','47','48','49',
               '50','51','52','53','54','55','56','57','58','59',
               '60','61','62','63','64','65','66','67','68','69',
               '70','71','72','73','74','75','76','77','78','79'];      
    return classes
    #-------------------------------------------    #--- Random pick image
def pick_VOC_image(path= None, index= None):
    if path is None:  path = '../../../Repository/VOC2012/JPEGImages/'
    file_list = os.listdir(path)
    if index is None:
      index   = np.random.randint(0, len(file_list)-1)
    fn        = path + file_list[index];
    img       = plt.imread(fn)
    return img, index
    #-------------------------------------------    #--- Image align
def image_align(img, img_size= 416, expand_dim= True):
    img_h, img_w = img.shape[:2];
    longer = np.maximum(img_w, img_h);
    bimg   = np.zeros((longer, longer, 3))
    x0     = (longer- img_w)//2;  x1= x0 + img_w;
    y0     = (longer- img_h)//2;  y1= y0 + img_h;
    bimg[y0:y1, x0:x1, :] = img;
    bimg   = tf.image.resize(bimg/256, (img_size,img_size))
    bimg   = tf.cast(bimg, dtype= tf.float32)
    if expand_dim == True:
      bimg = tf.expand_dims(bimg, axis= 0)  
    return bimg 
    #............................................   #--- xywh -> [xmin,ymin,xmax,ymax]
def xywh_to_ltrb(xywh):
    """ Input:
        - xywh: (N,4) [Xc, Yc, Width, Height]
        Return:
        - ltrb: (N,4) [Left, Top, Right, Bottom]
    """
    shape = xywh.shape;                             # Get shape and restore it
    xywhs = np.reshape(xywh, (-1,4))
    ltrb  = np.zeros_like(xywhs);
    ltrb[:,0] = xywhs[:,0] - 0.5 * xywhs[:,2];
    ltrb[:,1] = xywhs[:,1] - 0.5 * xywhs[:,3];
    ltrb[:,2] = xywhs[:,0] + 0.5 * xywhs[:,2];
    ltrb[:,3] = xywhs[:,1] + 0.5 * xywhs[:,3];
    ltrb  = np.reshape(ltrb, shape)
    return ltrb
    #............................................   #--- [xmin,ymin,xmax,ymax] -> xywh
def ltrb_to_xywh(ltrb):
    """ Input:
        - ltrb: (N,4) [Left, Top, Right, Bottom]
        Return:
        - xywh: (N,4) [Xc, Yc, Width, Height]
    """    
    shape = ltrb.shape;                             # Get shape of ltrb and reserve it
    ltrbs = np.reshape(ltrb, (-1,4))
    xywh  = np.zeros_like(ltrbs);
    xywh[:,0] = (ltrbs[:,0] + ltrbs[:,2])/2;
    xywh[:,1] = (ltrbs[:,1] + ltrbs[:,3])/2;
    xywh[:,2] = (ltrbs[:,2] - ltrbs[:,0]);
    xywh[:,3] = (ltrbs[:,3] - ltrbs[:,1]);
    xywh  = np.reshape(xywh, shape) 
    return xywh
    #............................................   #--- IOU estimation
def estimate_IOU(boxs, rbox):
    """ Input:
        - boxs:  (N,4) box array to be estimated
        - rbox:  (4,)  reference box
        Return:
        - IOU:   IOU value
    """
    shape   = boxs.shape[0:-1];                     # restore shape and remove last dimension
    boxt    = np.reshape(boxs, (-1,4)) 
    i_xmin  = np.maximum(boxt[:,0], rbox[0]);
    i_ymin  = np.maximum(boxt[:,1], rbox[1]);
    i_xmax  = np.minimum(boxt[:,2], rbox[2]);
    i_ymax  = np.minimum(boxt[:,3], rbox[3]);
    i_width = np.maximum((i_xmax - i_xmin), 0);     # width  of intersection
    i_height= np.maximum((i_ymax - i_ymin), 0);     # height of intersection
    i_area  = i_width * i_height;
    r_area  = (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]);
    b_area  = (boxt[:,2] - boxt[:,0]) * (boxt[:,3] - boxt[:,1]);
    u_area  = b_area + r_area - i_area;
    IOU     = i_area / u_area
    IOU     = np.reshape(IOU, shape)
    return IOU

#=================================================== Dataset (index_generator)
class Index_generator(object):
    """ Parameters: 
        - buf_size:     Size of shuffle buffer
        - batch_size:   Size of training batch
        - sample_size:  Size of total accessible images

        Attributes:            
        - group_size:   Size of group (=Batch/4*13) 
        - sample_no:    How many image samples are accessed
        - buffer:       Shuffle buffer, store number for next access     
        - group:        Image index prepared for processing
        
        Functions (public):
          take(count):                              # Return N group indexes

        Functions (private):
          __iter__();
          __next__();
          _reset_buffer();                          # Clear buffer and reset sample_no
          _generate();                              # Generate 1 group
    """    
    def __init__(self, 
                 batch_size     = 4,
                 buf_size       = 100,
                 sample_size    = 17125,
                 **_kwargs):
      self.batch_size   = batch_size;               # Size of training batch
      self.group_size   = int(batch_size/4*13);     # Size of image group
      self.sample_size  = sample_size;              # Size of total accessible images   
      self.buf_size     = np.maximum(buf_size, self.group_size+50);  # Size of shuffle buffer
      
      self._reset_buffer();                         # Reset buffer
    #.............................................. #--- Iteration
    def __iter__(self):
        return self
    #.............................................. #---  Next
    def __next__(self):
      return self._generate(); 
    #.............................................. #--- Reset buffer
    def _reset_buffer(self):
      self.buffer = np.arange(self.buf_size);       # Reload image index
      self.sample_no = self.buf_size;        
    #---------------------------------------------- #--- Take one group   
    def take(self, count):
      if count == 1: return self._generate();  
      groups = [];
      for k in range(count):
        groups.append(self._generate())  
      return groups
    #---------------------------------------------- #--- Take one group   
    def _generate(self):
      with tf.device("/cpu:0"):
        if len(self.buffer) > self.group_size: 
          np.random.shuffle(self.buffer);               # Randomize index
          group = self.buffer[0: self.group_size].copy() # Get group index
          if self.sample_no + self.group_size < self.sample_size:
            self.buffer[0: self.group_size]= self.sample_no + np.arange(self.group_size)
            self.sample_no += self.group_size;
          else:
            self.buffer = self.buffer[self.group_size:]  
          return group  
        else:
          self._reset_buffer();  
          raise StopIteration 
    #..............................................  
          
#=================================================== Dataset (Single image processor)
class Image_processor(object):
    """ Parameters:
        - voc_list:     VOC2012 annotation database
        - img_size:     Final output image size
        - img_path:     Training image path
        - IoU_crop:     IoU threshold for filtering low confidence box
        
        Functions (public):
          simg, sbox, sconf = augment(index)            # Generate 1 image  
          simg, sbox, sconf = process_sample(index)     # Process 1 sample

        Functions (private):
          _random_cue():                            # Random change color
          _random_fliplr():                         # Random flip image and box by left-right
          _random_crop():                           # Random crop image and modify box, conf
    """
    def __init__(self,
                 voc_list   = None,
                 img_size   = 416,
                 img_path   = '../../../Repository/VOC2012/JPEGImages/',
                 IoU_crop   = 0.1,
                 **_kwargs):
      if voc_list is None: voc_list = np.load('db_voc.npy',allow_pickle= True)
      self.voc_list = voc_list;                     # VOC 2012 annotation database
      self.img_size = img_size;                     # Final output image size (pixel)
      self.img_path = img_path;                     # Training image path
      self.IoU_crop = IoU_crop;                       # IOU threshold for box filtering
    #.............................................. #--- Generate one image  
    def augment(self, indexes):
      if len(indexes)== 1:                          #-- Single image, no mosaic 
        simg, sbox, sconf = self.process_sample(indexes[0])
      else:                                         #-- 4 images mosaic
        simg = np.zeros((self.img_size, self.img_size, 3))
        sbox = None;
        sconf= None;
        tsize= np.array((1,1))                      # image size of each subplot
        ry = 0.3 + np.random.rand(1)*0.4;           # Subplot ratio along Y axis (0.3 ~ 0.7)
        ty = int(self.img_size * ry);               #
        for ky in range(2):  
          tsize[0]= ty* (1-ky) + (self.img_size - ty)*ky; 
          rx = 0.3 + np.random.rand(1)*0.4;         # Subplot ratio along Y axis (0.3 ~ 0.7)
          tx = int(self.img_size * rx);             #              
          for kx in range(2): 
            tsize[1]= tx* (1-kx) + (self.img_size - tx)*kx;
            timg, tbox, tconf = self.process_sample(indexes[ky*2+kx],tsize)             
            simg[ty*ky: ty*ky+ tsize[0], tx*kx: tx*kx+tsize[1], :] = timg; 
            if len(tbox)>0: 
              tbox[:,0]= tbox[:,0]* (tsize[1]/self.img_size) + kx* rx
              tbox[:,1]= tbox[:,1]* (tsize[0]/self.img_size) + ky* ry
              tbox[:,2]= tbox[:,2]* (tsize[1]/self.img_size) + kx* rx
              tbox[:,3]= tbox[:,3]* (tsize[0]/self.img_size) + ky* ry
              if sbox is None:                  
                sbox = tbox; sconf = tconf;
              else:
                sbox = np.concatenate(( sbox, tbox), axis= 0);
                sconf= np.concatenate((sconf,tconf), axis= 0);
      return simg, sbox, sconf     
    #.............................................. #--- Augment one sample  
    def process_sample(self, index, tsize= None):
      """ index: sample index to be processed
          tsize: target size (h,w) in pixel
      """
      vobj = self.voc_list[index]  
      file_name = self.img_path + vobj[0]+ '.jpg';  # Full path of VOC2012 image
      rimg    = plt.imread(file_name);              # Load image file    
      rbox    = vobj[2];                            # Bounding boxes
      rlabel  = vobj[3];                            # Label of each bounding box
      rconf   = np.zeros((len(rlabel),80));         # Conference of each bounding box
      for k in range(len(rlabel)):                  # Convert label into confidence
        rconf[k, rlabel[k]] = 1.0;
      """ augmentation """
      if tsize is None: tsize = (self.img_size, self.img_size)
      simg = self._random_cue(rimg); #
      simg, sbox = self._random_fliplr(simg, rbox); # Random flip image lef-right
      simg, sbox, sconf = self._random_crop(simg, sbox, rconf, tsize); # Random crop 
      """ prevent from zero bounding bboxes """
      if sbox.shape[0] == 0: 
        simg = tf.image.resize(rimg/256, tsize)
        sbox = rbox; 
        sconf= rconf;
      return simg, sbox, sconf
    #.............................................. #--- Random cue
    def _random_cue(self, rimg):
      simg = np.zeros_like(rimg);
      for ch in range(3):
        simg[:,:,ch] = rimg[:,:,ch]* (0.8+ 0.2*np.random.rand(1))
      return simg
    #.............................................. #--- Random flip image
    def _random_fliplr(self, rimg, rbox):
      fliplr  = np.random.rand(1)> 0.5;             # True = flip left/right
      if fliplr == False: return rimg, rbox
      simg = np.fliplr(rimg);                       # 
      sbox = rbox.copy(); 
      sbox[:,0] = 1 - rbox[:,2]                     # rbox format = ltrb
      sbox[:,2] = 1 - rbox[:,0]
      return simg, sbox  
    #.............................................. #--- Random crop
    def _random_crop(self, rimg, rbox, rconf, tsize):
      img_h, img_w = rimg.shape[:2];                # Get image height and weight (pixel)
      """ compute cropping area (width, height)"""
      c_w = tsize[1]*(0.8+ np.random.rand(1)*0.4);  # Crop width  = target width * (0.8~1.2)
      c_h = tsize[0]*(0.8+ np.random.rand(1)*0.4);  # Crop height = target height* (0.8~1.2)
      shrink = np.minimum(img_h/c_h, img_w/c_w)*(0.9+ np.random.rand(1)*0.1)
      c_w = int(c_w * shrink);
      c_h = int(c_h * shrink);
      """ compute cropping area ltrb """
      x0 = int((img_w - c_w)/2* np.random.rand(1)); # Cropping box left
      y0 = int((img_h - c_h)/2* np.random.rand(1)); # Cropping box top
      x1 = int(x0 + c_w);                           # Cropping box right
      y1 = int(y0 + c_h);                           # Cropping box bottom
      """ resize image and adjust box """
      simg = tf.image.resize(rimg[y0:y1, x0:x1, :]/256, tsize) # Resize image
      tbox = np.zeros((rbox.shape[0], 4))           # 
      for n in range(rbox.shape[0]):
        tbox[n,0] = (rbox[n,0] - x0/img_w) / (c_w/img_w);  
        tbox[n,1] = (rbox[n,1] - y0/img_h) / (c_h/img_h);  
        tbox[n,2] = (rbox[n,2] - x0/img_w) / (c_w/img_w);  
        tbox[n,3] = (rbox[n,3] - y0/img_h) / (c_h/img_h);  
      """ label smoothing """
      eps   = 0.01;
      tconf = rconf*(1- eps) + eps/80; 
      """ filter out low confidence """
      clip  = np.maximum(np.minimum(tbox, 1), 0)    # Clipped box within cropped area
      sbox = []; sconf = [];
      for n in range(rbox.shape[0]):
        area_s = (tbox[n,2]- tbox[n,0]) * (tbox[n,3] - tbox[n,1])
        area_c = (clip[n,2]- clip[n,0]) * (clip[n,3] - clip[n,1])
        reveal_ratio = area_c/ area_s
        if reveal_ratio> self.IoU_crop:             # Only reserve high confidence bbox
          sbox.append(tbox[n,:])  
          sconf.append(tconf[n,:])
      sbox  = np.array(sbox);
      sconf = np.array(sconf);
      return simg, sbox, sconf  
    #.............................................. #--- Show image with bbox
    def show_image_box(self, img, box, title= None):
      plt.imshow(img); ax= plt.gca();               # Show image
      h, w,_ = img.shape
      for k in range(len(box)):                       # Draw each box
        xmin = w* box[k][0];
        ymin = h* box[k][1];
        ws   = w*(box[k][2]-  box[k][0]);      
        hs   = h*(box[k][3]-  box[k][1]);
        clr  = 'b'
        rect= patches.Rectangle((xmin,ymin),ws,hs,linewidth=1,edgecolor= clr,facecolor='none')  
        ax.add_patch(rect)                            # Show box
        if title is not None: plt.xlabel(title)
      plt.show()  
    #.............................................. #---
    
#=================================================== Compose batch
class Batch_generator(object):
    """ Convert {image, bbox, conf} into YOLO.v4 training dataset
        Parameters:            
        - img_size:     Final output image size
        
        Attributes:
        - batch_size:   Size of training batch
        - anchors:      YOLO.v4 anchors
        - max_bbox_per_scale: Max bounding boxes in 1 scale (=150)
        - IP:           Image pre-processor
        
        Functions (public):
          bimg, blabels, bboxes = convert(index);   # Convert index into 1 batch training data
          labels_list, sbox_list= encode(sbox, sconf)# Encode 1 iamge (box, conf) into list
    """    
    def __init__(self,
                 img_size   = 416,
                 IoU_thP    = 0.5,
                 IoU_crop   = 0.1,
                 **_kwargs):  
      self.img_size= img_size;                      # Image size   
      self.IP      = Image_processor(img_size= img_size, IoU_crop= IoU_crop);
      """ YOLO.v4 hyper parameters """
      self.IoU_thP = IoU_thP;                       # IoU threshold for valid bounding box
      self.max_bbox_per_scale = 100;                # Maximum bounding box for each scale
      self.strides = np.array([8, 16, 32])          # Strides
      self.grids   = (self.img_size / self.strides).astype(int);  
      self.anchors = get_anchors();
    #.............................................. #--- Convert index into training data
    def convert(self, index):
      self.batch_size = int(len(index)/13*4)
      grid_no = int((self.img_size/32)**2 *21*3)
      blabels= np.zeros((self.batch_size, grid_no, 85))         
      bimg   = np.zeros((self.batch_size, self.img_size, self.img_size, 3))
      bboxes = np.zeros((self.batch_size, 3, 100, 4)) # (s/m/l)
      """ """            
      for seg in range(int(len(index)/13)):         # 13 images -> 4 mosaic = 1 segment
        for m in range(4):
          if m<2: src_images =  index[seg*13+m*4: seg*13+(m+1)*4]
          else:   src_images = [index[seg*13+m*4]]
          simg, sbox, sconf      = self.IP.augment(src_images)
          labels_list, sbox_list = self.encode(sbox, sconf);
          bimg[   seg*4+ m] = simg; 
          blabels[seg*4+ m] = labels_list; 
          bboxes [seg*4+ m] = sbox_list;
      """ Convert into tensor """    
      bimg    = tf.cast(bimg,    dtype= tf.float32);  # Convert np-array into tensor
      blabels = tf.cast(blabels, dtype= tf.float32);  # 
      bboxes  = tf.cast(bboxes,  dtype= tf.float32);  # 
      return bimg, blabels, bboxes
    #.............................................. #--- Encode bbox into yolo.v4 training database
    def encode(self, sbox, sconf):
      """ labels_list:   (10647,85)  Label for focal loss and confidence loss computation
          sbox_list:     [s,m,l] (100,4):    Bounding box info for CIoU loss computation
      """  
      sbox_xywh = ltrb_to_xywh(sbox)* self.img_size;  # Convert sbox: xywh -> ltrb (pixel)
      sbox_list = np.zeros((3, 100, 4));
      labels = [];                                  # [(52,52,3,85), (26,26,3,85), (13,13,3,85)]  
      for layer in range(3):
        grid   = self.grids[layer];                 # Grid size of this layer  
        cmap   = np.zeros((grid, grid, 3, 85));     # Declare blank conv map
        labels.append(cmap);          # Declare blank labels
      """ check each box """ 
      anchors_xywh = np.zeros((9,4));               # Anchors
      anchors_xywh[:,2:4] = self.anchors;           # Anchors [:, 2:4] = [w, h]
      for s in range(sbox.shape[0]):
        strides = self.strides[np.arange(9)//3];    # = [8, 8, 8,16,16,16,32,32,32]
        xc      = sbox_xywh[s][0];                  # Box center X (in pixel)
        yc      = sbox_xywh[s][1];                  # Box center Y (in pixel) 
        anchors_xywh[:,0] = ((np.floor(xc / strides) + 0.5) * strides).astype(int);
        anchors_xywh[:,1] = ((np.floor(yc / strides) + 0.5) * strides).astype(int);
        anchors_ltrb = xywh_to_ltrb(anchors_xywh)/self.img_size   # Anchors in ltrb format
        """ Find anchor with largest IoU """
        IoU     = estimate_IOU(anchors_ltrb, sbox[s]);
        cand    = np.where(IoU > self.IoU_thP)[0];  # Candidate boxes for IoU > threshold
        if len(cand) == 0: cand = [np.argmax(IoU)]; # If no anchors > IoU_thP, choose best one
        #cand = [np.argmax(IoU)];                    # Select best only ???
        for m in range(len(cand)):
          layer = int(cand[m] / 3);
          chosen= int(cand[m] % 3);
          x_uind = int((np.floor(xc / strides))[cand[m]] )  # uind: un-bounded index
          y_uind = int((np.floor(yc / strides))[cand[m]] )  #  ind:    bounded index
          """ Out of bound process """
          max_grid= labels[layer].shape[0];         #  
          x_ind   = np.maximum(np.minimum(x_uind, max_grid -1),0)
          y_ind   = np.maximum(np.minimum(y_uind, max_grid -1),0)
          
          inlier  = ((x_ind == x_uind) and (y_ind == y_uind)) + 0.0
          
          labels[layer][y_ind, x_ind, chosen, :]   = 0.0;
          labels[layer][y_ind, x_ind, chosen, 0:4] = sbox_xywh[s];
          labels[layer][y_ind, x_ind, chosen, 4]   = inlier;
          #labels[layer][y_ind, x_ind, chosen, 4]   = 1.0;
          labels[layer][y_ind, x_ind, chosen, 5:]  = sconf[s];
          
          sbox_list[layer,s,:] = sbox_xywh[s];
      """ Reshape (13,13,3,85) into (507,85) and concat """
      a0 = tf.reshape(labels[0], (-1,85));
      a1 = tf.reshape(labels[1], (-1,85));
      a2 = tf.reshape(labels[2], (-1,85));
      labels_list = tf.concat([a0, a1, a2], axis=0)
      return labels_list, sbox_list
    #.............................................. #---  
    
#=================================================== Show result    
def get_box_colors(n_class):
      hsv_tuples = [(1.0 * x / n_class, 1., 1.) for x in range(n_class)]
      colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
      colors = list(map(lambda x: (x[0], x[1], x[2]), colors))  
      np.random.shuffle(colors)
      return colors 
    #.............................................. #---  
def show_desktop(img, kboxes, show_zoom, show_source):
      """ desktop = source image + zoom-in area """    
      if show_zoom == False: plt.imshow(img); return
      img_h  = img.shape[0];                        # Image height
      img_w  = img.shape[1];                        # Image width
      cell_s = int(img_h//4);                       # Cell size
      desktop= np.zeros((img_h, img_w+ cell_s*2,3)) # Blank desktop image
      desktop[:, :img_w, :] = img;
      if show_source == True:
        plt.imshow(desktop); plt.show();  
      #--- sort patches by patch size ---
      small_boxes = np.zeros((8,6));                # At most 8 boxes to be zoom-in
      small_boxes[:,1] = cell_s;  
      for k in range(len(kboxes)):  
        xc, yc, ws, hs = kboxes[k];  
        patch_size = np.maximum(ws, hs);            # Larger one of {ws, hs}
        idx  = np.argmax(small_boxes[:,1])          # Find largest one in candidate list
        if patch_size< small_boxes[idx, 1]: 
          small_boxes[idx] = np.array([k, patch_size, xc, yc, ws, hs])  
      #--- Zoom in box ---      
      for k in range(8):
        if small_boxes[k,1]< cell_s:
          x0 = int(small_boxes[k,2] - small_boxes[k,4] * 0.5);  
          y0 = int(small_boxes[k,3] - small_boxes[k,5] * 0.5);  
          x1 = int(small_boxes[k,2] + small_boxes[k,4] * 0.5);  
          y1 = int(small_boxes[k,3] + small_boxes[k,5] * 0.5); 
          patch= img[y0:y1, x0:x1, :]*256;
          patch= image_align(patch, img_size=cell_s, expand_dim= False)
          y2 = (k//2)     * cell_s; 
          x2 = np.mod(k,2)* cell_s + img_w; 
          desktop[y2: y2+cell_s, x2:x2+cell_s, :] = patch; 
      #--- Show label ---    
      plt.imshow(desktop);
      for k in range(8):
        if small_boxes[k,1]< cell_s:
          y2 = (k//2)     * cell_s; 
          x2 = np.mod(k,2)* cell_s + img_w;
          no = str(int(small_boxes[k,0]))
          plt.text(x2, y2+20, no, fontsize=8, color='white');
    #.............................................. #---  
def show_boxes(output, classes, colors, show_label, ax):  
      """ show boxes """
      kboxes = output[0];
      kclass = output[1];
      kscores= output[2];
      kprobs = output[3];
      for k in range(len(kboxes)):
        x0   = kboxes[k,0] - kboxes[k,2]*0.5;       # Box left
        y0   = kboxes[k,1] - kboxes[k,3]*0.5;       # Box top
        ws   = kboxes[k,2];
        hs   = kboxes[k,3]; 
        stype= classes[kclass[k]]
        score= int(kscores[k]*1000)/1000;
        prob = int(kprobs[ k]*1000)/1000;
        print(f'{k:3d} |  {stype:14s}|  S= {score:.3f}|  P= {prob:.3f}  |  (x0,y0)= ({int(x0):4d},{int(y0):4d})  |  (w,h)= ({int(ws):4d},{int(hs):4d})')  
        #--- define boxes style and show boxes  ---
        clr  = colors[kclass[k]];                   # Box edge color
        ls   = '-'
        if kscores[k]< 0.4: ls='--'
        if kscores[k]< 0.2: ls=':'
        rect= patches.Rectangle((x0,y0),            # Show box
                              ws,hs,
                              linewidth= 1,
                              edgecolor= clr,
                              linestyle= ls,
                              facecolor='none')  
        ax.add_patch(rect)                          # Show box        
        #--- show labels ---
        if show_label == True:
          if kboxes[k,2]>150 and kscores[k]>0.4:  
            label= classes[kclass[k]] + ' : ' + str(int(kscores[k]*1000)/1000)
            plt.text(x0, y0, label, fontsize=8, color='green');
      
#---------------------------------------------------- Show evaluation result    
def show_result(img,                                # Image to be show
                output,                             # Output from YOLO4 = [kboxes, kclass, kscores, kindex]
                title               = None,         # Plot title
                classes             = None,         # Class name of detector head
                show_source         = False,        # Show original image to be contrast or not
                show_zoom           = True,         # Zoom in small objects
                show_label          = True):        # Show label or not
    """ Input:
        - img:      (h,w,3) Source image to be detected
        - output:   [4,]    Output list from YOLO4 = [kboxes, kclass, kscores, kindex]
        - classes:          Class name list
        - show_source:      True: Show source image first
        - show_zoom:        True: zoom in small object
        - show_label:       True: show labels
    """
    if classes is None: 
        classes = get_class();                      # Preload class name    
    colors = get_box_colors(len(classes));          # Get box edge color
    if len(img.shape)==4: 
        img= np.squeeze(img, axis=0);   # Remove batch axis
    kboxes = output[0];
    show_desktop(img, kboxes, show_zoom, show_source) # Show desktop first
    ax= plt.gca();                                  # Show image
    show_boxes(output, classes, colors, show_label,ax) # Show boxes and labels
    if title is not None:
        plt.title(title)
    plt.show()  
    return    
#===================================================    
 