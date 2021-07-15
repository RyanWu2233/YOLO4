# -*- coding: utf-8 -*-
"""
Module name: _train
Author:      Ryan Wu
Date:        V0.1- 2020/12/01: Initial release
Description: Demo YOLO.v4 inference 
"""

#---------------------------------------------------- Import libraries
from __future__ import absolute_import, division, unicode_literals, print_function
import yolo4_main 

#----------------------------------------------------
YOLO4 = yolo4_main.YOLO_V4(img_size= 416, batch_size= 32,dataset='VOC');
#YOLO4.interference(thresh=0.15);
#YOLO4.train(restart= True)
YOLO4.train(restart= False)
#YOLO4.show_curve() 
YOLO4.evaluate()

