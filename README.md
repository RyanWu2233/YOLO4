## YOLO4 (You Only Look Once, version4) - using TensorFlow 2.1
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![TensorFlow 2.10](https://img.shields.io/badge/tensorflow-2.10-green.svg?style=plastic)
![Repo COCO/VOC](https://img.shields.io/badge/Repository-COCO/VOC-green.svg?style=plastic)
![Image size 608](https://img.shields.io/badge/Image_size-512x512-green.svg?style=plastic) 

Tensorflow 2.1 implementation for YOLO V4

## YOLO V4 Detection example:
Folloing picture illustrates the capability of object localization and object identification capability of YOLO V4. Table below lists objects recoginized by YOLO V4. S represents probability of each object where 1 means 100% sure. P represents confidence level of the object class. (x0, y0) means left and top boundary. (w,h) means width and height of the box.  
It is clear that YOLO V4 performance well for both large object (object 0, 1) and small object (object 9, 10, 11 which are amplified on the right of image). 
![Result_2](./JPG/YOLO4_01B.jpg) 
![Result_1A](./JPG/YOLO4_01A.jpg)  

![Result_2](./JPG/YOLO4_02.jpg) 
![Result_2A](./JPG/YOLO4_02A.jpg)  


## False positive example:
YOLO V4 detects 26 objects Folloing picture 
![Result_3](./JPG/YOLO4_03B.jpg) 
![Result_3A](./JPG/YOLO4_03A.jpg)  


