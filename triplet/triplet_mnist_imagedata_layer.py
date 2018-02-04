"""
A python layer for loading MNIST data to Triplet LeNet
An alternative of Caffe lmdb data layer

Author: Thanh Vu 
Date: Feb 3, 2018

References:
  - Caffe MNIST tutorial: 
    + http://caffe.berkeleyvision.org/gathered/examples/mnist.html
  - Caffe python layers:
    + https://gist.github.com/rafaspadilha/a67008cc3bd93bc2c1fc368c363ee363#--why-would-i-want-to-do-that
    + https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/roi_data_layer/layer.py
    + https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pascal_multilabel_datalayers.py
"""

import caffe
import numpy as np
from PIL import Image
from tools import SimpleTransformer
import random
from random import getrandbits as randBool
import sys


class TripletMnistImageData(caffe.Layer):

  # ==================================================================================== #
  #                                   HELPER METHODS                                     #
  # ==================================================================================== #
  
  def readImagePathTuples(self, source):
    '''
    Input:  source file containing lines of ImagePath tuples 
            in the format of "AnchorImgPath1 PosImgPath1 NegImgPath1\nAnchorImgPath2 PosImgPath2 NegImgPath2\n ..."
    Output: a list of (AnchorImgPath, PosImgPath, NegImgPath) tuples
    '''
    imagePathTuples = []
    with open(source) as f:
      for line in f.readlines():
        imagePathTuple = line.strip("\n").split(" ")
        imagePathTuples.append(imagePathTuple)
    return imagePathTuples
  
  
  def rand_triplet_paths(self, img_paths_by_label):
    labels = img_paths_by_label.keys()
    label, label_neg = random.sample(labels, 2)

    anchor_path = random.choice(img_paths_by_label[label])
    pos_path = random.choice(img_paths_by_label[label])
    neg_path = random.choice(img_paths_by_label[label_neg])

    return (anchor_path, pos_path, neg_path)

  
  def readImagePaths(self, source):
    img_paths_by_label = {}
    with open(source) as f:
      for line in f.readlines():
        imagePath, label = line.strip("\n").split(" ")
        if label not in img_paths_by_label:
          img_paths_by_label[label] = [imagePath]
        else:
          img_paths_by_label[label].append(imagePath)
    return img_paths_by_label

  
  def loadNextImage(self):
    '''
    Input:  none
    Output: the next image.
            This method should be called after readImagePathTuples() is called
    '''
    # Finished forwarding all image. Start a new epoch
    if self._cur == len(self.imagePathTuples):
      self._cur = 0
      if self.shuffle:
        random.shuffle(self.imagePathTuples)

    imgPaths = self.imagePathTuples[self._cur]
    imgs = [Image.open(path) for path in imgPaths]
    
    mode = 'RGB' if self.is_color else 'L'
    for i in range(len(imgs)):
      if imgs[i].mode != mode: imgs[i] = imgs[i].convert(mode)
      imgs[i] = np.array(imgs[i])
      
    self._cur += 1
    return imgs

  
  # ==================================================================================== #
  #                                        MAIN                                          #
  # ==================================================================================== #
  
  def setup(self, bottom, top):
    # Check top shape
    if len(top) != 3:
      raise Exception("Need to define 3 top blobs (data__anchor, data__pos, data__neg)")
    
    # Check bottom shape
    if len(bottom) != 0:
      raise Exception("Do not define a bottom.")

    # Read parameters
    params = eval(self.param_str)
    source = params["source"]
    self.batch_size = params["batch_size"]
    self.image_size = params["image_size"]
    self.train_size = params["train_size"]
    self.scale = params.get("scale", 1)
    self.crop_size = params.get("crop_size", None)
#     self.mean_file = params.get("mean_file", None)
#     self.mirror = params.get("mirror", False)
    self.shuffle = params.get("shuffle", False)
    self.is_color = params.get("is_color", False)
    self.chanels = 3 if self.is_color else 1 # RGB

#     self.transformer = SimpleTransformer()
    
    # Reshape top
    if self.crop_size:
      top[0].reshape(self.batch_size, self.chanels, self.crop_size, self.crop_size) # image 1
      top[1].reshape(self.batch_size, self.chanels, self.crop_size, self.crop_size) # image 2
      top[2].reshape(self.batch_size, self.chanels, self.crop_size, self.crop_size) # image 2
    else:
      top[0].reshape(self.batch_size, self.chanels, self.image_size, self.image_size) # image 1
      top[1].reshape(self.batch_size, self.chanels, self.image_size, self.image_size) # image 2
      top[2].reshape(self.batch_size, self.chanels, self.image_size, self.image_size) # image 2
    
    # Read source file
    self.imagePathTuples = self.readImagePathTuples(source) # list of (AnchorImgPath, PosImgPath, NegImgPath) tuples
#     train_img_paths_by_label = self.readImagePaths(source)
#     self.imagePathTuples = []
#     for i in range(self.train_size):
#       self.imagePathTuples.append(self.rand_triplet_paths(train_img_paths_by_label))

#     # Config preprocessing
#     if self.mean_file:
#       self.transformer.set_mean(np.load(self.mean_file).mean(1).mean(1)) 
#     if self.scale:
#       self.transformer.set_scale(self.scale)

    self._cur = 0 # used to check if we finished the list of images
    # print_info("DataFileLayer", params)
    
    
  def forward(self, bottom, top):
    """
    Load images
    """
    for j in range(self.batch_size):
      # Load next image using the next imagePath from the list of image paths
      # which was loaded at setup
      imgTuple = self.loadNextImage()      
      for i in range(len(imgTuple)):
        img = imgTuple[i]
        
#         # Resize image
#         img = np.array(img.resize((self.image_size, self.image_size), Image.ANTIALIAS))

#         # Random crop
#         if self.crop_size:
#           h_off = random.randint(0, self.image_size - self.crop_size)
#           w_off = random.randint(0, self.image_size - self.crop_size)
#           img = img[h_off : (h_off + self.crop_size), w_off : (w_off + self.crop_size)]

#         # Random mirror (horizontal flip)
#         if self.mirror:
#           if randBool(1): img = img[:, ::-1, :]
            
        # Caffe preprocess
#         img = self.transformer.preprocess(img)
        img = img * self.scale

        # Add directly to the top blob
        top[i].data[j, ...] = img

  
  def reshape(self, bottom, top):
    """
    No reshape is need at each pass because the input'size is fixed
    """
    pass


  def backward(self, bottom, top):
    """
    Input data layer does not back propagate
    """
    pass
