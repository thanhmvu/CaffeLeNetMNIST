# ==================================================================================== #
# A python layer for loading MNIST data to LeNet
# An alternative of Caffe lmdb data layer
# Built on top of Caffe's LeNet MNIST example http://caffe.berkeleyvision.org/gathered/examples/mnist.html
# 
# Sample prototxt:
#   layer {
#     name: "mnist"
#     type: "Python"
#     top: "data"
#     top: "label"
#     include { 
#       phase: TRAIN 
#     }
#     python_param {
#       module: "mnist_imagedata_layer"
#       layer: "MnistImageData"
#       param_str: "{'batch_size': 64,'scale': 0.00390625, 'image_size': 28, 'source': '/auto/research2/vut/thesis/CaffeLeNetMNIST/train.txt'}"   # optional
#     }
#   }
#
# Note: "source" file should be in the format of "ImagePath1 Label1\nImagePath2 Label2\n ..."
# 
# ==================================================================================== #

import caffe
import numpy as np
from random import shuffle
from PIL import Image
import sys


class MnistImageData(caffe.Layer):
  
  # ==================================================================================== #
  #                                   HELPER METHODS                                     #
  # ==================================================================================== #
  
  def readImagePathLabelPairs(self, source):
    '''
    Input:  source file containing lines of ImagePath-Label pairs 
            in the format of "ImagePath1 Label1\nImagePath2 Label2\n ..."
    Output: a list of (ImagePath, Label) pairs
    '''
    imagePathLabelPairs = []
    with open(source) as f:
      for line in f.readlines():
        imagePathLabelPair = line.strip("\n").split(" ")
        imagePathLabelPairs.append(imagePathLabelPair)
    return imagePathLabelPairs
  
  
  def loadNextImage(self):
    '''
    Input:  none
    Output: the next image.
            This method should be called after readImagePathLabelPairs() is called
    '''
    # Finished forwarding all image. Start a new epoch
    if self._cur == len(self.imagePathLabelPairs):
      self._cur = 0
      shuffle(self.imagePathLabelPairs)
    imagePath, label = self.imagePathLabelPairs[self._cur]
    image = np.array(Image.open(imagePath))
    self._cur += 1
    return image, label
  
  
  def loadImageLabelPairs(self, source):
    '''
    Input:  source file containing lines of ImagePath-Label pairs 
            in the format of "ImagePath1 Label1\nImagePath2 Label2\n ..."
    Output: a list of (Image, Label) pairs
    '''
    imageLabelPairs = []
    with open(source) as f:
      for line in f.readlines():
        imagePath, label = line.strip("\n").split(" ")
        image = np.array(Image.open(imagePath))
        imageLabelPairs.append([image, label])
    return imageLabelPairs
  
  
  def getNextImage(self):
    '''
    Input:  none
    Output: the next image.
            This method should be called after loadImageLabelPairs() is called
    '''
    # Finished forwarding all image. Start a new epoch
    if self._cur == len(self.imageLabelPairs):
      self._cur = 0
      shuffle(self.imageLabelPairs)
    image, label = self.imageLabelPairs[self._cur]
    self._cur += 1
    return image, label
  
  
  # ==================================================================================== #
  #                                        MAIN                                          #
  # ==================================================================================== #
  
  def setup(self, bottom, top):
    # Check top shape
    if len(top) != 2:
      raise Exception("Need to define 2 top blobs (data, label)")
    
    # Check bottom shape
    if len(bottom) != 0:
      raise Exception("Do not define a bottom.")

    # Read parameters
    params = eval(self.param_str)
    source = params["source"]
    self.batch_size = params["batch_size"]
    self.scale = params["scale"]
    self.image_size = params["image_size"]

    # Reshape?
    top[0].reshape(self.batch_size, 1, self.image_size, self.image_size) # image
    top[1].reshape(self.batch_size) # label

    # Read source file
    # returns a list of (images, label) pairs
#     self.imageLabelPairs = self.loadImageLabelPairs(source) # Load all images upfront
    self.imagePathLabelPairs = self.readImagePathLabelPairs(source) # Load images paths only
    
    self._cur = 0 # use this to check if we need to restart the list of images
    # print_info("DataFileLayer", params)
  
  
  def forward(self, bottom, top):
    """
    Load images
    """
    for i in range(self.batch_size):
#       # Get next image from the list of images
#       # which was loaded at setup
#       image, label = self.getNextImage() 
      # Load next image using the next imagePath from the list of image paths
      # which was loaded at setup
      image, label = self.loadNextImage() 
      
      # Preprocess
      image = image * self.scale
      
      # Add directly to the top blob
      top[0].data[i, ...] = image
      top[1].data[i, ...] = label
      sys.stdout.write('\ntop[0].data[i, ...]:' + str(top[0].data[i, ...].shape) +'\n')
      sys.stdout.write('\ntop[0].data:' + str(top[0].data.shape) +'\n')
      sys.stdout.write('\nimage shape:' + str(image.shape) + ' ' + str(type(image)) +'\n')
      sys.stdout.write('\ntop[1].data:' + str(top[1].data.shape) +'\n')
      sys.stdout.write('\nlabel:' + str(label) + ' ' + str(type(label)) + '\n')
  
  
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
  