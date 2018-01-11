# =========================================
# Load images by batches

# import caffe
# import numpy as np
# # from random import getrandbits as randBool
# from random import shuffle
# from PIL import Image
# # import scipy.misc
# # from tools import SimpleTransformer

# class MnistImageData(caffe.Layer):
  
#   def read_data(self, source):
#     imgLabelPairs = []
#     with open(source) as f:
#       for line in f.readlines():
#         imgLabelPair = line.strip("\n").split(" ")
#         imgLabelPairs.append(imgLabelPair)
#     return imgLabelPairs
  
#   def load_next_image(self):
#     # If we have finished forwarding all images, then an epoch has finished
#     # and it is time to start a new one
#     if self._cur == len(self.imgLabelPairs):
#       self._cur = 0
#       shuffle(self.imgLabelPairs)
    
#     imagePath, label = self.imgLabelPairs[self._cur]
#     image = np.array(Image.open(imagePath))
# #     print "Image: ", image

#     self._cur += 1
#     return image, label
  
#   def setup(self, bottom, top):
#     # Check top shape
#     if len(top) != 2:
#       raise Exception("Need to define 2 top blobs (data, label)")
    
#     # Check bottom shape
#     if len(bottom) != 0:
#       raise Exception("Do not define a bottom.")

#     # Read parameters
#     params = eval(self.param_str)
#     source = params["source"]
#     self.batch_size = params["batch_size"]
#     self.scale = params["scale"]
#     self.image_size = params["image_size"]

# #     self.transformer = SimpleTransformer()

#     # Reshape?
#     top[0].reshape(self.batch_size, 1, self.image_size, self.image_size) # image
#     top[1].reshape(self.batch_size) # label

#     # Read source file
#     # returns a list of tuples (imagePath, label)
#     self.imgLabelPairs = self.read_data(source) 
    
#     self._cur = 0 #use this to check if we need to restart the list of imgs
#     # print_info("DataFileLayer", params)
  
  
#   def forward(self, bottom, top):
#     """
#     Load images
#     """
#     for i in range(self.batch_size):
#       # Use the batch loader to load the next image.
#       image, label = self.load_next_image()
      
#       # Preprocess
# #       # =========== TODO: what preprocessing that caffe lmdb layer do?
# #       img1 = scipy.misc.imresize(img1, (self.image_size, self.image_size))
# #       img2 = scipy.misc.imresize(img2, (self.image_size, self.image_size))
      
# #       self.transformer.set_transpose('data', (2,0,1))
# #       self.transformer.set_mean('data', np.load('/auto/research2/vut/caffe-rc5/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # ?
# #       self.transformer.set_channel_swap('data', (2,1,0)) #? imagenet, dyads
# #       self.transformer.set_raw_scale('data', 255.0) # ? check the actual image? first row
      
#       # Add directly to the top blob
#       top[0].data[i, ...] = image
#       top[1].data[i, ...] = label
  
#   def reshape(self, bottom, top):
#     """
#     There is no need to reshape the data, since the input is of fixed size
#     (img shape and batch size)

#     If we were processing a fixed-sized number of images (for example in Testing)
#     and their number wasn't a  multiple of the batch size, we would need to
#     reshape the top blob for a smaller batch.
#     """
#     pass

#   def backward(self, bottom, top):
#     """
#     This layer does not back propagate
#     """
#     pass

# =========================================
# Load all images upfront

import caffe
import numpy as np
from random import shuffle
from PIL import Image
# import scipy.misc
# from tools import SimpleTransformer
# from random import getrandbits as randBool

class MnistImageData(caffe.Layer):
  
  def read_data(self, source):
    imgLabelPairs = []
    with open(source) as f:
      for line in f.readlines():
        imagePath, label = line.strip("\n").split(" ")
        image = np.array(Image.open(imagePath))
        imgLabelPairs.append([image, label])
    return imgLabelPairs
  
  def load_next_image(self):
    # If we have finished forwarding all images, then an epoch has finished
    # and it is time to start a new one
    if self._cur == len(self.imgLabelPairs):
      self._cur = 0
      shuffle(self.imgLabelPairs)
    image, label = self.imgLabelPairs[self._cur]
    self._cur += 1
    return image, label
  
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

#     self.transformer = SimpleTransformer()

    # Reshape?
    top[0].reshape(self.batch_size, 1, self.image_size, self.image_size) # image
    top[1].reshape(self.batch_size) # label

    # Read source file
    # returns a list of tuples (imagePath, label)
    self.imgLabelPairs = self.read_data(source) 
    
    self._cur = 0 # use this to check if we need to restart the list of imgs
    # print_info("DataFileLayer", params)
  
  
  def forward(self, bottom, top):
    """
    Load images
    """
    for i in range(self.batch_size):
      # Use the batch loader to load the next image.
      image, label = self.load_next_image()
      
      # Preprocess
      image = image * self.scale
#       # =========== TODO: what preprocessing that caffe lmdb layer do?
#       img1 = scipy.misc.imresize(img1, (self.image_size, self.image_size))
#       img2 = scipy.misc.imresize(img2, (self.image_size, self.image_size))
      
#       self.transformer.set_transpose('data', (2,0,1))
#       self.transformer.set_mean('data', np.load('/auto/research2/vut/caffe-rc5/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # ?
#       self.transformer.set_channel_swap('data', (2,1,0)) #? imagenet, dyads
#       self.transformer.set_raw_scale('data', 255.0) # ? check the actual image? first row
      
      # Add directly to the top blob
      top[0].data[i, ...] = image
      top[1].data[i, ...] = label
  
  def reshape(self, bottom, top):
    """
    There is no need to reshape the data, since the input is of fixed size
    (img shape and batch size)

    If we were processing a fixed-sized number of images (for example in Testing)
    and their number wasn't a  multiple of the batch size, we would need to
    reshape the top blob for a smaller batch.
    """
    pass

  def backward(self, bottom, top):
    """
    This layer does not back propagate
    """
    pass