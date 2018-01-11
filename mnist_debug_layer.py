import caffe
# import numpy as np
import sys

class MnistDebug(caffe.Layer):
  
  def setup(self, bottom, top):
    top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2], bottom[0].data.shape[3])
    top[1].reshape(bottom[1].data.shape[0])
    self.printing = True
  
  def forward(self, bottom, top):
    top[0].data[...] = bottom[0].data
    top[1].data[...] = bottom[1].data
    
    if self.printing:
      image = bottom[0].data[0, ...]
      label = bottom[1].data[0, ...]
      sys.stdout.write('\nDebug print image:\n' + str(type(image)) + '\n' + image.dtype.name + '\n' + str(image))
      sys.stdout.write('\nDebug print label:\n' + str(type(label)) + '\n' + label.dtype.name + '\n' + str(label))
#       sys.stdout.write('\nDebug print image:\n')
#       sys.stdout.write(str(type(image))) # <type 'numpy.ndarray'>
#       sys.stdout.write('\n')
#       sys.stdout.write(image.dtype.name)
#       sys.stdout.write('\n')
#       sys.stdout.write(str(image))
#       sys.stdout.write('\nDebug print label:\n')
#       sys.stdout.write(str(type(label)))
#       sys.stdout.write('\n')
#       sys.stdout.write(label.dtype.name)
#       sys.stdout.write('\n')
#       sys.stdout.write(str(label))
#       sys.stdout.write('\n')
      self.printing = False
  
  def reshape(self, bottom, top):
    pass

  def backward(self, bottom, top):
    pass