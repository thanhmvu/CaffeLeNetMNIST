"""
Train triplet using custom solver loop with online selection of hard triplets 

Author: Thanh Vu 
Date: Feb 3, 2018

References:
  + Custom solver loop: http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb
"""

import scipy.misc
import sys
import caffe
from caffe import layers as L, params as P
import numpy as np
from PIL import Image
import random
import time

caffe_root = '/auto/research2/vut/caffe-rc5/'
current_dir = '/auto/research2/vut/thesis/CaffeLeNetMNIST/triplet/'

# Names of blobs
DATA_ANCHOR = 'data_anchor'
DATA_POS = 'data_pos'
DATA_NEG = 'data_neg'
LOSS = 'loss'

# Network params
niter = 50000
# # test_iter specifies how many forward passes the test should carry out.
# # In the case of MNIST, we have test batch size 100 and 100 test iterations,
# # covering the full 10,000 testing images.
# test_iter = 100 
# test_images = 1e4
# # Carry out testing every 500 training iterations.
# test_interval = 500 #25
IS_COLOR = False
CHANNELS = 3 if IS_COLOR else 1
IMG_SIZE = 28
# For Mnist, use range [0,1]
# thus, scale = 0.00390625 if use Image.open()
# scale = 1.0 if use caffe.io.load_image(),
SCALE = 0.00390625 


sys.path.insert(0, current_dir)

# ==================================================================================== #
#                                       METHODS                                        #
# ==================================================================================== #

# use net, blob_name, scale instead
# to eliminate hard-coded 'data' name
def config_transformer(shape, scale, is_color):
  transformer = caffe.io.Transformer({'data': shape})
  transformer.set_raw_scale('data', scale)
#   transformer.set_transpose('data', (2,0,1)) # C x H x W
#   if is_color:
#     transformer.set_channel_swap('data', (2,1,0)) # imagenet and caffe use BGR
  # transformer.set_mean('data', np.load('/auto/research2/vut/caffe-rc5/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
  return transformer


def preprocess(im, transformer):
  return transformer.preprocess('data', im) 


def load_image(path, is_color):
  """
  Load an image converting to RGB or grayscale.
  Parameters
  ----------
  path : string
  is_color : boolean
  
  Returns
  -------
  image : an image with type np.float32 in range [0, 255]
      of size (H x W x 3) if RGB or (H x W x 1) if grayscale.
  """
  mode = 'RGB' if is_color else 'L'
  img = Image.open(path)
  if img.mode != mode: img = img.convert(mode)
  return np.array(img, dtype='f')


def rand_triplet_paths(img_paths_by_label):
  labels = img_paths_by_label.keys()
  label, label_neg = random.sample(labels, 2)
  
  anchor_path = random.choice(img_paths_by_label[label])
  pos_path = random.choice(img_paths_by_label[label])
  neg_path = random.choice(img_paths_by_label[label_neg])
  
  return (anchor_path, pos_path, neg_path)


def is_hard_triplet(net, triplet_imgs):
  if net == None or triplet_imgs == None:
    return False
  
  batch_shape = np.shape(net.blobs[DATA_ANCHOR].data)
  
  # Load data
  blob_names = [DATA_ANCHOR, DATA_POS, DATA_NEG]
  for i, blob_name in enumerate(blob_names):
    net.blobs[blob_name].reshape(1, batch_shape[1], batch_shape[2], batch_shape[3]) # feed 1 image at a time 
    net.blobs[blob_name].data[...] = triplet_imgs[i]
  
  # Compute
  out = net.forward() # OR net.forward(start='conv1')
  loss = net.blobs[LOSS].data
  is_hard = (loss > 0)
  
  # Reset data shape for training
  for blob_name in blob_names:
    net.blobs[blob_name].reshape(batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3])
    
  return is_hard


def find_hard_triplets(net, img_paths_by_label, is_color, transformer):
  batch_size = len(net.blobs[DATA_ANCHOR].data)
  minibatch = []
  while len(minibatch) < batch_size:
    triplet_imgs = None
    while not is_hard_triplet(net, triplet_imgs):
      triplet_paths = rand_triplet_paths(img_paths_by_label)
      triplet_imgs = [preprocess(load_image(path, is_color), transformer) for path in triplet_paths]
    minibatch.append(triplet_imgs)
  return minibatch    
    
  
def readImagePaths(source):
  img_paths_by_label = {}
  with open(source) as f:
    for line in f.readlines():
      imagePath, label = line.strip("\n").split(" ")
      if label not in img_paths_by_label:
        img_paths_by_label[label] = [imagePath]
      else:
        img_paths_by_label[label].append(imagePath)
  return img_paths_by_label


# ==================================================================================== #
#                                        MAIN                                          #
# ==================================================================================== #
def main(): 
  sys.stdout.write('\nStart training\n')
  caffe.set_device(0)
  caffe.set_mode_gpu()
  
  # Load the Solver
#   solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
  solver = caffe.SGDSolver(current_dir + 'tripletLenet_customSolverLoop_solver.prototxt')

  # Load data paths
  train_img_paths_by_label = readImagePaths(current_dir + '../train.txt')
  test_img_paths_by_label = readImagePaths(current_dir + '../test.txt')

  # Config preprocessing
  transformer = config_transformer(solver.net.blobs[DATA_ANCHOR].data.shape, SCALE, IS_COLOR)

  # losses will also be stored in the log
  train_loss = np.zeros(niter)
#   test_acc = np.zeros(int(np.ceil(niter / test_interval)))
#   output = np.zeros((niter, 8, 10))

  train_batch_size = len(solver.net.blobs[DATA_ANCHOR].data)
  next_batch = []
  # Randomly select the first minibatch
  for i in range(train_batch_size):
    triplet_paths = rand_triplet_paths(train_img_paths_by_label)
    triplet_imgs = [preprocess(load_image(path, IS_COLOR), transformer) for path in triplet_paths]
    next_batch.append(triplet_imgs)

  # Main solver loop
  # this custom loop allow for training with additional computations
  # for training with only specification from solver.prototxt, use solver.solve() 
  start = time.time()
  for it in range(niter):
    # Add minibatch to data blob
    for i in range(len(next_batch)):
      anc, pos, neg = next_batch[i]
      solver.net.blobs[DATA_ANCHOR].data[i, ...] = anc
      solver.net.blobs[DATA_POS].data[i, ...] = pos
      solver.net.blobs[DATA_NEG].data[i, ...] = neg
      
    # Train the network
    solver.step(1)  # SGD by Caffe with dummy input data layer
    
    # Find hard triplets using current weights
    next_batch = find_hard_triplets(solver.net, train_img_paths_by_label, IS_COLOR, transformer)
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data

#     # run a full test every so often (Caffe itself can also do this)
#     if it % test_interval == 0:
#       print 'Iteration', it, 'testing...'
#       correct = 0
#       for test_it in range(test_iter):
#         solver.test_nets[0].forward()
#         correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1) == solver.test_nets[0].blobs['label'].data)
#       test_acc[it // test_interval] = correct / test_images

    if it % 50 == 0:
      end = time.time()
      print 'Time train 50 batches: ', (end - start)
      start = time.time()

  print 'train_loss', train_loss
#   print 'test_acc', test_acc


if __name__ == "__main__":
    main()
