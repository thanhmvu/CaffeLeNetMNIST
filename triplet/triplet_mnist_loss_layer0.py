"""
Triplet Mnist Loss Layer

Author:Thanh Vu
Date: Feb 3, 2018
"""

import caffe
import numpy as np
# from random import randint
# from random import shuffle
# from PIL import Image
# from tools import SimpleTransformer
# from random import getrandbits as randBool
# import sys


class TripletMnistLoss(caffe.Layer):
  
  # ==================================================================================== #
  #                                        MAIN                                          #
  # ==================================================================================== #
  
  def setup(self, bottom, top):
    # Check top shape
    if len(top) != 1:
      raise Exception("Need to define 1 top blob (loss).")
    
    # Check bottom shape
    if len(bottom) != 3:
      raise Exception("Need to define 3 bottom blobs (feat_anchor, feat_pos, feat_neg).")

    assert np.shape(bottom[0].data) == np.shape(bottom[1].data)
    assert np.shape(bottom[0].data) == np.shape(bottom[2].data)

    # Read parameters
    params = eval(self.param_str)
    self.margin = params["margin"]
    
    # Reshape top
    top[0].reshape(1)
  

  def reshape(self, bottom, top):
    """
    No reshape is needed at each pass because the input's size is fixed
    """
    # void EuclideanLossLayer<Dtype>::Reshape(
    # const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    # LossLayer<Dtype>::Reshape(bottom, top);
    # CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
    #     << "Inputs must have the same dimension.";
    # diff_.ReshapeLike(*bottom[0]);
    pass
    
    
  def forward(self, bottom, top):
    """
    Calculate loss
    """
    batchLoss = float(0)
    self.posLoss = []

    batch_size = bottom[0].num
    for i in range(batch_size):

      anchor = np.array(bottom[0].data[i], dtype='f') # shape (channel, height, width)
      pos = np.array(bottom[1].data[i], dtype='f')
      neg = np.array(bottom[2].data[i], dtype='f')

      ap_dist = np.sum((anchor - pos) ** 2)
      an_dist = np.sum((anchor - neg) ** 2)

      imgLoss = max(ap_dist - an_dist + self.margin, 0.0)
      if imgLoss > 0: self.posLoss.append(i)

      batchLoss += imgLoss
   
    avgLoss = batchLoss/batch_size ###### [?] divide by 2? divide by batch_size?
    top[0].data[...] = avgLoss


  def backward(self, top, propagate_down, bottom):
    """Get top diff and compute diff in bottom."""
    if propagate_down[0]:
      batch_size = bottom[0].num
      for i in range(batch_size):

        if i in self.posLoss:
          anchor = np.array(bottom[0].data[i]) # shape (channel, height, width)
          pos = np.array(bottom[1].data[i])
          neg = np.array(bottom[2].data[i])
          
          bottom[0].diff[i] =  2*(pos - neg) ##### pos/neg of 1 img?
          bottom[1].diff[i] =  -2*(anchor - pos)
          bottom[2].diff[i] =  2*(anchor - neg)

        else:
          bottom[0].diff[i] = np.zeros((bottom[0].data)[1].shape)
          bottom[1].diff[i] = np.zeros((bottom[0].data)[1].shape)
          bottom[2].diff[i] = np.zeros((bottom[0].data)[1].shape)


# def tripletLoss(pred, margin):
#     a = pred[0::3]
#     p = pred[1::3]
#     n = pred[2::3]
#     d1 = T.sum((a - p)**2, axis=-1)
#     d2 = T.sum((a - n)**2, axis=-1)
#     return T.mean(T.maximum(0, d1 + margin - d2))

# with tf.name_scope("triplet_loss"):
#         d_p_squared = tf.square(compute_euclidean_distance(anchor_feature, positive_feature))
#         d_n_squared = tf.square(compute_euclidean_distance(anchor_feature, negative_feature))

#         loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
#         #loss = d_p_squared - d_n_squared + margin

#         return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)
