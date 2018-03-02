"""
Triplet Mnist Loss Layer

Author:Thanh Vu
Date: Feb 9, 2018
"""

import caffe
import numpy as np


class TripletMnistLoss(caffe.Layer):
  
  def setup(self, bottom, top):
    # Check top shape
    assert len(top) == 1, "Top needs to have exactly 1 blob: loss."
    
    # Check bottom shape
    assert len(bottom) == 3, "Bottom needs to have exactly 3 blobs: feat_anchor, feat_pos, feat_neg."
    assert np.shape(bottom[0].data) == np.shape(bottom[1].data), "bottom[0] and bottom[1] should have the same shape."
    assert np.shape(bottom[0].data) == np.shape(bottom[2].data), "bottom[0] and bottom[2] should have the same shape."

    # Read parameters
    params = eval(self.param_str)
    self.margin = params["margin"]
    
    # Reshape top
    top[0].reshape(1)
  

  def reshape(self, bottom, top):
    """
    No reshape is needed at each pass because the input's size is fixed
    """
    pass
    
    
  def forward(self, bottom, top):
    """
    Calculate loss
    """
    total_loss = float(0)
    self.pos_losses = []
    batch_size = bottom[0].num

    for i in range(batch_size):
      anchor_feat = bottom[0].data[i]
      pos_feat = bottom[1].data[i]
      neg_feat = bottom[2].data[i]

      ap_sq_dist = np.sum((anchor_feat - pos_feat) ** 2)
      an_sq_dist = np.sum((anchor_feat - neg_feat) ** 2)
      img_loss = max(ap_sq_dist - an_sq_dist + self.margin, 0.0)
      
      if img_loss > 0: self.pos_losses.append(i)
      total_loss += img_loss
   
    avg_loss = total_loss / batch_size
    top[0].data[...] = avg_loss


  def backward(self, top, propagate_down, bottom):
    """Get top diff and compute diff in bottom."""
    if propagate_down[0]:
      batch_size = bottom[0].num

      for i in range(batch_size):
        if i in self.pos_losses:
          anchor_feat = bottom[0].data[i]
          pos_feat = bottom[1].data[i]
          neg_feat = bottom[2].data[i]
          
          bottom[0].diff[i] = 2 * (neg_feat - pos_feat) / batch_size
          bottom[1].diff[i] = 2 * (pos_feat - anchor_feat) / batch_size
          bottom[2].diff[i] = 2 * (anchor_feat - neg_feat) / batch_size

        else:
          bottom[0].diff[i] = 0.0
          bottom[1].diff[i] = 0.0
          bottom[2].diff[i] = 0.0
