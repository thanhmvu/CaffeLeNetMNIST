"""
Triplet Mnist Loss Layer

Author: Peng Zhang (hizhangp)
Adopt by: Thanh Vu
Date: Feb 3, 2018

References:
  - Source: https://github.com/hizhangp/triplet/blob/master/triplet/tripletloss_layer.py
"""

import caffe
import numpy as np


class TripletMnistLoss(caffe.Layer):

    def setup(self, bottom, top):
        """Setup the TripletLayer."""

        assert bottom[0].num == bottom[1].num, '{} != {}'.format(
            bottom[0].num, bottom[1].num)
        assert bottom[0].num == bottom[2].num, '{} != {}'.format(
            bottom[0].num, bottom[2].num)

        params = eval(self.param_str)
        self.margin = params["margin"]

        top[0].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        
        anchor = np.array(bottom[0].data)                                     # shape = (batch_size, feature_size)
        positive = np.array(bottom[1].data)
        negative = np.array(bottom[2].data)
        
        aps = np.sum((anchor - positive) ** 2, axis=1)                        # shape = (batch_size)
        ans = np.sum((anchor - negative) ** 2, axis=1)

        dist = self.margin + aps - ans                                        # shape = (batch_size)
        dist_hinge = np.maximum(dist, 0.0)                                    # shape = (batch_size)

        self.residual_list = np.asarray(dist_hinge > 0.0, dtype=np.float)     # shape = (batch_size)
        loss = np.sum(dist_hinge) / bottom[0].num                             # ex: 0.447827339172

        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""
        if propagate_down[0]:
            anchor = np.array(bottom[0].data)
            positive = np.array(bottom[1].data)
            negative = np.array(bottom[2].data)

            coeff = 2.0 * top[0].diff / bottom[0].num                         # coeff = 0.03125
            bottom_a = coeff * \
                np.dot(np.diag(self.residual_list), (negative - positive))
            bottom_p = coeff * \
                np.dot(np.diag(self.residual_list), (positive - anchor))
            bottom_n = coeff * \
                np.dot(np.diag(self.residual_list), (anchor - negative))

            bottom[0].diff[...] = bottom_a                                    # shape = (batch_size, feature_size)
            bottom[1].diff[...] = bottom_p
            bottom[2].diff[...] = bottom_n

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass