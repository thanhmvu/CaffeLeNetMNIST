# ==================================================================================== #
# Plot siamese embeddings
#
# Author: Thanh Vu 
# Date: Feb 3, 2018
#
# References:
#   + http://caffe.berkeleyvision.org/gathered/examples/siamese.html
#   + https://github.com/BVLC/caffe/blob/master/examples/siamese/mnist_siamese.ipynb
# ==================================================================================== #

# Handle _tkinter.TclError: no display name and no $DISPLAY environment variable
import matplotlib
matplotlib.use('Agg')

caffe_root = '/auto/research2/vut/caffe-rc5/'
current_dir = '/auto/research2/vut/thesis/CaffeLeNetMNIST/siamese/'

import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


MODEL_FILE = current_dir + 'mnist_siamese.prototxt' # deploy model
# PRETRAINED_FILE = caffe_root + 'examples/siamese/mnist_siamese_iter_50000.caffemodel' 
ITERATION = 1000
PRETRAINED_FILE = current_dir + '../snapshots/siamese_imagedata/mnist_siamese_imagedata_iter_%d.caffemodel' % (ITERATION) 

# caffe.set_mode_cpu()
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED_FILE, caffe.TEST)


TEST_DATA_FILE = caffe_root + 'data/mnist/t10k-images-idx3-ubyte'
TEST_LABEL_FILE = caffe_root + 'data/mnist/t10k-labels-idx1-ubyte'
n = 10000

with open(TEST_DATA_FILE, 'rb') as f:
    f.read(16) # skip the header
    raw_data = np.fromstring(f.read(n * 28*28), dtype=np.uint8)

with open(TEST_LABEL_FILE, 'rb') as f:
    f.read(8) # skip the header
    labels = np.fromstring(f.read(n), dtype=np.uint8)
    
    
# reshape and preprocess
caffe_in = raw_data.reshape(n, 1, 28, 28) * 0.00390625 # manually scale data instead of using `caffe.io.Transformer`
out = net.forward_all(data=caffe_in) # Does this work if remove TEST layer in prototxt?


feat = out['feat']
f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
     '#ff00ff', '#990000', '#999900', '#009900', '#009999']
for i in range(10):
    plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.grid()
# plt.show()
plt.ioff()
f.savefig(current_dir + 'images/mnist_siamese_imagedata_%d_itr_embedding.png' % (ITERATION))

