import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
from PIL import Image

image_dir = "/auto/research2/vut/caffe-rc5/examples/mnist_images/test_images/"
data_file = "/auto/research2/vut/caffe-rc5/examples/mnist_images/test.txt"
lmdb_file = "/auto/research2/vut/caffe-rc5/examples/mnist/mnist_test_lmdb"

# image_dir = "/auto/research2/vut/caffe-rc5/examples/mnist_images/train_images/"
# data_file = "/auto/research2/vut/caffe-rc5/examples/mnist_images/train.txt"
# lmdb_file = "/auto/research2/vut/caffe-rc5/examples/mnist/mnist_train_lmdb"

lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

with open(data_file, "w+") as f:
  i = 60000
  for key, value in lmdb_cursor:
    # Extract data from lmdb
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    im = data.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0)) # original (dim, col, row)
    im = np.array([l.flatten() for l in im])

    # Save image as jpg
    im_path = image_dir + "mnist_" + str(i) + ".jpg"
    Image.fromarray(im).save(im_path)

    # Add image path and label to text file
    f.write(im_path + " " + str(label) + "\n")

    print "Index: ", i, ", Label: ", label
    i += 1

#     # Testing
#     if i == 10:
#       break