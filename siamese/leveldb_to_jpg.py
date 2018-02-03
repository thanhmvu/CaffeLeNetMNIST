# ==================================================================================== #
# Convert caffe siamese mnist data (leveldb) to jpg images
#
# Author: Thanh Vu 
# Date: Feb 3, 2018
#
# References:
#   + Read leveldb in Python: http://research.beenfrog.com/code/2015/03/28/read-leveldb-lmdb-for-caffe-with-python.html
# ==================================================================================== #
import caffe
import leveldb
from caffe.proto import caffe_pb2
from PIL import Image

PHASE = 'test' #'train'
curr_dir = '/auto/research2/vut/thesis/CaffeLeNetMNIST/siamese/'
leveldb_file = '/auto/research2/vut/caffe-rc5/examples/siamese/mnist_siamese_'+ PHASE +'_leveldb'

db = leveldb.LevelDB(leveldb_file)
datum = caffe_pb2.Datum()

f1 = open(curr_dir + PHASE + '_1.txt', 'w+')
f2 = open(curr_dir + PHASE + '_2.txt', 'w+')

with open(curr_dir + PHASE + '.txt', 'w+') as f:
  i = 0
  for key, value in db.RangeIter():
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)

    im1 = data[0]
    im2 = data[1]
    
    # Save image as jpg
    im1_path = curr_dir + PHASE + '_images/' + PHASE + '_' + str(i) + '_a.jpg'
    im2_path = curr_dir + PHASE + '_images/' + PHASE + '_' + str(i) + '_b.jpg'
    Image.fromarray(im1).save(im1_path)
    Image.fromarray(im2).save(im2_path)

    # Add image path and label to text file
    f.write(im1_path + ' ' + im2_path + ' ' + str(label) + '\n')
    f1.write(im1_path + ' ' + str(label) + '\n')
    f2.write(im2_path + ' ' + str(label) + '\n')

    print 'Index: ', i, ', Label: ', label
    i += 1
    
#     if i == 3:
#       break
          
f1.close()
f2.close()