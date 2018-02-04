# ==================================================================================== #
# Convert caffe siamese mnist data (leveldb) to jpg images
#
# Author: Thanh Vu 
# Date: Feb 3, 2018
# ==================================================================================== #

import numpy as np
import random
from random import getrandbits as randBool


def rand_triplet_paths(img_paths_by_label):
  labels = img_paths_by_label.keys()
  label, label_neg = random.sample(labels, 2)

  anchor_path = random.choice(img_paths_by_label[label])
  pos_path = random.choice(img_paths_by_label[label])
  neg_path = random.choice(img_paths_by_label[label_neg])

  return (anchor_path, pos_path, neg_path)

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

source = '/auto/research2/vut/thesis/CaffeLeNetMNIST/train.txt'
img_paths_by_label = readImagePaths(source)
n = 10000
with open('/auto/research2/vut/thesis/CaffeLeNetMNIST/triplet/triplet_train.txt','w+') as f:
  for i in range(n):
    anchor_path, pos_path, neg_path = rand_triplet_paths(img_paths_by_label)
    f.write(anchor_path +' '+ pos_path +' '+ neg_path +'\n')