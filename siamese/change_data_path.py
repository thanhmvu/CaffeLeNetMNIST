# ==================================================================================== #
# Update the paths of data files
#
# Author: Thanh Vu 
# Date: Feb 3, 2018
# ==================================================================================== #
import sys


def main(): 
  curr_dir = "/auto/research2/vut/thesis/CaffeLeNetMNIST/siamese/"
  SRC_FILENAME = curr_dir + "test0_2.txt" 
  DST_FILENAME = curr_dir + "test_2.txt" 
  
  OLD_DIR = "/auto/research2/vut/thesis/CaffeLeNetMNIST/siamese/test_images/"
  NEW_DIR = "/auto/research2/vut/thesis/CaffeLeNetMNIST/data/siamese/test_images/"
  
  with open(SRC_FILENAME) as infile:
    src_data = infile.readlines()
    
  with open(DST_FILENAME, "w+") as outfile:
    for i, line in enumerate(src_data):
      outfile.write(line.replace(OLD_DIR, NEW_DIR))
      sys.stdout.write('Finished %d/%d lines\r' % (i+1, len(src_data)))
      sys.stdout.flush()
    print

    
if __name__ == "__main__":
    main()


