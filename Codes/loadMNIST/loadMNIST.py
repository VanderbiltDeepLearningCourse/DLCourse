import os
import numpy as np
import struct
from matplotlib import pyplot


imgNumber = 25  # choose the which image you want to display
path = '/home/feiyang/'   # the training set is stored in this directory
fname_img = os.path.join(path, 'train-images-idx3-ubyte')  # the training set image file path
fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')  # the training set label file path


# open the label file and load it to the "lbl"
with open(fname_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.uint8)

# open the image file and load it to the "img"
with open(fname_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)


print('The training set contains', len(img), 'images')  # print the how many images contained in the training set
print('The shape of the image is', img[0].shape)  # print the shape of the image
print('The label of the image is', lbl[imgNumber])  # print the label of the image displayed

pyplot.imshow(img[imgNumber], cmap='gray')  # plot the image in "gray" colormap
pyplot.show()

