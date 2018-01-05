import os
import numpy as np
import struct
from matplotlib import pyplot


imgNumber = 25  # the number of the image you want to display
path = '/home/feiyang'   # the training set is stored in this directory
fname_img = os.path.join(path, 't10k-images-idx3-ubyte')  # the training set image file path
fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')  # the training set label file path


# open the
with open(fname_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.uint8)

with open(fname_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)


print('The training set contains', len(img), 'images')  # print the number of the
print('The shape of the image is', img[0].shape)  # print the shape of the image
print('The label of the image is', lbl[imgNumber])  # print the label of the image displayed

pyplot.imshow(img[imgNumber], cmap='gray')  # plot the image in "gray" colormap
pyplot.show()

