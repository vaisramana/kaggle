
from scipy.misc import imread, imsave, imresize  
import numpy as np
import pandas as pd


def saveToJpeg(idx, images, labels):
    image = images[idx,:]
    label = labels[idx]
    imsave("sample"+str(idx)+"_num"+str(label)+".jpeg", image.reshape(28,28))


train = pd.read_csv("../input/train.csv")
#test = pd.read_csv('../input/test.csv')
trainImages = (train.ix[:,1:].values).astype('float32')
trainLabels = (train.ix[:,0].values).astype('int32')

saveToJpeg(10,trainImages,trainLabels);
saveToJpeg(100,trainImages,trainLabels);
saveToJpeg(1000,trainImages,trainLabels);
