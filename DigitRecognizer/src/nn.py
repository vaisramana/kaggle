
from scipy.misc import imread, imsave, imresize  
import numpy as np

train = pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')
trainImages = (train.ix[:,1:].values).astype('float32')
trainLabels = (train.ix[:,0].values).astype('float32')

def saveToJpeg(idx, images):
    image = images.ix[idx,:]
    imsave("sample"+str(idx)+".jpeg", image)
    
