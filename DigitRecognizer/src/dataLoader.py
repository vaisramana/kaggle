
import numpy as np
import pandas as pd
from scipy.misc import imread, imsave, imresize 

TOTAL_TRAIN_SAMPLE_SIZE = 42000
VALIDATION_SAMPLE_SIZE = 12000
TRAIN_SAMPLE_SIZE = (TOTAL_TRAIN_SAMPLE_SIZE-VALIDATION_SAMPLE_SIZE)

def saveToJpeg(idx, images, labels):
    """images is N*784 matrix and labels is N*10*1 matrix"""
    image = images[idx,:]
    labelVector = labels[idx]
    for i in range(10):
        if labelVector[i] != 0:
            label = i
    imsave("sample"+str(idx)+"_num"+str(label)+".jpeg", image.reshape(28,28))


def vectorized_result(idx):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[idx] = 1.0
    return e

    
def dataLoader():
    train = pd.read_csv("../input/train.csv")
    #test = pd.read_csv('../input/test.csv')
    trainImages = (train.ix[0:(TRAIN_SAMPLE_SIZE-1),1:].values).astype('float32')
    trainOriginalLabels = (train.ix[0:(TRAIN_SAMPLE_SIZE-1),0].values).astype('int32')
    #numpy.ndarray to list to numpy.ndarray (35000, 10, 1) to numpy.ndarray (35000, 10) 
    trainLabels = (np.array([vectorized_result(label) for label in trainOriginalLabels])).reshape(TRAIN_SAMPLE_SIZE,10)

    validateImages = (train.ix[TRAIN_SAMPLE_SIZE:,1:].values).astype('float32')
    validateOriginalLabels = (train.ix[TRAIN_SAMPLE_SIZE:,0].values).astype('int32')
    #numpy.ndarray to list to numpy.ndarray 
    validateLabels = (np.array([vectorized_result(label) for label in validateOriginalLabels])).reshape(VALIDATION_SAMPLE_SIZE,10)
    
    return (trainImages,trainLabels,validateImages,validateLabels)

