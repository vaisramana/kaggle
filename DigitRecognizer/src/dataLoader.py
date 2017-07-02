
import numpy as np
import pandas as pd
from scipy.misc import imread, imsave, imresize 
import csv

TOTAL_TRAIN_SAMPLE_SIZE = 42000
VALIDATION_SAMPLE_SIZE = 12000
TRAIN_SAMPLE_SIZE = (TOTAL_TRAIN_SAMPLE_SIZE-VALIDATION_SAMPLE_SIZE)

def saveToJpeg(idx, images, labels=None):
    """images is N*784 matrix and labels is N*10*1 matrix"""
    image = images[idx,:]
    if labels.size == 0:
        label = "?"
    else:
        #vectorized result
        if labels.ndim == 2:
            labelVector = labels[idx]
            for i in range(10):
                if labelVector[i] != 0:
                    label = i
        elif labels.ndim == 1:
            label = labels[idx]
        else:
            print "invalid dimension"
            return 
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
    test = pd.read_csv('../input/test.csv')
    trainImages = (train.ix[0:(TRAIN_SAMPLE_SIZE-1),1:].values).astype('float32')
    trainOriginalLabels = (train.ix[0:(TRAIN_SAMPLE_SIZE-1),0].values).astype('int32')
    #numpy.ndarray to list to numpy.ndarray (35000, 10, 1) to numpy.ndarray (35000, 10) 
    trainLabels = (np.array([vectorized_result(label) for label in trainOriginalLabels])).reshape(TRAIN_SAMPLE_SIZE,10)

    validateImages = (train.ix[TRAIN_SAMPLE_SIZE:,1:].values).astype('float32')
    validateOriginalLabels = (train.ix[TRAIN_SAMPLE_SIZE:,0].values).astype('int32')
    #numpy.ndarray to list to numpy.ndarray 
    validateLabels = (np.array([vectorized_result(label) for label in validateOriginalLabels])).reshape(VALIDATION_SAMPLE_SIZE,10)

    testImages = (test.values).astype('float32')
    
    return (trainImages,trainLabels,validateImages,validateLabels,testImages)

def predictionSaver(prediction):
    # write header 
    """
    csvFile = open("prediction.csv",'w')
    writer = csv.writer(csvFile)
    writer.writerow(["ImageId","ImageId"])
    csvFile.close()
    """
    output = np.zeros((prediction.shape[0],2))
    for i in range(prediction.shape[0]):
        output[i] = [i,prediction[i]]
        
    # write data
    np.savetxt("prediction.csv",output,fmt='%d',delimiter=',',header="ImageId,Label",comments='')

    