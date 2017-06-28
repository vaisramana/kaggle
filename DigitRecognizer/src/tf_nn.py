
from scipy.misc import imread, imsave, imresize  
import numpy as np
import pandas as pd
import tensorflow as tf


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




TOTAL_TRAIN_SAMPLE_SIZE = 42000
VALIDATION_SAMPLE_SIZE = 12000
TRAIN_SAMPLE_SIZE = (TOTAL_TRAIN_SAMPLE_SIZE-VALIDATION_SAMPLE_SIZE)
SAMPLE_SIZE_PER_BATCH = 200
BATCH_NUMBER = (TRAIN_SAMPLE_SIZE/SAMPLE_SIZE_PER_BATCH)

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

saveToJpeg(11,trainImages,trainLabels);
saveToJpeg(110,trainImages,trainLabels);
saveToJpeg(1100,trainImages,trainLabels);

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# forward propagation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#train
for i in range(BATCH_NUMBER):
    if i%10 == 0:
        print "Run Epoch {0}".format(i)
    batch_xs = trainImages[i:(i+1)*SAMPLE_SIZE_PER_BATCH,:]
    batch_ys = trainLabels[i:(i+1)*SAMPLE_SIZE_PER_BATCH,:]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%10 == 0:
        print "Epoch {0}: accuracy {1}".format(
                    i, sess.run(accuracy, feed_dict={x: validateImages, y_: validateLabels}))

# Test trained model

