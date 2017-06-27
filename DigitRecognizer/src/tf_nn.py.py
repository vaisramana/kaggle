
from scipy.misc import imread, imsave, imresize  
import numpy as np
import pandas as pd
import tensorflow as tf


def saveToJpeg(idx, images, labels):
    image = images[idx,:]
    label = labels[idx]
    imsave("sample"+str(idx)+"_num"+str(label)+".jpeg", image.reshape(28,28))






TOTAL_TRAIN_SAMPLE_SIZE = 42000
VALIDATION_SAMPLE_SIZE = 7000
TRAIN_SAMPLE_SIZE = (TOTAL_TRAIN_SAMPLE_SIZE-VALIDATION_SAMPLE_SIZE)
SAMPLE_SIZE_PER_BATCH = 100
BATCH_NUMBER = (TRAIN_SAMPLE_SIZE/SAMPLE_SIZE_PER_BATCH)

train = pd.read_csv("../input/train.csv")
#test = pd.read_csv('../input/test.csv')
trainImages = (train.ix[0:(TRAIN_SAMPLE_SIZE-1),1:].values).astype('float32')
trainLabels = (train.ix[0:(TRAIN_SAMPLE_SIZE-1),0].values).astype('int32')
validateImages = (train.ix[TRAIN_SAMPLE_SIZE:,1:].values).astype('float32')
validateLabels = (train.ix[TRAIN_SAMPLE_SIZE:,0].values).astype('int32')

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
    batch_xs = trainImages[i:((i+1)*SAMPLE_SIZE_PER_BATCH-1),:]
    batch_ys = trainLabels[i:((i+1)*SAMPLE_SIZE_PER_BATCH-1)]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%10 == 0:
        print("tainning accuracy: ",
                sess.run(accuracy, feed_dict={x: validateImages, y_: validateLabels}))

# Test trained model

