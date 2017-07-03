import numpy as np
import tensorflow as tf 
import dataLoader as dl



def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))

def model(X, w1, w2, w3, inputLayerDropoutProbability, hiddenLayerDropoutProbability):
    X = tf.nn.dropout(X, inputLayerDropoutProbability)
    y1 = tf.nn.relu(tf.matmul(X, w1))

    y1 = tf.nn.dropout(y1, hiddenLayerDropoutProbability)
    y2 = tf.nn.relu(tf.matmul(y1, w2))

    y2 = tf.nn.dropout(y2, hiddenLayerDropoutProbability)

    return tf.matmul(y2, w3)


SAMPLE_SIZE_PER_BATCH = 100

(trainImages,trainLabels,validateImages,validateLabels,testImages) = dl.dataLoader()

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])


w1 = init_weights([784, 625])
w2 = init_weights([625, 625])
w3 = init_weights([625, 10])
"""
w1 = tf.Variable(tf.zeros([784, 625]))
w2 = tf.Variable(tf.zeros([625, 625]))
w3 = tf.Variable(tf.zeros([625, 10]))
"""

# dropout parameter
inputLayerDropoutProbability = tf.placeholder("float")
hiddenLayerDropoutProbability = tf.placeholder("float")


estimatedY = model(X, w1, w2, w3, inputLayerDropoutProbability, hiddenLayerDropoutProbability)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=estimatedY))
#train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
#why gradient descent doesn't work? learning rate 0.5 is far too large in this case.
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
predict_op = tf.argmax(estimatedY, 1)

with tf.Session() as sess:

    tf.global_variables_initializer().run()

    for i in xrange(1000):
        for start, end in zip(range(0, len(trainImages), SAMPLE_SIZE_PER_BATCH), range(SAMPLE_SIZE_PER_BATCH, len(trainImages), SAMPLE_SIZE_PER_BATCH)):
            sess.run(train_op, feed_dict = {X: trainImages[start:end], 
                                            Y: trainLabels[start:end],
                                            inputLayerDropoutProbability: 0.8, 
                                            hiddenLayerDropoutProbability: 0.5})
        print i, np.mean(np.argmax(validateLabels, axis = 1) == sess.run(predict_op, 
                        feed_dict = {X: validateImages,
                                     inputLayerDropoutProbability: 1.0, 
                                     hiddenLayerDropoutProbability: 1.0}))


    # apply in test data
    testLabels = sess.run(predict_op, feed_dict = {X: testImages, 
                                                   inputLayerDropoutProbability: 1.0, 
                                                   hiddenLayerDropoutProbability: 1.0})
    
print testLabels[2]
print testLabels[22]
print testLabels[222]
print testLabels[2222]
print testLabels[22222]
dl.saveToJpeg(2,testImages,testLabels)
dl.saveToJpeg(22,testImages,testLabels)
dl.saveToJpeg(222,testImages,testLabels)
dl.saveToJpeg(2222,testImages,testLabels)
dl.saveToJpeg(22222,testImages,testLabels)

dl.predictionSaver(testLabels)
    