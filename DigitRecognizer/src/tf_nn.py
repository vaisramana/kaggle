# one layer neural network which archive maximum 90% accuracy

 
import tensorflow as tf
import dataLoader as dl


SAMPLE_SIZE_PER_BATCH = 100
BATCH_NUMBER = (dl.TRAIN_SAMPLE_SIZE/SAMPLE_SIZE_PER_BATCH)

(trainImages,trainLabels,validateImages,validateLabels,testImages) = dl.dataLoader()

dl.saveToJpeg(11,trainImages,trainLabels)
dl.saveToJpeg(110,trainImages,trainLabels)
dl.saveToJpeg(1100,trainImages,trainLabels)

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


