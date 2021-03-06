#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

'''
Deep CNN for mnist
'''

mnist = input_data.read_data_sets("MNIST_data/", one_hot='True')

# parameters
tf.set_random_seed(777)
learning_rate = 0.001
training_epochs = 15
batch_size = 100

keep_prob = tf.placeholder(tf.float32)

# train data
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape = (?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev = 0.01))
# conv1    -> (?, 28, 28, 32)
# pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# L2 ImgIn shape = (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev = 0.01))
# conv2    -> (?, 14, 14, 64)
# pool2    -> (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# L3 ImgIn shape = (?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev = 0.01))
# conv3    -> (?, 7, 7, 128)
# pool3    -> (?, 4, 4, 128)
# Reshape  -> (?, 4*4*128)  for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1,4*4*128])

# L4 FC 4*4*128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128*4*4, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# L5 FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625,10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5

# define cost & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print 'Learning started. It takes sometime. '
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs, Y:batch_ys, keep_prob:0.7}
        c, _= sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
    print 'Epoch:','%04d'%(epoch+1),'cost=','{:.9f}'.format(avg_cost)

print 'Learning Finished !'

# test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print 'Accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1})




'''

Epoch: 0001 cost= 0.384751925
Epoch: 0002 cost= 0.094981200
Epoch: 0003 cost= 0.068119203
Epoch: 0004 cost= 0.058526521
Epoch: 0005 cost= 0.052847991
Epoch: 0006 cost= 0.046346503
Epoch: 0007 cost= 0.039445299
Epoch: 0008 cost= 0.038612851
Epoch: 0009 cost= 0.035883930
Epoch: 0010 cost= 0.033090527
Epoch: 0011 cost= 0.031622195
Epoch: 0012 cost= 0.029686315
Epoch: 0013 cost= 0.028285102
Epoch: 0014 cost= 0.026164962
Epoch: 0015 cost= 0.026030481
Learning Finished !
Accuracy: 0.9933

'''
