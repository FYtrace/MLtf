#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

'''
NN for mnist 
'''

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# input placeholders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1,28,28,1])  
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape = (?,  28, 28, 1)
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
# conv   -> (?, 28, 28, 32)
# pool   -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

# L2 ImgIn shape = (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
# Conv   -> (?, 14, 14, 64)
# pool   -> (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.reshape(L2, [-1, 7*7*64])

# final FC 7*7*64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b

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
        feed_dict = {X:batch_xs, Y:batch_ys}
        c, _= sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
    print 'Epoch:','%04d'%(epoch+1),'cost=','{:.9f}'.format(avg_cost)

print 'Learning Finished !'

# test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print 'Accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})


'''
# parameters
learning_rate = 0.001
no drop out

Epoch: 0001 cost= 0.345597459
Epoch: 0002 cost= 0.091797432
Epoch: 0003 cost= 0.068372396
Epoch: 0004 cost= 0.056379004
Epoch: 0005 cost= 0.046920019
Epoch: 0006 cost= 0.041171568
Epoch: 0007 cost= 0.036644606
Epoch: 0008 cost= 0.032853815
Epoch: 0009 cost= 0.028126524
Epoch: 0010 cost= 0.024791351
Epoch: 0011 cost= 0.022118081
Epoch: 0012 cost= 0.020218051
Epoch: 0013 cost= 0.017048809
Epoch: 0014 cost= 0.015361157
Epoch: 0015 cost= 0.013110995
Learning Finished !
Accuracy: 0.9879

'''
