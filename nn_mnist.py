#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

'''
NN for mnist 
'''

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_class = 10

# MNIST data image of shape 28*28 = 784
X = tf.placeholder(tf.float32, [None,784])
Y = tf.placeholder(tf.float32, [None,nb_class])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([784,256]), name='W1')
b1 = tf.Variable(tf.random_normal([256]), name='b1')
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.random_normal([256,256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)

W3 = tf.Variable(tf.random_normal([256,10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2,W3) + b3

# define cost / loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(tf.nn.softmax(hypothesis), 1), tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c,_ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c/total_batch

        print 'Epoch:','%04d'%(epoch+1),'cost=', '{:.9f}'.format(avg_cost)
    # accuracy
    print 'accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})
