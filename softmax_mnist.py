#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

'''
softmax for mnist 
'''

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_class = 10

# MNIST data image of shape 28*28 = 784
X = tf.placeholder(tf.float32, [None,784])
Y = tf.placeholder(tf.float32, [None,nb_class])

W = tf.Variable(tf.random_normal([784,nb_class]))
b = tf.Variable(tf.random_normal([nb_class]))

# hypoesis
hypoesis = tf.nn.softmax(tf.matmul(X,W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypoesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# test model
is_correct = tf.equal(tf.arg_max(hypoesis, 1), tf.arg_max(Y,1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Train:
#    parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # train cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c,_ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c / total_batch

        print 'Epoch:','%04d'%(epoch+1),'cost=', '{:.9f}'.format(avg_cost)
    # accuracy
    print 'accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})

print 'Learning Finished !'


