#!/usr/bin/env python
# coding=utf-8

'''
softmax for mutil classify
'''

import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder("float", [None,4])
Y = tf.placeholder("float", [None,3])

nb_class = 3
W = tf.Variable(tf.random_normal([4,nb_class]), name='Weight')
b = tf.Variable(tf.random_normal([nb_class]), name='bias')

# tf.nn.softmax computes softmaxt activations
hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)

# cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# lauch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y:y_data})
        if step%200 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

