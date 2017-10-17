#!/usr/bin/env python
# coding=utf-8


import tensorflow as tf

x = tf.placeholder(tf.float32,shape=[None])
y = tf.placeholder(tf.float32,shape=[None])

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

# our hypothesis XW + b
hypothesis = x*W + b
# cost function
cost = tf.reduce_mean(tf.square(hypothesis - y))
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session

sess = tf.Session()
# Initializes global varibles in the graph
sess.run(tf.global_variables_initializer())

# fit the line
for step in range(0,2001):
    cost_val,W_val,b_val,_ = sess.run([cost, W, b, train], feed_dict={x:[1,2,3],y:[1,2,3]})
    if step%20 == 0:
        print(step, cost_val, W_val, b_val)

