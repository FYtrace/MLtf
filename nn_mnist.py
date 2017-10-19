#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

'''
NN for mnist 
'''

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
nb_class = 10

tf.set_random_seed(777)

# MNIST data image of shape 28*28 = 784
X = tf.placeholder(tf.float32, [None,784])
Y = tf.placeholder(tf.float32, [None,nb_class])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([784,256]), name='W1')
b1 = tf.Variable(tf.random_normal([256]), name='b1')
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
# drop out
#L1 = tf.nn.dropout(L1, keep_prob=0.5)

W2 = tf.Variable(tf.random_normal([256,256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
# drop out
#L2 = tf.nn.dropout(L2, keep_prob=0.5)

W3 = tf.Variable(tf.random_normal([256,10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2,W3) + b3

# define cost / loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

is_correct = tf.equal(tf.arg_max(tf.nn.softmax(hypothesis), 1), tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X:batch_xs, Y:batch_ys}
            c,_ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c/total_batch

        print 'Epoch:','%04d'%(epoch+1),'cost=', '{:.9f}'.format(avg_cost)
    # accuracy
    print 'accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})


'''
# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

Epoch: 0001 cost= 141.906161877
Epoch: 0002 cost= 38.011038122
Epoch: 0003 cost= 23.669904509
Epoch: 0004 cost= 16.446176507
Epoch: 0005 cost= 11.833091606
Epoch: 0006 cost= 8.662033729
Epoch: 0007 cost= 6.387574036
Epoch: 0008 cost= 4.664141869
Epoch: 0009 cost= 3.604831241
Epoch: 0010 cost= 2.611421092
Epoch: 0011 cost= 2.018309432
Epoch: 0012 cost= 1.529276167
Epoch: 0013 cost= 1.213648296
Epoch: 0014 cost= 0.897490756
Epoch: 0015 cost= 0.761403872
accuracy: 0.9469

'''

'''
# parameters
learning_rate = 0.1
training_epochs = 15
batch_size = 100

Epoch: 0001 cost= 37.618684440
Epoch: 0002 cost= 1.750353368
Epoch: 0003 cost= 1.717822712
Epoch: 0004 cost= 1.597515878
Epoch: 0005 cost= 1.511162808
Epoch: 0006 cost= 1.806237637
Epoch: 0007 cost= 1.812293498
Epoch: 0008 cost= 1.622250364
Epoch: 0009 cost= 1.559475886
Epoch: 0010 cost= 1.550885480
Epoch: 0011 cost= 1.561158639
Epoch: 0012 cost= 1.600426292
Epoch: 0013 cost= 1.785046338
Epoch: 0014 cost= 1.696188058
Epoch: 0015 cost= 1.836739351
accuracy: 0.2727

'''

'''
# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
nb_class = 10

tf.set_random_seed(777)

Epoch: 0001 cost= 143.093838038
Epoch: 0002 cost= 39.032787415
Epoch: 0003 cost= 24.338254912
Epoch: 0004 cost= 16.938986932
Epoch: 0005 cost= 12.265815377
Epoch: 0006 cost= 9.039960710
Epoch: 0007 cost= 6.772742046
Epoch: 0008 cost= 5.077370779
Epoch: 0009 cost= 3.785548024
Epoch: 0010 cost= 2.879454931
Epoch: 0011 cost= 2.093655378
Epoch: 0012 cost= 1.587276842
Epoch: 0013 cost= 1.267390615
Epoch: 0014 cost= 0.971150772
Epoch: 0015 cost= 0.775061108
accuracy: 0.9457

'''
