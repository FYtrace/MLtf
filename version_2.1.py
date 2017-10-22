#!/usr/bin/env python
# coding=utf-8

'''
This script shows how to predict stock prices using a basic RNN

20171022: sort out the code

'''
import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def restore2price(data, price):
    denominator = np.max(data,0) - np.min(data,0)
    price = (price+1e-7) * denominator
    price = price + np.min(data,0)

    return np.array(price[:,np.array(denominator.shape)[0]-1])

def loadData(datapath):
    xy = np.loadtxt(datapath, delimiter=',')

    return xy

def buildDataSet(x, y, seq_length):
    # build a dataset
    dataX = []
    dataY = []
    for i in range(0, len(y) - seq_length):
        _x = x[i:i + seq_length]
        _y = y[i + seq_length]  # Next close price
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)

    # train/test split
    train_size = int(len(dataY) * 0.7)
    #test_size = len(dataY) - train_size
    trainX, testX = np.array(dataX[0:train_size]), np.array(
        dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(
        dataY[train_size:len(dataY)])

    return trainX,testX,trainY,testY

def writeFile(filename, test_predict, testY):
    # write to file 
    file = open(filename, 'w+')
    for i in range(1,len(np.array(testY))):
        file.write(str(100.0*(test_predict[i]-testY[i-1])/testY[i-1]))
        file.write("    ")
        file.write(str(100.0*(testY[i]-testY[i-1])/testY[i-1]))
        file.write("    ")
        file.write(str(test_predict[i]))
        file.write("    ")
        file.write(str(testY[i]))
        file.write('\n')
    file.close()

    return 

# train parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        #with tf.variable_scope(self.name):
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])

        # build a LSTM network
        self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
        self.outputs, _states = tf.nn.dynamic_rnn(self.cell, self.X, dtype=tf.float32)

        self.Y_pred = tf.contrib.layers.fully_connected(
        self.outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

        # cost/loss
        self.loss = tf.reduce_sum(tf.square(self.Y_pred - self.Y))  # sum of the squares
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
        # RMSE
        self.targets = tf.placeholder(tf.float32, [None, 1])
        self.predictions = tf.placeholder(tf.float32, [None, 1])
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.targets - self.predictions)))

    def train(self, trainX, trainY):
        return sess.run([self.loss,self.optimizer], feed_dict={self.X: trainX, self.Y: trainY})

    def predict(self, testX, testY):
        # Test step
        test_predict = sess.run(self.Y_pred, feed_dict={self.X: testX})
        rmse_val = sess.run(self.rmse, feed_dict={self.targets: testY, self.predictions: test_predict})
        print("RMSE: {}".format(rmse_val))

        return test_predict
    

# load data
xy = loadData('./data/data-02-stock_daily.csv')
#xy = xy[::-1]  # reverse order (chronically ordered)
#xy = xy[:,2:]
x = MinMaxScaler(xy)
y = x[:, [-1]]  # Close as label

# build dataset
trainX,testX,trainY,testY = buildDataSet(xy,y,seq_length)

# get a session
sess = tf.Session()
m1 = Model(sess, 'm1')

sess.run(tf.global_variables_initializer())

# train Model
for i in range(iterations):
    step_loss,_ = m1.train(trainX, trainY)
    print("[step: {}] loss: {}".format(i, step_loss))

# predict
test_predict = m1.predict(testX,testY)

print trainX.shape

# Plot predictions
#plt.plot(testY)
#plt.plot(test_predict)
print test_predict.shape
testY = restore2price(xy, testY)
test_predict = restore2price(xy, test_predict)
plt.plot(testY)
plt.plot(test_predict)

# writeFile
writeFile('out.out',test_predict,testY)

plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.show()
