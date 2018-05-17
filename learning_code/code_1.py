# author: Chen Dengshi
# Date: 2018.5.17
# content: y = 0.1 * x + 0.3 (TensorFlow)
# references: https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-2-example2/
#             https://blog.csdn.net/hq86937375/article/details/79696023

import tensorflow as tf
import numpy as np
import os

# ignore a warning: Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# https://blog.csdn.net/hq86937375/article/details/79696023
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create data (ignore this rand warning)
# numbers'type in tensorflow is float32 in most cases, so we define numbers as float32.
x_data = np.random.rand(100).astype(np.float32)  # create 100 random numbers(float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure begin
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# create tensorflow structure end

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))


