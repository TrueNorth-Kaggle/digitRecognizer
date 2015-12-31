
# coding: utf-8

# In[116]:

# This model is highly based on the MNIST 
# tutorial provided by TensorFlow with MNIST dataset 
# for Kaggle Competition

import tensorflow as tf
import numpy as np


# In[115]:

# getData is used to get train and test sets 
# CorssValidation set is to be added
os.chdir("../scripts/")
import getData as gd


# In[ ]:

# Use the following to reload getData.py if it is changed
# os.chdir("../scripts/")
# reload(getData)


# In[107]:

# Currently manual partition the train and test sets
train_num = 30000
test_min = 30001
test_max = 35000

train_x, train_y = gd.getTrain(train_num)
test_x, test_y = gd.getTest(test_min, test_max)


# In[108]:

# A softmax regression containing only one layer of 
# Network. The regression is in form of y = softmax(Wx + b)
# W is the weigt and b is the bias 

# First we need a placeholder for each input value x
x = tf.placeholder(tf.float32, [None, 784])

# Then we decalre W and b as Variables for model parameters
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# In[109]:

# Implementation of the model 
y = tf.nn.softmax(tf.matmul(x, W) + b)


# In[110]:

# Training step: tell TensorFlow what makes up a good model
# We use corss-entropy as the cost function 
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Backgpropagation with selected optimizer algorithm
# There are a variety of optimizers to choose from
# GradientSecentOpimizer will fail to converge if the batch size exceeds 200
# Alternative solution is to use AdagradOptimizer, AdamOptimizer ...
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# In[111]:

# Launch the model 
init = tf.initialize_all_variables()

# All computations won't start until the session is initilized
sess = tf.Session()
sess.run(init)


# In[112]:

# Using stochastic fitting 
# Good to have visulization to find the best learning rate 
for i in range(100):
    batch_xs, batch_ys = train_x[i*(100):(i+1)*100], train_y[i*(100):(i+1)*100]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# In[113]:

# Correction_prediction will result in a list of boolean in form of 
# [True, True, False, Flase ...] True if prediction matches the true label
# Flase otherwise
correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[114]:

# Accuracy on the test set
print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))


# In[ ]:




# In[ ]:




# In[ ]:



