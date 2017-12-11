import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


import os
os.chdir('F:\\#ARL\\drive-download-20171101T170526Z-001')


#import data
data=pd.read_csv('lily.csv')
data.columns = ['f1','f2','f3','f4','f5']

print(data.head())

data["f5"].value_counts()

#map data into arrays
s=np.asarray([1,0,0])
ve=np.asarray([0,1,0])
vi=np.asarray([0,0,1])
data['f5'] = data['f5'].map({'I. setosa': s, 'I. versicolor': ve,'I. virginica':vi})

print(data.head())

#shuffle the data
data=data.iloc[np.random.permutation(len(data))]


features = data.drop("f5",axis = 1)
features.head()

labels = data["f5"]
labels.head()

data=data.reset_index(drop=True)

from sklearn.cross_validation import train_test_split

x_input,x_test,y_input,y_test = train_test_split(features,labels,train_size = .60)


#placeholders and variables. input has 4 features and output has 3 classes
x=tf.placeholder(tf.float32,shape=[None,4])
y_=tf.placeholder(tf.float32,shape=[None, 3])
#weight and bias
W=tf.Variable(tf.zeros([4,3]))
b=tf.Variable(tf.zeros([3]))
# model
#softmax function for multiclass classification
y = tf.nn.softmax(tf.matmul(x, W) + b)
#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#optimiser -
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#calculating accuracy of our model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#session parameters
sess = tf.InteractiveSession()
#initialising variables
init = tf.initialize_all_variables()
sess.run(init)
#number of interations
epoch=2000

for step in range(epoch):
   _, c=sess.run([train_step,cross_entropy], feed_dict={x: x_input, y_:[t for t in y_input.as_matrix()]})
   if step%500==0 :
       print (c)

#random testing
a=data.loc[130,['f1','f2','f3','f4']]
b=a.reshape(1,4)
largest = sess.run(tf.arg_max(y,1), feed_dict={x: b})[0]
if largest==0:
    print ("flower is :I. setosa")
elif largest==1:
    print ("flower is :I. versicolor")
else :
    print ("flower is :I. virginica")
    
# Accuracy
print('Accuracy\n')
print (sess.run(accuracy,feed_dict={x: x_test, y_:[t for t in y_test.as_matrix()]}))


'''
output
f1   f2   f3   f4         f5
0  5.1  3.5  1.4  0.2  I. setosa
1  4.9  3.0  1.4  0.2  I. setosa
2  4.7  3.2  1.3  0.2  I. setosa
3  4.6  3.1  1.5  0.2  I. setosa
4  5.0  3.6  1.4  0.3  I. setosa
    f1   f2   f3   f4         f5
0  5.1  3.5  1.4  0.2  [1, 0, 0]
1  4.9  3.0  1.4  0.2  [1, 0, 0]
2  4.7  3.2  1.3  0.2  [1, 0, 0]
3  4.6  3.1  1.5  0.2  [1, 0, 0]
4  5.0  3.6  1.4  0.3  [1, 0, 0]
1.09861
0.140238
0.0848137
0.0651702
flower is :I. setosa
Accuracy

0.983333
'''





