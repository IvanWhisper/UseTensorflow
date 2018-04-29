#分类
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os
#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

digits=load_digits()
X=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name='layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            #biase 建议值不为0 所以+上0.1
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(layer_name + '/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.add(tf.matmul(inputs,Weights),biases)
            Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob)
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

keep_prob=tf.placeholder(tf.float32)

xs=tf.placeholder(tf.float32,[None,64])#8*8
ys=tf.placeholder(tf.float32,[None,10])

layer1=add_layer(xs,64,50,n_layer='l1',activation_function=tf.nn.tanh)

prediction=add_layer(layer1,50,10,n_layer='l2',activation_function=tf.nn.softmax)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                            reduction_indices=[1])) #loss
tf.summary.scalar('loss',cross_entropy)

train_step=tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

with tf.Session() as sess:
    merged=tf.summary.merge_all()
    train_writer=tf.summary.FileWriter(".\\logs\\train",sess.graph)
    test_writer = tf.summary.FileWriter(".\\logs\\test", sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5})
        if i%50==0:
            train_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
            test_result=sess.run(merged, feed_dict={xs: X_test, ys: y_test,keep_prob:1})
            train_writer.add_summary(train_result,i)
            test_writer.add_summary(test_result,i)