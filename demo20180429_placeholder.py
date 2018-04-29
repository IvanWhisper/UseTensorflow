#传入值
import tensorflow as tf
import os
#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.multiply(input1,input2) #乘法 视频中mul不可用，暂用multiply

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.0]}))