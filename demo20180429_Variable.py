#变量
import tensorflow as tf
import os
#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

state=tf.Variable(0,name='counter')
#print(state.name)
one=tf.constant(1)

new_value=tf.add(state,one)
update=tf.assign(state,new_value)
#初始化变量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
#print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
