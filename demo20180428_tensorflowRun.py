import tensorflow as tf
import numpy as np
import os
#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#create data

#定义 begin
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3
#参数变量
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
#初始值
biases=tf.Variable(tf.zeros([1]))
#目标值
y=Weights*x_data+biases
#误差
loss=tf.reduce_mean(tf.square(y-y_data))
#优化器 参数：学习效率0-1
optimizer=tf.train.GradientDescentOptimizer(0.5)
#训练
train=optimizer.minimize(loss)

#init=tf.initialize_all_variables()
init=tf.global_variables_initializer()

##定义 end

sess=tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    print("每次：% s" % step, sess.run(Weights), sess.run(biases))
    if step %20==0:
        print("每20次：% s" % step,sess.run(Weights),sess.run(biases))
