#激励函数
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#inputs 输入值 activation_function 激励函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    #biase 建议值不为0 所以+上0.1
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.add(tf.matmul(inputs,Weights),biases)
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

#线性-1,1  数量300 []维度    300行
x_data=np.linspace(-1,1,300)[:,np.newaxis]
#噪点
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

layer1=add_layer(xs,1,10,activation_function=tf.nn.relu)
predition=add_layer(layer1,10,1,activation_function=None)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),
                   reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess=tf.Session()

init=tf.global_variables_initializer()

sess.run(init)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        predition_value=sess.run(predition,feed_dict={xs:x_data,ys:y_data})
        lines=ax.plot(x_data,predition_value,'r-',lw=10)
        plt.pause(0.5)


