#coding=gbk
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#导入数据
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#超参数
lr=0.001
training_iters=100000   #train step 上限
batch_size=128
n_inputs=28
n_step=28               #time step
n_hidden_units=128      #hidden layer 神经元
n_classes=10      

#weight biases
x=tf.placeholder(tf.float32, [None,n_step,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])
weights={
         #shape(28,128)
         'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
         #shape(128,10)
         'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
         }
biases={
        #shape(128,)
        'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
        #shape(10,)
        'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
        }

#定义主体结构
def RNN(x,weights,biases):
    #原始的X是3维数据，需要把它变成2维数据才能使用weights的矩阵乘法
    x=tf.reshape(x, [-1,n_inputs])
    x_in=tf.matmul(x, weights['in'])+biases['in']
    x_in=tf.reshape(x_in,shape=[-1,n_step,n_hidden_units])
    
    #cell
    #lstm cell 分为主线剧情和分线剧情  c_state和m_state
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=0.8,state_is_tuple=True)
    init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)
    #1是分线剧情的结果，在这里就是outputs[-1]
    #outputs记录了每个时间点分支的输出，final_state[1]只是最后一个时间点的分支输出
    results=tf.matmul(final_state[1], weights['out'])+biases['out']
    
    ''' 
    outputs是tensor，需要解开成为list  final_state是最后一个output
            把 outputs 变成 列表 [(batch, outputs)..] * steps
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))     #[1,0,2]是告诉tf要如何翻转现有的三维向量，假设原有的张量是[0,1,2]的维度顺序，使用tf.transpose会将[0,1,2]的0和1维数据互换维度 
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    #选取最后一个 output 
    '''
    return results

pred=RNN(x, weights, biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train=tf.train.AdamOptimizer(lr).minimize(cost)

#评估
correct_pred=tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))#一行中最大值的下标
accurary=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    step=0
    while step*batch_size<training_iters:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape([batch_size,n_step,n_inputs])
        sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        if step % 20==0:
            print(sess.run(accurary,feed_dict={x:batch_xs,y:batch_ys}))
        step+=1