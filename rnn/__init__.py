#coding=gbk
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#��������
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#������
lr=0.001
training_iters=100000   #train step ����
batch_size=128
n_inputs=28
n_step=28               #time step
n_hidden_units=128      #hidden layer ��Ԫ
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

#��������ṹ
def RNN(x,weights,biases):
    #ԭʼ��X��3ά���ݣ���Ҫ�������2ά���ݲ���ʹ��weights�ľ���˷�
    x=tf.reshape(x, [-1,n_inputs])
    x_in=tf.matmul(x, weights['in'])+biases['in']
    x_in=tf.reshape(x_in,shape=[-1,n_step,n_hidden_units])
    
    #cell
    #lstm cell ��Ϊ���߾���ͷ��߾���  c_state��m_state
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=0.8,state_is_tuple=True)
    init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)
    #1�Ƿ��߾���Ľ�������������outputs[-1]
    #outputs��¼��ÿ��ʱ����֧�������final_state[1]ֻ�����һ��ʱ���ķ�֧���
    results=tf.matmul(final_state[1], weights['out'])+biases['out']
    
    ''' 
    outputs��tensor����Ҫ�⿪��Ϊlist  final_state�����һ��output
            �� outputs ��� �б� [(batch, outputs)..] * steps
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))     #[1,0,2]�Ǹ���tfҪ��η�ת���е���ά����������ԭ�е�������[0,1,2]��ά��˳��ʹ��tf.transpose�Ὣ[0,1,2]��0��1ά���ݻ���ά�� 
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    #ѡȡ���һ�� output 
    '''
    return results

pred=RNN(x, weights, biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train=tf.train.AdamOptimizer(lr).minimize(cost)

#����
correct_pred=tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))#һ�������ֵ���±�
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