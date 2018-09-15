import sys
import tensorflow as tf
sys.path.append('../..')
from Layers.Affine import *
from Layers.Convolution import *
from Layers.Pooling import *
from Dataset.Mnist.loader import *


# net struction
with tf.name_scope('Data_input'):
    data = tf.placeholder(dtype=tf.float32)
    label = tf.placeholder(dtype=tf.float32)

conv1 = Convolution('conv1',data,3,3,1,8,1,1,'VALID',tf.nn.relu)
pooling1 = Pooling('pooling1',conv1,2,2,2,2)
conv2 = Convolution('conv2',pooling1,5,5,8,16,1,1,'VALID',tf.nn.relu)
pooling2 = Pooling('pooling2',conv2,2,2,1,1)

with tf.name_scope('FullConnection'):
    fullconnection = tf.reshape(pooling2,(-1,16*8*8),'FullConnection')

affine1 = Affine('Affine1',fullconnection,16*8*8,100,tf.nn.relu,0.5)
affine2 = Affine('Affine2',affine1,100,10)
# net struction end

#loss
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = affine2,labels = label))

# train_method
train = tf.train.AdamOptimizer(0.01).minimize(loss)

#saver
saver = tf.train.Saver()

# Session
sess = tf.Session()

# record
summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./log',sess.graph)
####################    END     #########################

epochs = 100
batch = 100
init = tf.global_variables_initializer()
sess.run(init)

train_images = Train_images_load('../../DataSet/Mnist/train-images.idx3-ubyte')
train_labels = Train_labels_load('../../DataSet/Mnist/train-labels.idx1-ubyte')
train_images = train_images
print('Data Loaded!')

shuffle_train_set = np.zeros_like(train_images)
shuffle_train_labels = np.zeros_like(train_labels)
for epoch in range(epochs):

    shuffle = np.arange(train_images.shape[0])
    np.random.shuffle(shuffle)
    for k in range(train_images.shape[0]):
        shuffle_train_set[k] = train_images[shuffle[k]]
        shuffle_train_labels[k] = train_labels[shuffle[k]]

    for batch_num in range(int(60000/batch)):
        sess.run(train,feed_dict={data:shuffle_train_set[batch*batch_num:batch*batch_num+batch,:].reshape((-1,1,28,28)).transpose((0,2,3,1)),\
                                    label:shuffle_train_labels[batch*batch_num:batch*batch_num+batch]})


    log = sess.run(summary)
    writer.add_summary(log,epoch+1)
    print('---loss:',sess.run(loss,feed_dict={data:shuffle_train_set[batch*batch_num:batch*batch_num+batch,:].reshape((-1,1,28,28)).transpose((0,2,3,1)),\
                                    label:shuffle_train_labels[batch*batch_num:batch*batch_num+batch]}))

# save parameter
saver.save(sess,'./net_parameters')