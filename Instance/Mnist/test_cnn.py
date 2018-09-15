import sys
import tensorflow as tf
sys.path.append('../..')
from Layers.Affine import *
from Layers.Convolution import *
from Layers.Pooling import *
from Dataset.Mnist.loader import *


# net struction
data = tf.placeholder(dtype=tf.float32)
label = tf.placeholder(dtype=tf.float32)

conv1 = Convolution('conv1',data,3,3,1,8,1,1,'VALID',tf.nn.relu)
pooling1 = Pooling('pooling1',conv1,2,2,2,2)
conv2 = Convolution('conv2',pooling1,5,5,8,16,1,1,'VALID',tf.nn.relu)
pooling2 = Pooling('pooling2',conv2,2,2,1,1)

fullconnection = tf.reshape(pooling2,(-1,16*8*8),'FullConnection')

affine1 = Affine('Affine1',fullconnection,16*8*8,100,tf.nn.relu,1)
affine2 = Affine('Affine2',affine1,100,10)
# net struction end

#loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = affine2,labels = label))

#saver
saver = tf.train.Saver()

# Session
sess = tf.Session()
####################    END     #########################

init = tf.global_variables_initializer()
sess.run(init)
saver.restore(sess,'./net_parameters')

test_images = Test_images_load('../../DataSet/Mnist/t10k-images.idx3-ubyte')
test_labels = Test_labels_load('../../DataSet/Mnist/t10k-labels.idx1-ubyte')
test_images = test_images.reshape((-1,1,28,28)).transpose((0,2,3,1))
print('Data Loaded!')

predict = sess.run(affine2,feed_dict={data:test_images,label:test_labels})

count = 0
for num in range(test_images.shape[0]):
    if np.argmax(predict[num]) == np.argmax(test_labels[num]):
        count += 1

print('accuracy:',count/test_images.shape[0])