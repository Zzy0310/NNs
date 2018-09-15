import tensorflow as tf

def Affine(layer_name,input,input_size,output_size,activation_function = None,keep_rate = 1):

    with tf.name_scope(layer_name):
        with tf.name_scope('W'):
            W = tf.Variable(tf.random_normal((input_size,output_size)),name = 'W')
            tf.summary.histogram(layer_name+"'s W",W)
        with tf.name_scope('b'):
            b = tf.Variable(tf.zeros((1,output_size)),name = 'b')
            tf.summary.histogram(layer_name+"'s b",b)
        with tf.name_scope('Dropout'):
            dropout = tf.nn.dropout(input,keep_rate)
        with tf.name_scope('Affine'):
            affine = tf.add(tf.matmul(dropout,W),b)
        with tf.name_scope('Activation_function'):
            if activation_function == None:
                return affine
            else:
                return activation_function(affine)