import tensorflow as tf

def Pooling(layer_name,input,filter_height,filter_width,strides_h=1,strides_v=1,\
                padding='VALID'):

    filter_size = [1,filter_height, filter_width,1]
    strides = [1, strides_h, strides_v, 1]

    with tf.name_scope(layer_name):
        with tf.name_scope('pooling'):
            pooling = tf.nn.max_pool(input,filter_size,strides,padding)

    return pooling