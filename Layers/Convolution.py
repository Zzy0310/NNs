import tensorflow as tf

def Convolution(layer_name,input,filter_height,filter_width,filter_channel,filter_num,strides_h=1,strides_v=1,\
                padding='VALID',activation_function = None):

    filter = tf.Variable(tf.random_normal((filter_height,filter_width,filter_channel,filter_num)))
    strides = [1,strides_h,strides_v,1]
    bais = tf.Variable(tf.zeros((filter_height,filter_width,filter_channel,filter_num)))

    with tf.name_scope(layer_name):
        with tf.name_scope('convolution'):
            conv = tf.nn.conv2d(input,filter,strides,padding)
            y = tf.add(conv,bais)
            tf.summary.histogram(layer_name+' filter',filter)
            tf.summary.histogram(layer_name + ' bais', bais)
        with tf.name_scope('Activation_function'):
            if activation_function == None:
                return conv
            else:
                return activation_function(conv)