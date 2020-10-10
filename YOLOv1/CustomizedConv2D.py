import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import numpy as np

class CustomizedConv2D(keras.layers.Conv2D):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,**kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        super(CustomizedConv2D, self).__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format= data_format,**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)  # 判断input_shape是否是list类型的
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[0][1], self.output_dim),  # input_shape应该长得像[(2,2),(3,3)]
        #                               initializer='uniform',
        #                               trainable=True)
        super(CustomizedConv2D, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        if(self.padding.lower() == "same"):
            # padding, 注意: 不用padding="SAME",否则可能会导致坐标计算错误
            pad_size = self.kernel_size // 2
            pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
            x = tf.pad(x, pad_mat)
        return super(CustomizedConv2D, self).call(x)

