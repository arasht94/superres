import tensorflow as tf

from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Conv2D, UpSampling2D


class DefaulModel(object):

    def __new__(cls, input_shape):
        inputs = Input(shape=input_shape)

        init_conv = Conv2D(3, (3, 3), activation='relu', padding='same',
                           input_shape=input_shape)(inputs)
        upsampling_1 = UpSampling2D()(init_conv)

        conv_1 = cls.conv_block(upsampling_1, 3, (3, 3))
        upsampling_2 = UpSampling2D()(conv_1)

        conv_2 = cls.conv_block(upsampling_2, 3, (3, 3))
        upsampling_3 = UpSampling2D()(conv_2)

        conv_3 = cls.conv_block(upsampling_3, 3, (3, 3))

        model = Model(inputs=inputs, outputs=conv_3)
        return model

    @staticmethod
    def conv_block(input, filters, kernel_size):
        with tf.name_scope("Conv2D_Relu"):
            conv2d_relu = Conv2D(filters, kernel_size, activation='relu', padding='same')(input)

        return conv2d_relu