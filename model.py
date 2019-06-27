import tensorflow as tf

from abc import abstractmethod
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Conv2D, UpSampling2D, Concatenate, Add
from tensorflow.python.keras.layers.advanced_activations import PReLU


class BaseModel(object):

    def __new__(cls, input_shape):
        inputs = Input(shape=input_shape)
        outputs = cls.initialize(cls, inputs)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    @abstractmethod
    def initialize(self, inputs):
        raise NotImplementedError("Error: function initialize must be implemented.")


class DefaulModel(BaseModel):

    def initialize(self, inputs):

        init_conv = Conv2D(3, (3, 3), activation='relu', padding='same',
                           input_shape=inputs.shape)(inputs)
        upsampling_1 = UpSampling2D()(init_conv)

        conv_1 = self.conv_block(upsampling_1, 3, (3, 3))
        upsampling_2 = UpSampling2D()(conv_1)

        conv_2 = self.conv_block(upsampling_2, 3, (3, 3))
        upsampling_3 = UpSampling2D()(conv_2)

        conv_3 = self.conv_block(upsampling_3, 3, (3, 3))

        return conv_3

    @staticmethod
    def conv_block(input, filters, kernel_size):
        with tf.name_scope("Conv2D_Relu"):
            conv2d_relu = Conv2D(filters, kernel_size, activation='relu', padding='same')(input)

        return conv2d_relu


class DCSCNModel(BaseModel):

    def initialize(self, inputs):
        filter_set_1 = self.conv_prelu(inputs, 96, (3, 3))
        filter_set_2 = self.conv_prelu(filter_set_1, 76, (3, 3))
        filter_set_3 = self.conv_prelu(filter_set_2, 65, (3, 3))
        filter_set_4 = self.conv_prelu(filter_set_3, 55, (3, 3))
        filter_set_5 = self.conv_prelu(filter_set_4, 47, (3, 3))
        filter_set_6 = self.conv_prelu(filter_set_5, 39, (3, 3))
        filter_set_7 = self.conv_prelu(filter_set_6, 32, (3, 3))

        concat = Concatenate()([filter_set_1, filter_set_2, filter_set_3, filter_set_4,
                                filter_set_5, filter_set_6, filter_set_7])

        a1 = self.conv_prelu(concat, 64, (1, 1))
        b1 = self.conv_prelu(concat, 32, (1, 1))
        b2 = self.conv_prelu(b1, 32, (1, 1))

        concat_2 = Concatenate()([a1, b2])

        l = Conv2D(3, (1, 1), padding='same')(concat_2)

        upsampling = UpSampling2D(interpolation='bilinear')(inputs)

        add = Add()([upsampling, l])

        return add

    @staticmethod
    def conv_prelu(input, filters, kernel_size):
        with tf.name_scope("Conv2D_PRelu"):
            conv2d = Conv2D(filters, kernel_size, padding='same')(input)
            prelu = PReLU()(conv2d)

        return prelu
