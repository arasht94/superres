import tensorflow as tf

from abc import abstractmethod
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Conv2D, UpSampling2D, Concatenate, Add, Lambda, Activation
from tensorflow.python.keras.layers.advanced_activations import PReLU

from convweightnormalization import conv2d_weight_norm


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

        concat = Concatenate(axis=3)([filter_set_1, filter_set_2, filter_set_3, filter_set_4,
                                      filter_set_5, filter_set_6, filter_set_7])

        a1 = self.conv_prelu(concat, 64, (1, 1))
        b1 = self.conv_prelu(concat, 32, (1, 1))
        b2 = self.conv_prelu(b1, 32, (1, 1))

        concat_2 = Concatenate(axis=3)([a1, b2])

        input_depth = int(concat_2.shape[-1])
        scale = 8
        conv = self.conv_prelu(concat_2, scale * scale * input_depth, (3, 3))
        l = Lambda(lambda x: tf.depth_to_space(x, scale))(conv)
        l = Conv2D(3, (1, 1), padding='same')(l)

        upsampling = UpSampling2D(size=(8, 8), interpolation='bilinear')(inputs)

        add = Add()([upsampling, l])

        return add

    @staticmethod
    def conv_prelu(input, filters, kernel_size):
        with tf.name_scope("Conv2D_PRelu"):
            conv2d = Conv2D(filters, kernel_size, padding='same')(input)
            prelu = PReLU()(conv2d)

        return prelu

    
class WDSRModelA(BaseModel):
    
    def initialize(self, inputs):
        scale = 8
        num_filters = 32
        num_residual_blocks = 32
        res_block_expansion = 8
    
        m = conv2d_weight_norm(inputs, num_filters, 1, padding='valid')
        for i in range(num_residual_blocks):
            m = self.res_block_a(m, num_filters, res_block_expansion, kernel_size=1, scaling=None)
        print("Input M.shape: ", m.shape)
        m = conv2d_weight_norm(m, 3 * scale ** 2, 1, padding='valid')
        print("M.shape: ", m.shape)
        m = self.SubpixelConv2D(scale)(m)

        # skip branch
        s = conv2d_weight_norm(inputs, 3 * scale ** 2, 1, padding='valid')
        print("S.shape: ", s.shape)
        s = self.SubpixelConv2D(scale)(s)

        x = Add()([m, s])

        return x

    def res_block_a(x_in, num_filters, expansion, kernel_size, scaling):
        x = conv2d_weight_norm(x_in, num_filters * expansion, kernel_size, padding='same')
        x = Activation('relu')(x)
        x = conv2d_weight_norm(x, num_filters, kernel_size, padding='same')
        x = Add()([x_in, x])
        if scaling:
            x = Lambda(lambda t: t * scaling)(x)
        return x

    def SubpixelConv2D(scale, **kwargs):
        return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)

    
class WDSRModelB(BaseModel):
    
    def initialize(self, inputs):
        scale = 8
        num_filters = 32
        num_residual_blocks = 32
        res_block_expansion = 6

        # main branch (revise padding)
        m = conv2d_weight_norm(inputs, num_filters, 3, padding='valid')
        for i in range(num_residual_blocks):
            m = self.res_block_b(m, num_filters, res_block_expansion, kernel_size=3, scaling=None)
        m = Lambda(lambda x: tf.pad(x, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])))(m)
        m = conv2d_weight_norm(m, 3 * scale ** 2, 3, padding='same')
        m = self.SubpixelConv2D(scale)(m)

        # skip branch
#         s = Lambda(lambda x: tf.pad(x, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])))(inputs)
        s = conv2d_weight_norm(inputs, 3 * scale ** 2, 5, padding='same')
        s = self.SubpixelConv2D(scale)(s)

        x = Add()([m, s])
        print(x.shape)
        return x

    def res_block_b(x_in, num_filters, expansion, kernel_size, scaling):
        linear = 0.8
        x = conv2d_weight_norm(x_in, num_filters * expansion, 1, padding='same')
        x = Activation('relu')(x)
        x = conv2d_weight_norm(x, int(num_filters * linear), 1, padding='same')
        x = conv2d_weight_norm(x, num_filters, kernel_size, padding='same')
        x = Add()([x_in, x])
        if scaling:
            x = Lambda(lambda t: t * scaling)(x)
        return x

    def SubpixelConv2D(scale, **kwargs):
        return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)
