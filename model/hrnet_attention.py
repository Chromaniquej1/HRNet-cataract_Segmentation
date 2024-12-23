import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation, Add, Multiply, concatenate
from tensorflow.keras.models import Model

def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x

def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = conv3x3(input, out_filters, strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = Add()([x, residual])
    else:
        x = Add()([x, input])

    x = Activation('relu')(x)
    return x

def attention_block_2d(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, [1, 1])(x)
    phi_g = Conv2D(inter_channel, [1, 1])(g)
    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1])(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = Multiply()([x, rate])
    return att_x

def seg_hrnet_attention(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = conv3x3(inputs, 64)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = transition_layer([x], [32, 64])
    branches = [make_branch(x[i], out_filters) for i, out_filters in enumerate([32, 64])]
    x = fuse_layer(branches)

    x = transition_layer(x, [32, 64, 128])
    branches = [make_branch(x[i], out_filters) for i, out_filters in enumerate([32, 64, 128])]
    x = fuse_layer(branches)

    x = transition_layer(x, [32, 64, 128, 256])
    branches = [make_branch(x[i], out_filters) for i, out_filters in enumerate([32, 64, 128, 256])]
    x = fuse_layer(branches)

    upsampled_branches = [
        UpSampling2D(size=(2**i, 2**i))(branch) if i > 0 else branch
        for i, branch in enumerate(x)
    ]

    x = concatenate(upsampled_branches, axis=-1)
    x = Conv2D(num_classes, 1, activation='softmax')(x)

    model = Model(inputs, x)
    return model
