import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
import numpy as np
import os 


def vgg11(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
                      kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
                      name='block_1_conv_1')(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=32,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
                      kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
                      name='block_1_conv_2')(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool2D(pool_size=2, strides=2, name='block_1_maxpool')(x)

    x = layers.Conv2D(filters=64,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
                      kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
                      name='block_2_conv_1')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=64,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
                      kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
                      name='block_2_conv_2')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool2D(pool_size=2, strides=2, name='block_2_maxpool')(x)

    x = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
                      kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
                      name='block_3_conv_1')(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
                      kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
                      name='block_3_conv_2')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=128,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
                      kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
                      name='block_3_conv_3')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool2D(pool_size=2, strides=2, name='block_3_maxpool')(x)

    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
                      kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
                      name='block_4_conv_1')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
                      kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
                      name='block_4_conv_2')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      kernel_initializer='he_normal',
                      # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
                      kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
                      name='block_4_conv_3')(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool2D(pool_size=2, strides=2, name='block_4_maxpool')(x)

    # x = layers.Conv2D(filters=num_classes,
    #                   kernel_size=1,
    #                   strides=1,
    #                   padding='same',
    #                   activation='relu',
    #                   kernel_initializer='he_normal',
    #                   kernel_regularizer=l1_l2(1e-7,1e-6),
    #                   name='conv_out')(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Activation('softmax')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=x)


if __name__ == '__main__':
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = vgg11((32, 32, 3), 10)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    x = np.random.randn(1, 32, 32, 3)
    y = model.predict(x)
    print(y.shape)
