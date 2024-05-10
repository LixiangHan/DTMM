import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.regularizers import l1_l2

import numpy as np

#define model
def resnet12(input_shape, num_classes, num_filters=32):
    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='conv_0')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x) # uncomment this for official resnet model

    # First stack

    # Weight layers
    y = Conv2D(num_filters,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='block_1_conv_1')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='block_1_conv_2')(y)
    y = BatchNormalization()(y)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    # Second stack

    # Weight layers
    num_filters *= 2  # Filters need to be double for each stack
    y = Conv2D(num_filters,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='block_2_conv_1')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='block_2_conv_2')(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
               kernel_size=1,
               strides=2,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='block_2_shortcut')(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    # Third stack

    # Weight layers
    num_filters *= 2
    y = Conv2D(num_filters,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='block_3_conv_1')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='block_3_conv_2')(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
               kernel_size=1,
               strides=2,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='block_3_shortcut')(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    # Fourth stack.
    
    # Weight layers
    num_filters *= 2
    y = Conv2D(num_filters,
               kernel_size=3,
               strides=2,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='block_4_conv_1')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='block_4_conv_2')(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
               kernel_size=1,
               strides=2,
               padding='same',
               kernel_initializer='he_normal',
               # kernel_regularizer=l1_l2(1e-7,1e-6), # vww
               kernel_regularizer=l1_l2(1e-5,1e-4), # cifar10
               name='block_4_shortcut')(x)

    # Overall residual, connect weight layer and identity paths
    x = tf.keras.layers.add([x, y])
    x = Activation('relu')(x)

    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    x = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(x)
    # x = tf.keras.layers.Conv2D(filters=num_classes,
    #                   kernel_size=1,
    #                   strides=1,
    #                   padding='same',
    #                   activation='relu',
    #                   kernel_initializer='he_normal',
    #                   kernel_regularizer=l1_l2(1e-7,1e-6),
    #                   name='conv_out')(x)
    # x = tf.keras.layers.BatchNormalization()(x)

    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # outputs = tf.keras.layers.Activation('softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    import numpy as np

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # model = ResNet12(num_classes=10, input_size=32)
    model = resnet12(input_shape=(32, 32, 3), num_classes=10)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    x = np.random.randn(1, 32, 32, 3)
    y = model.predict(x)
    print(y.shape)