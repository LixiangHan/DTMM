import tensorflow as tf
import os


def get_gtsrb_train_and_test_generator(root_path, batch_size, image_size=32, val=0.0):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        rescale=1. / 255,
        validation_split=val)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

    train_path = os.path.join(root_path, 'train')
    train_size = 39209
    train_generator = datagen.flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb',
        subset='training')
    val_generator = datagen.flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb',
        subset='validation')

    test_path = os.path.join(root_path, 'test')
    test_size = 12630
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='rgb')

    return train_generator, val_generator, test_generator, int(train_size * (1 - val)) // batch_size, int(train_size * val) // batch_size, test_size // batch_size
