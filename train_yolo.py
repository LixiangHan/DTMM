import tensorflow as tf
from matplotlib import pyplot as plt
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import argparse
import os
import numpy as np


from models import yolo, yolo_loss
from dataset import load_fddb_dataset, FDDBGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', '-lr', type=float)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--epochs', '-e', type=int)

    return parser.parse_args()


def plot_training_curve(history, time_stamp):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], c='r')
    plt.plot(history.history['val_loss'], c='b')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('images/yolo_loss_%s.png' % time_stamp)


def main():
    training_data_file = '/root/data/FDDB/VocFormat/train.txt'
    validation_data_file = '/root/data/FDDB/VocFormat/val.txt'
    args = parse_args()
    
    print(tf.test.is_gpu_available())
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    image_size = 112
    
    inputs = Input((image_size, image_size, 3))
    outputs = yolo(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss=yolo_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    X_train, Y_train = load_fddb_dataset(training_data_file)
    X_val, Y_val = load_fddb_dataset(validation_data_file)
    
    
    training_data_generator = FDDBGenerator(X_train, Y_train, batch_size, image_shape=(image_size, image_size), augmentation=True)
    validation_data_generator = FDDBGenerator(X_val, Y_val, batch_size, image_shape=(image_size, image_size))
    
    
    def lr_schedule(epoch):
        initial_learning_rate = learning_rate
        decay_per_epoch = 0.99
        lrate = initial_learning_rate * (decay_per_epoch**epoch)
        print('Learning rate = %f' % lrate)
        return lrate

    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath='weights/pretrain/yolo_%s.h5' % (time_stamp),
    #     # save_weights_only=True,
    #     monitor='val_loss',
    #     mode='min',
    #     save_best_only=True)
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True)
    
    history = model.fit_generator(
        training_data_generator,
        steps_per_epoch=len(training_data_generator),
        epochs=epochs,
        validation_data=validation_data_generator,
        validation_steps=len(validation_data_generator),
        # callbacks=[lr_scheduler_callback, model_checkpoint_callback],
        # callbacks=[lr_scheduler_callback, early_stopping_callback],
        callbacks=[lr_scheduler_callback],
        workers=8)
    
    np.savetxt('yolo_training_loss.txt', history.history['loss'])
    
    model.save('weights/pretrain/yolo_%s.h5' % (time_stamp))
    
    plot_training_curve(history, time_stamp)
    os.system('python convert2tflite_yolo.py --model_path weights/pretrain/yolo_%s.h5 --save_path weights/pretrain/yolo_%s.tflite' % (time_stamp, time_stamp))
    
if __name__ == '__main__':
    main()
    