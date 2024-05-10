import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import argparse
import time
from datetime import datetime

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--input_size', '-is', type=int)
    parser.add_argument('--batch_size', '-bs', type=int)
    # parser.add_argument('--learning_rate', '-lr', type=float)
    parser.add_argument('--epoch', '-e', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    # config_device()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    args = parse_args()

    train_generator, val_generator, test_generator, train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch = get_train_and_test_generator(
        args.input_size, args.batch_size, args.dataset, 0.2)

    model = get_model(args.model, args.dataset, args.input_size)
    
    def lr_schedule(epoch):
        initial_learning_rate = 1e-3
        decay_per_epoch = 0.99
        lrate = initial_learning_rate * (decay_per_epoch**epoch)
        print('Learning rate = %f' % lrate)
        return lrate
    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='weights/pretrain/%s_%s_%d_%s.h5' % (args.model, args.dataset, args.input_size, time_stamp),
        # save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

    # model.compile(
    #     loss='categorical_crossentropy',
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
    #     # optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate),
    #     metrics=['accuracy'])
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        # optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate),
        metrics=['accuracy'])

    history = model.fit(train_generator,
                        epochs=args.epoch,
                        steps_per_epoch=train_steps_per_epoch,
                        validation_data=val_generator,
                        callbacks=[lr_scheduler, model_checkpoint_callback])
                        # callbacks=[lr_scheduler, earlystop_callback])
    # callbacks=[lr_scheduler, tensorboard_callback, earlystop_callback])
    # np.savetxt('%s_%s_training_loss.txt' % (args.model, args.dataset),
    #            np.array(history.history['loss']))

    _, test_acc = model.evaluate(test_generator)

    # time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    
    np.savetxt('training_loss.txt', history.history['loss'])
    plot_training_curve(
        history.history, 'images/%s_%s_%d_%s.png' %
        (args.model, args.dataset, args.input_size, time_stamp))
    
    if args.dataset == 'cifar10':
        image_size = 32
    elif args.dataset == 'vww':
        image_size = 64
    convert_to_tflite(
        model, args.dataset, image_size,
        'weights/pretrain/%s_%s_%d_%s.tflite' % (args.model, args.dataset, args.input_size, time_stamp))
    # model.save('weights/pretrain/%s_%s_%d_%s.h5' %
    #            (args.model, args.dataset, args.input_size, time_stamp))
