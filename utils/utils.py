from models import *
from dataset import *
import tensorflow as tf
from matplotlib import pyplot as plt

import sys

sys.path.insert(0, '/pruneprune')


def plot_training_curve(history, save_path):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    lns1 = ax1.plot(history['loss'], 'r-', label='Train Loss')
    ax2 = ax1.twinx()
    lns3 = ax2.plot(history['accuracy'], 'b-', label='Train Acc')

    lns = lns1 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.grid()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    plt.savefig(save_path)


def get_train_and_test_generator(input_size, batch_size, dataset, val=0.0):
    if dataset == 'cifar10':
        train_generator, val_generator, test_generator, train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch = get_cifar10_train_and_test_generator(
            '/root/data/cifar10', batch_size, val)
    elif dataset == 'gtsrb':
        train_generator, val_generator, test_generator, train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch = get_gtsrb_train_and_test_generator(
            '/root/data/gtsrb', batch_size, input_size, val)
    elif dataset == 'vww':
        train_generator, val_generator, test_generator, train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch = get_vww_train_and_test_generator(
            '/root/data/vww', input_size, batch_size, val)
    elif dataset == 'camera_catalogue':
        train_generator, val_generator, test_generator, train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch = get_camera_catalogue_train_and_test_generator(
            '/root/data/camera_catalogue', input_size,
            batch_size, val)

    return train_generator, val_generator, test_generator, train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch


def get_model(model_name, dataset, input_size):
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'gtsrb':
        num_classes = 43
    elif dataset == 'vww' or dataset == 'camera_catalogue':
        num_classes = 2

    if model_name == 'resnet12':
        model = resnet12((input_size, input_size, 3), num_classes)
    elif model_name == 'squeezenet':
        model = squeezenet((input_size, input_size, 3), num_classes)
    elif model_name == 'vgg11':
        model = vgg11((input_size, input_size, 3), num_classes)

    return model


def config_device():
    if tf.test.is_gpu_available():
        print('GPU available')
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9 * 1024)])


def model_flops(model):
    flops = []
    for layer in model.layers:
        name = layer.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            strides = layer.strides
            ks = layer.kernel_size
            filters = layer.filters
            i_shape = layer.input.get_shape()[1:4].as_list()
            o_shape = layer.output.get_shape()[1:4].as_list()

            f = 2 * ((filters * ks[0] * ks[1] * i_shape[2]) * (
                    (i_shape[0] / strides[0]) * (i_shape[1] / strides[1])))
            print('%s: %d' % (name, f))
            flops.append(f)
    return flops


def convert_to_tflite(model, dataset, input_size, save_path):
    train_generator, val_generator, test_generator, train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch = get_train_and_test_generator(
        input_size, 1, dataset, val=0.2)

    def representative_dataset():
        for idx, (x, y) in enumerate(train_generator):
            if idx == 500:
                break
            yield [x]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    open(save_path, 'wb').write(tflite_quant_model)
