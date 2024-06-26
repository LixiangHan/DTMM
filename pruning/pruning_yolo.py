import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from kerassurgeon import Surgeon
import sys
import os
import random
import numpy as np
from beautifultable import BeautifulTable
from types import MethodType
from tensorflow.keras.callbacks import LearningRateScheduler
import time
from argparse import ArgumentParser

from calculate_latency import calculate_latency
from solve_sa import solve_sa
from pruning_utils import *
from calculate_model_size import calculate_model_size
from calculate_peak_memory import calculate_peak_memory
import graph

sys.path.insert(0, '/root/codes/PTMM')

from dataset import load_fddb_dataset, FDDBGenerator
from models import yolo, yolo_loss
from predictor import Predictor, load_predictor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--flash', type=int)
    parser.add_argument('--sram', type=int)
    parser.add_argument('--acc', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    training_data_file = '/root/data/FDDB/VocFormat/train.txt'
    validation_data_file = '/root/data/FDDB/VocFormat/val.txt'
    
    args = parse_args()
    # config_device()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    # global pred
    pred = load_predictor('../predictor.pkl')
    graph.init_graph('yolo')
    GRAPH = graph.get_graph()
    
    if len(tf.config.list_physical_devices('GPU')) == 0:
        print('---------- No GPU ----------')
    random.seed(0)
    np.random.seed(0)

    
    X_train, Y_train = load_fddb_dataset(training_data_file)
    X_val, Y_val = load_fddb_dataset(validation_data_file)
    
    
    training_data_generator = FDDBGenerator(X_train, Y_train, 32, image_shape=(112, 112), augmentation=True)
    validation_data_generator = FDDBGenerator(X_val, Y_val, 32, image_shape=(112, 112))

    inputs = Input((112, 112, 3))
    outputs = yolo(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights('../weights/pretrain/yolo.h5', by_name=True)

    conv_id = 0
    for i in range(len(GRAPH)):
        layer = model.get_layer(GRAPH[i]['name'])
        if GRAPH[i]['type'] == 'conv' or GRAPH[i]['type'] == 'fc':
            GRAPH[i]['weights_shape'] = list(layer.weights[0].shape)
            GRAPH[i]['input_shape'] = list(layer.input.shape[1:])
            GRAPH[i]['output_shape'] = list(layer.output.shape[1:])
        if GRAPH[i]['type'] == 'conv':
            strides = layer.strides
            kernel_size = layer.kernel_size
            filters = layer.filters
            flops = 2 * ((filters * kernel_size[0] * kernel_size[1] *
                          GRAPH[i]['input_shape'][2]) *
                         ((GRAPH[i]['input_shape'][0] / strides[0]) *
                          (GRAPH[i]['input_shape'][1] / strides[1])))
            GRAPH[i]['flops'] = flops
            GRAPH[i]['conv_id'] = conv_id
            conv_id += 1
    
    filter_importance = []
    filterlet_importance = []
    
    for node in GRAPH:
        if node['type'] == 'conv' and node['prune'] == True:
            filter_filepath = os.path.join('../importance/yolo',
                                           '%s_filter.npy' % node['name'])
            filterlet_filepath = os.path.join('../importance/yolo',
                                              '%s_filterlet.npy' % node['name'])
            filter_importance.append(np.load(filter_filepath))
            filterlet_importance.append(np.load(filterlet_filepath))

    alpha_final, beta_final = solve_sa(np.zeros(12),
                                       np.zeros(12),
                                       calculate_latency,
                                       filterlet_importance,
                                       filter_importance,
                                       flash_constrain=args.flash * 1024,
                                       sram_constrain=args.sram * 1024,
                                       accuracy_constrain=args.acc,
                                       predictor=pred,
                                       T_0=100,
                                       T_t=20,
                                       k=0.9,
                                       L=100)
    
    info = BeautifulTable(maxwidth=200)
    info.columns.header = ['layer', 'alpha', 'beta']

    for node in GRAPH:
        if node['type'] == 'conv' and node['prune'] == True:
            info.rows.append([
                node['name'], alpha_final[node['conv_id']],
                beta_final[node['conv_id']]
            ])
    print(info)
    print(calculate_model_size(alpha_final, beta_final) / 1024, 'KB')
    print(calculate_peak_memory(beta_final, 256 * 1024)[0] / 1024, 'KB')
    
    # =======================================================================================================
    mask, filter_mask, filterlet_mask = generate_mask(model, filter_importance, filterlet_importance, alpha_final, beta_final)
    set_weights_to_zero(model, mask)

    filter_index = {}
    for node in GRAPH:
        if node['type'] == 'conv' and node['prune'] == True:
            name = node['name']
            delete_index = list(
                np.where(filter_mask[name][0, 0, 0, :] == 1)[0])
            filter_index[name] = delete_index

    surgeon = Surgeon(model)
    for node in GRAPH:
        if node['type'] == 'conv' and node['prune'] == True:
            name = node['name']
            layer = model.get_layer(name)
            surgeon.add_job('delete_channels',
                            layer,
                            channels=filter_index[name])
    model = surgeon.operate()

    # =======================================================================================================
    mask = generate_mask_from_weights(model)

    def train_step(self, data):
        x, y = data
        sample_weight = None
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y,
                                      y_pred,
                                      sample_weight=sample_weight,
                                      regularization_losses=self.losses)

            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            for i, trainable_weight in enumerate(self.trainable_weights):
                if 'conv' in trainable_weight.name and 'kernel' in trainable_weight.name and 'conv_out' not in trainable_weight.name:
                    name = trainable_weight.name.split('/')[0]
                    gradients[i] = tf.multiply(
                        tf.cast(tf.constant(1 - mask[name]), tf.float32),
                        gradients[i])

            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_weights))

            self.compiled_metrics.update_state(y,
                                               y_pred,
                                               sample_weight=sample_weight)

            return {m.name: m.result() for m in self.metrics}

    model.train_step = MethodType(train_step, model)

    def lr_schedule(epoch):
        initial_learning_rate = args.lr
        decay_per_epoch = 0.99
        lrate = initial_learning_rate * (decay_per_epoch**epoch)
        print('Learning rate = %f' % lrate)
        return lrate

    lr_scheduler_callback = LearningRateScheduler(lr_schedule)
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='../weights/ptmm/yolo_%s/model' % (time_stamp),
        monitor='val_loss',
        mode='min',
        # save_weights_only=True,
        save_best_only=True)

    model.compile(
        loss=yolo_loss,
        optimizer='adam')

    model.fit(training_data_generator,
              epochs=args.epochs,
              steps_per_epoch=len(training_data_generator),
              validation_data=validation_data_generator,
              workers=4,
              callbacks=[model_checkpoint_callback, lr_scheduler_callback])
    
    os.system('python ../convert2tflite_yolo.py --model_path ../weights/ptmm/yolo_%s/model --save_path ../weights/ptmm/yolo_%s/model.tflite' % (time_stamp, time_stamp))

    os.system(
        'python ../generate_compressed_tflite.py ../weights/ptmm/yolo_%s/model.tflite -o ../weights/ptmm/yolo_%s/compressed_model.tflite -c ../weights/ptmm/yolo_%s/compressed_model.h'
        % (time_stamp, time_stamp, time_stamp))
    print(os.path.getsize('../weights/ptmm/yolo_%s/compressed_model.tflite' % (time_stamp)) / 1024, 'KB')
