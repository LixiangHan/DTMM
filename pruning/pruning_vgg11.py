import tensorflow as tf
from kerassurgeon import Surgeon
import sys
import os
import random
import numpy as np
from beautifultable import BeautifulTable
from types import MethodType
from tensorflow.keras.callbacks import LearningRateScheduler
import time

from calculate_latency import calculate_latency
from solve_sa import solve_sa
from pruning_utils import *
from parse_args import parse_args
from calculate_model_size import calculate_model_size
from calculate_peak_memory import calculate_peak_memory
import graph

sys.path.insert(0, '/root/codes/PTMM')

from utils import config_device, get_train_and_test_generator, convert_to_tflite
from predictor import Predictor, load_predictor


if __name__ == '__main__':
    args = parse_args()
    # config_device()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    # global pred
    pred = load_predictor('../predictor.pkl')
    graph.init_graph('vgg11')
    GRAPH = graph.get_graph()
    print(graph.get_graph)
    if len(tf.config.list_physical_devices('GPU')) == 0:
        print('---------- No GPU ----------')
    random.seed(0)
    np.random.seed(0)
    
    if args.dataset == 'cifar10':
        image_size = 32
    elif args.dataset == 'vww':
        image_size = 64
    
    train_generator, val_generator, test_generator, train_steps_per_epoch, val_steps_per_epoch, test_steps_per_epoch = get_train_and_test_generator(input_size=image_size, batch_size=32, dataset=args.dataset, val=0.2)

    model = tf.keras.models.load_model('../weights/pretrain/vgg11_%s.h5' % args.dataset)

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
        if node['type'] == 'conv':
            filter_filepath = os.path.join('../importance/vgg11_%s' % args.dataset,
                                           '%s_filter.npy' % node['name'])
            filterlet_filepath = os.path.join('../importance/vgg11_%s' % args.dataset,
                                              '%s_filterlet.npy' % node['name'])
            filter_importance.append(np.load(filter_filepath))
            filterlet_importance.append(np.load(filterlet_filepath))

    alpha_final, beta_final = solve_sa(np.zeros(10),
                                       np.zeros(10),
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
        if node['type'] == 'conv':
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
        if node['type'] == 'conv':
            name = node['name']
            delete_index = list(
                np.where(filter_mask[name][0, 0, 0, :] == 1)[0])
            filter_index[name] = delete_index

    surgeon = Surgeon(model)
    for node in GRAPH:
        if node['type'] == 'conv':
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
        filepath='../weights/ptmm/vgg11_%s_%s/model' % (args.dataset, time_stamp),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        # optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=args.lr, decay=0.0005, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    history = model.fit(train_generator,
              epochs=args.epochs,
              steps_per_epoch=train_steps_per_epoch,
              validation_data=test_generator,
              workers=4,
              callbacks=[model_checkpoint_callback, lr_scheduler_callback])
    
    np.savetxt('retraining_loss.txt', history.history['loss'])
    
    convert_to_tflite(
        model, args.dataset, image_size,
        '../weights/ptmm/vgg11_%s_%s/model.tflite' % (args.dataset, time_stamp))

    os.system(
        'python ../generate_compressed_tflite.py ../weights/ptmm/vgg11_%s_%s/model.tflite -o ../weights/ptmm/vgg11_%s_%s/compressed_model.tflite -c ../weights/ptmm/vgg11_%s_%s/compressed_model.h'
        % (args.dataset, time_stamp, args.dataset, time_stamp, args.dataset, time_stamp))
    print(os.path.getsize('../weights/ptmm/vgg11_%s_%s/compressed_model.tflite' % (args.dataset, time_stamp)) / 1024, 'KB')
